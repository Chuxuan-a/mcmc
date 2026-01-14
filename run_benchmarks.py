"""
Comprehensive MCMC benchmarking script with adaptive warmup.

This script runs all sampler-target combinations using the adaptive warmup
framework. Mass matrix learning can be enabled (default) or disabled.

Usage:
    # With mass matrix learning (default)
    python run_benchmarks.py --dim 20 --targets standard_normal ill_conditioned_gaussian
    python run_benchmarks.py --dim 20 --all-targets --output-dir results_with_mass

    # Without mass matrix learning (use identity matrix)
    python run_benchmarks.py --dim 20 --all-targets --no-mass-matrix --output-dir results_no_mass --max-samples 100000
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import scipy.stats

import jax
import jax.numpy as jnp
from jax import random, grad
import arviz as az

from benchmarks.targets import get_target, TargetDistribution, get_reference_sampler
from benchmarks.metrics import compute_sliced_w2
from tuning.adaptation import run_adaptive_warmup
from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run, get_friction_schedule
from samplers.RWMH import rwMH_run
from tuning.dual_averaging import dual_averaging_tune_rwmh


# ============================================================================
# Quality Gate Constants
# ============================================================================
#
# Two-tier system for fixed-budget benchmarking:
# - Hard gate (usable): Minimum for comparative analysis
# - Quality pass: Gold standard, publishable results
#
# References:
# - Stan: R-hat < 1.01, ESS ≥ 400 (bulk), ESS ≥ 100 (tail)
# - Vehtari et al. (2021): Rank-normalized R-hat, ESS thresholds
# ============================================================================

# Fixed ESS thresholds (absolute)
MIN_ESS_HARD_GATE = 400        # Hard gate: bulk ESS minimum (Stan standard)
MIN_ESS_TAIL_HARD_GATE = 100   # Hard gate: tail ESS minimum (Stan standard)
MIN_ESS_QUALITY = 400          # Quality check: bulk ESS minimum
MIN_ESS_TAIL_QUALITY = 200     # Quality check: tail ESS (relaxed from 400, still stricter than Stan's 100)

# Efficiency thresholds (ESS/N ratios)
INEFFICIENT_THRESHOLD = 0.01   # Flag if ESS/N < 1%
HIGH_EFFICIENCY_THRESHOLD = 0.1  # Flag if ESS/N > 10%


def get_log_checkpoints(max_samples: int, base: float = 1.5) -> List[int]:
    """Generate log-spaced checkpoint sample counts for convergence tracking.

    Creates exponentially-spaced checkpoints to get more resolution early
    (when convergence is fastest) and less later (when changes are slower).

    Example for max_samples=10000, base=1.5:
        [100, 150, 225, 337, 506, 759, 1138, 1707, 2561, 3841, 5762, 8643, 10000]

    Args:
        max_samples: Maximum number of samples
        base: Multiplicative factor for log spacing (default: 1.5)

    Returns:
        List of checkpoint sample counts (roughly evenly spaced on log scale)
    """
    checkpoints = []
    current = 100  # Start at 100 samples minimum
    while current < max_samples:
        checkpoints.append(int(current))
        current *= base
    checkpoints.append(max_samples)  # Always include final count
    return checkpoints


def detect_divergences(delta_H: jnp.ndarray, threshold: float = 1000.0) -> jnp.ndarray:
    """Detect divergent transitions from Hamiltonian error.
    
    A divergence occurs when the Hamiltonian changes dramatically during
    integration, indicating numerical instability in the leapfrog integrator.
    
    Args:
        delta_H: Array of Hamiltonian changes (H_proposed - H_current)
        threshold: Maximum acceptable |delta_H| (default: 1000 nats)
        
    Returns:
        Boolean array where True indicates a divergent transition
    """
    return jnp.abs(delta_H) > threshold


# Corrected Z-score test with Bonferroni correction

def check_summary_statistics(
    diagnostics: Dict, 
    target: TargetDistribution, 
    significance: float = 0.05
) -> Dict:
    """Check if estimated mean matches true values using z-score test.

    Uses Monte Carlo Standard Error (MCSE) to compute z-scores for mean estimates.
    Applies Bonferroni correction for multiple testing across dimensions.
    
    FIX #5: Changed from fixed z_threshold=5.0 to Bonferroni-corrected threshold.
    For dim=20 at significance=0.05, threshold ≈ 3.29 (much stricter than 5.0).

    Args:
        diagnostics: Dictionary containing summary statistics with 'summary' from ArviZ
        target: TargetDistribution with true mean/covariance
        significance: Overall significance level (default: 0.05)

    Returns:
        Dict with 'pass' (bool), 'max_z' (float), 'threshold' (float), 'z_scores' (array)
    """
    if target.true_mean is None or target.true_cov is None:
        return {"pass": True, "max_z": 0.0, "threshold": None, "reason": "No ground truth"}

    summary = diagnostics["summary"]
    means = summary["mean"].values
    mcse = summary["mcse_mean"].values
    true_mean = np.array(target.true_mean)
    
    n_dim = len(means)
    
    # Bonferroni correction: divide significance by number of tests
    individual_alpha = significance / n_dim
    # Two-sided test: threshold for |z| > z_alpha/2
    z_threshold = scipy.stats.norm.ppf(1 - individual_alpha / 2)
    
    # Calculate Z-scores: (estimate - truth) / MCSE
    # Use relative epsilon to handle near-zero true means
    epsilon = 1e-8 * np.maximum(np.abs(true_mean), 1.0) + 1e-16
    z_scores = (means - true_mean) / (mcse + epsilon)
    max_z = float(np.max(np.abs(z_scores)))

    passed = max_z < z_threshold
    
    return {
        "pass": passed,
        "max_z": max_z,
        "threshold": z_threshold,
        "z_scores": z_scores,
        "reason": None if passed else f"max |z|={max_z:.2f} > {z_threshold:.2f}"
    }


def compute_diagnostics(samples: jnp.ndarray) -> Dict:
    """Compute convergence diagnostics and summary statistics."""
    n_samples, n_chains, n_dim = samples.shape

    # Convert to ArviZ InferenceData format: (chain, draw, *shape)
    samples_for_arviz = np.array(samples).transpose(1, 0, 2)

    idata = az.from_dict(
        posterior={"x": samples_for_arviz},
        coords={"dim": np.arange(n_dim)},
        dims={"x": ["dim"]}
    )

    # Compute split R-hat (rank-normalized)
    rhat = az.rhat(idata, var_names=["x"])
    rhat_values = rhat["x"].values

    # Compute ESS
    ess_bulk = az.ess(idata, var_names=["x"], method="bulk")["x"].values
    ess_tail = az.ess(idata, var_names=["x"], method="tail")["x"].values

    # Summary statistics (includes mean, mcse_mean, sd, etc.)
    summary = az.summary(idata, var_names=["x"])

    return {
        "rhat_max": float(np.max(rhat_values)),
        "rhat_mean": float(np.mean(rhat_values)),
        "ess_bulk_min": float(np.min(ess_bulk)),
        "ess_bulk_mean": float(np.mean(ess_bulk)),
        "ess_tail_min": float(np.min(ess_tail)),
        "ess_tail_mean": float(np.mean(ess_tail)),
        "mean_estimate": np.mean(samples_for_arviz, axis=(0, 1)),
        "std_estimate": np.std(samples_for_arviz, axis=(0, 1)),
        "summary": summary,
    }


def run_trajectory_length_grid_search(
    sampler: str,
    target: TargetDistribution,
    target_name: str,
    key: jnp.ndarray,
    n_chains: int,
    num_warmup: int,
    num_samples: int,
    schedule_type: str,
    num_steps_grid: List[int],
    learn_mass_matrix: bool = True,
    track_convergence: bool = False,
    convergence_base: float = 1.5,
) -> Dict:
    """Run grid search over trajectory lengths and return best result.

    For HMC and GRAHMC, we test multiple trajectory lengths (L) and select
    the one that achieves the best ESS per gradient evaluation.
    
    Now returns explicit failure result when no L produces usable samples,
    instead of picking the "best" among failed runs.

    Returns:
        Best result dict with additional 'grid_search_info' field,
        OR a failure result if no L values produce usable samples.
    """
    print(f"\n{'#'*80}")
    print(f"GRID SEARCH: Testing trajectory lengths {num_steps_grid}")
    print(f"{'#'*80}")

    grid_results = []

    for num_steps in num_steps_grid:
        key, subkey = random.split(key)

        print(f"\n--- Testing L={num_steps} ---")

        result = run_single_benchmark_with_L(
            sampler=sampler,
            target=target,
            target_name=target_name,
            key=subkey,
            n_chains=n_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            schedule_type=schedule_type,
            num_steps=num_steps,
            learn_mass_matrix=learn_mass_matrix,
            track_convergence=track_convergence,
            convergence_base=convergence_base,
        )

        # Calculate ESS per gradient for ranking
        # Use correct gradient count from result
        if result.get("error") is None:
            n_gradients = result.get("n_gradients", result["total_samples"] * num_steps)
            ess_per_grad = result["ess_bulk_min"] / n_gradients if n_gradients > 0 else 0
            result["n_gradients"] = n_gradients
            result["ess_per_gradient"] = ess_per_grad
        else:
            result["n_gradients"] = 0
            result["ess_per_gradient"] = 0

        grid_results.append(result)

    # Select best result based on ESS per gradient (among usable runs only)
    usable_results = [r for r in grid_results if r.get("usable", False)]

    # Return explicit failure when no L produces usable results
    if not usable_results:
        print("\n" + "!"*80)
        print("!!! GRID SEARCH FAILED: NO TRAJECTORY LENGTH PRODUCED USABLE RESULTS !!!")
        print("!"*80)
        
        # Find the "least bad" result for diagnostic info
        # Prioritize by: lowest rhat (convergence), then highest ESS
        def least_bad_score(r):
            if r.get("error"):
                return (float('inf'), 0)  # Errors are worst
            rhat = r.get("rhat_max", float('inf'))
            ess = r.get("ess_bulk_min", 0)
            return (rhat, -ess)  # Lower rhat is better, higher ESS is better
        
        least_bad = min(grid_results, key=least_bad_score)
        
        # Print diagnostic summary
        print(f"\nLeast-bad run: L={least_bad.get('num_steps')}")
        print(f"  R-hat: {least_bad.get('rhat_max', 'N/A'):.4f}" if least_bad.get('rhat_max') else "  R-hat: N/A")
        print(f"  ESS: {least_bad.get('ess_bulk_min', 'N/A'):.1f}" if least_bad.get('ess_bulk_min') else "  ESS: N/A")
        print(f"  Divergence rate: {least_bad.get('divergence_rate', 0):.1%}" if least_bad.get('divergence_rate') is not None else "  Divergence rate: N/A")
        
        # Build failure result with ALL diagnostic metrics from least-bad run
        # This allows post-hoc analysis of why the sampler failed
        failure_result = {
            # Core identification
            "sampler": sampler,
            "target": target.name,
            "schedule": schedule_type if sampler in ["grahmc", "rahmc"] else None,
            "dim": target.dim,
            "n_chains": n_chains,
            
            # Failure flags
            "grid_search_failed": True,
            "usable": False,
            "quality_pass": False,
            "error": "No trajectory length produced usable samples",
            
            # Configuration
            "num_samples": num_samples,
            "num_warmup": num_warmup,
            
            # Diagnostic metrics from least-bad run (for analysis)
            # These are clearly marked as coming from a failed run
            "num_steps": least_bad.get("num_steps"),  # The L that came closest
            "total_samples": least_bad.get("total_samples", 0),
            "n_gradients": least_bad.get("n_gradients", 0),
            
            # Convergence diagnostics (key for understanding failure)
            "rhat_max": least_bad.get("rhat_max"),
            "rhat_mean": least_bad.get("rhat_mean"),
            "ess_bulk_min": least_bad.get("ess_bulk_min", 0),
            "ess_bulk_mean": least_bad.get("ess_bulk_mean"),
            "ess_tail_min": least_bad.get("ess_tail_min"),
            "ess_tail_mean": least_bad.get("ess_tail_mean"),
            
            # Efficiency metrics (for comparison even when failed)
            "ess_per_sample": least_bad.get("ess_per_sample"),
            "ess_per_gradient": least_bad.get("ess_per_gradient", 0),
            
            # Stability diagnostics
            "divergence_rate": least_bad.get("divergence_rate"),
            "total_divergences": least_bad.get("total_divergences"),
            "accept_rate": least_bad.get("accept_rate"),
            
            # Timing (useful for understanding computational cost of failure)
            "warmup_time": least_bad.get("warmup_time"),
            "sample_time": least_bad.get("sample_time"),
            "total_time": least_bad.get("total_time"),
            
            # Quality metrics
            "sliced_w2": least_bad.get("sliced_w2"),
            "stats_pass": least_bad.get("stats_pass"),
            "z_score_max": least_bad.get("z_score_max"),
            "z_score_threshold": least_bad.get("z_score_threshold"),

            # Convergence tracking (None if not tracked)
            "convergence_trace": least_bad.get("convergence_trace"),
            
            # Pass/fail breakdown (shows which criteria failed)
            "rhat_pass": least_bad.get("rhat_pass"),
            "ess_pass": least_bad.get("ess_pass"),
            "ess_tail_pass": least_bad.get("ess_tail_pass"),

            # Efficiency flags
            "is_inefficient": least_bad.get("is_inefficient"),
            "is_high_efficiency": least_bad.get("is_high_efficiency"),
            
            # Sampler-specific params from least-bad run
            "step_size": least_bad.get("step_size"),
            "gamma": least_bad.get("gamma"),
            "steepness": least_bad.get("steepness"),
            "avg_tree_depth": least_bad.get("avg_tree_depth"),
            
            # Mass matrix info
            "mass_matrix_learned": least_bad.get("mass_matrix_learned"),
            "mass_matrix_min": least_bad.get("mass_matrix_min"),
            "mass_matrix_max": least_bad.get("mass_matrix_max"),
            "mass_matrix_mean": least_bad.get("mass_matrix_mean"),
            
            # Full grid search details for deep analysis
            "grid_search_info": {
                "tested_L_values": num_steps_grid,
                "selected_L": None,
                "has_usable": False,
                "least_bad_L": least_bad.get("num_steps"),
                "all_results": [{
                    "num_steps": r.get("num_steps"),
                    "ess_per_gradient": r.get("ess_per_gradient", 0),
                    "ess_bulk_min": r.get("ess_bulk_min", 0),
                    "ess_tail_min": r.get("ess_tail_min", 0),
                    "rhat_max": r.get("rhat_max", float('inf')),
                    "rhat_mean": r.get("rhat_mean", None),
                    "accept_rate": r.get("accept_rate", None),
                    "step_size": r.get("step_size", None),
                    "total_samples": r.get("total_samples", 0),
                    "n_gradients": r.get("n_gradients", 0),
                    "warmup_time": r.get("warmup_time", None),
                    "sample_time": r.get("sample_time", None),
                    "usable": r.get("usable", False),
                    "quality_pass": r.get("quality_pass", False),
                    "divergence_rate": r.get("divergence_rate", None),
                    "error": r.get("error"),
                    "sliced_w2": r.get("sliced_w2", None),
                    "z_score_max": r.get("z_score_max", None),
                    # GRAHMC-specific
                    "gamma": r.get("gamma", None),
                    "steepness": r.get("steepness", None),
                } for r in grid_results],
            },
        }
        return failure_result

    # Normal case: select best among usable runs
    # Prefer quality_pass runs over merely usable runs
    quality_results = [r for r in usable_results if r.get("quality_pass", False)]

    if quality_results:
        # Pick best quality_pass run
        best_result = max(quality_results, key=lambda r: r["ess_per_gradient"])
        selected_L = best_result["num_steps"]
        selection_tier = "quality_pass"
    else:
        # Fall back to best usable run
        best_result = max(usable_results, key=lambda r: r["ess_per_gradient"])
        selected_L = best_result["num_steps"]
        selection_tier = "usable_only"

    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results by trajectory length:")
    for r in grid_results:
        if r.get("error"):
            status = "[ERROR]"
        elif r.get("quality_pass"):
            status = "[PASS]"
        elif r.get("usable"):
            status = "[USABLE]"
        else:
            status = "[FAIL]"
        div_info = f", div={r.get('divergence_rate', 0):.1%}" if r.get('divergence_rate') is not None else ""
        print(f"  L={r['num_steps']:2d}: ESS/grad={r.get('ess_per_gradient', 0):.6f}, "
              f"ESS={r.get('ess_bulk_min', 0):7.1f}, R-hat={r.get('rhat_max', 0):.4f}{div_info} {status}")

    tier_label = "QUALITY_PASS" if selection_tier == "quality_pass" else "USABLE_ONLY"
    print(f"\n>>> BEST: L={selected_L} (ESS/grad={best_result['ess_per_gradient']:.6f}) [{tier_label}]")

    # Add grid search metadata
    best_result["grid_search_info"] = {
        "tested_L_values": num_steps_grid,
        "selected_L": selected_L,
        "selection_tier": selection_tier,
        "has_usable": True,
        "all_results": [{
            "num_steps": r["num_steps"],
            "ess_per_gradient": r.get("ess_per_gradient", 0),
            "ess_bulk_min": r.get("ess_bulk_min", 0),
            "ess_tail_min": r.get("ess_tail_min", 0),
            "rhat_max": r.get("rhat_max", float('inf')),
            "rhat_mean": r.get("rhat_mean", None),
            "accept_rate": r.get("accept_rate", None),
            "step_size": r.get("step_size", None),
            "total_samples": r.get("total_samples", 0),
            "n_gradients": r.get("n_gradients", 0),
            "warmup_time": r.get("warmup_time", None),
            "sample_time": r.get("sample_time", None),
            "usable": r.get("usable", False),
            "quality_pass": r.get("quality_pass", False),
            "divergence_rate": r.get("divergence_rate", None),
            "sliced_w2": r.get("sliced_w2", None),
            "z_score_max": r.get("z_score_max", None),
            # GRAHMC-specific
            "gamma": r.get("gamma", None),
            "steepness": r.get("steepness", None),
        } for r in grid_results],
    }

    return best_result


def run_single_benchmark_with_L(
    sampler: str,
    target: TargetDistribution,
    target_name: str,
    key: jnp.ndarray,
    n_chains: int,
    num_warmup: int,
    num_samples: int,
    schedule_type: str,
    num_steps: int,
    learn_mass_matrix: bool = True,
    track_convergence: bool = False,
    convergence_base: float = 1.5,
) -> Dict:
    """Run a single benchmark with a specific trajectory length.
    This is the internal function called by grid search.

    Args:
        track_convergence: If True, track W2 distance at log-spaced checkpoints
        convergence_base: Base for log spacing of checkpoints (default: 1.5)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {sampler.upper()} on {target.name}")
    if sampler in ["grahmc", "rahmc"]:
        print(f"  Schedule: {schedule_type}")
    print(f"  Trajectory Length: L={num_steps}")
    print(f"  Mass Matrix: {'Learned' if learn_mass_matrix else 'Identity'}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Initialize positions
        key, init_key = random.split(key)
        if target.init_sampler is not None:
            init_pos = target.init_sampler(init_key, n_chains)
        else:
            init_pos = random.normal(init_key, (n_chains, target.dim)) * 0.1

        # Create gradient function from log_prob_fn
        def grad_log_prob_fn(x):
            """Compute gradient of log probability at x."""
            return grad(lambda y: jnp.sum(target.log_prob_fn(y)))(x)

        # Adaptive warmup phase
        print("\n[Phase 1] Adaptive Warmup (tuning step size + mass matrix)...")
        warmup_start = time.time()

        if sampler == "rwmh":
            # RWMH uses old dual averaging (no mass matrix)
            tuned_scale, _ = dual_averaging_tune_rwmh(
                key, target.log_prob_fn, init_pos,
                target_accept=0.234, max_iter=1000
            )
            step_size = tuned_scale
            inv_mass_matrix = None
            warmup_pos = init_pos
            warmup_info = {"scale": tuned_scale}

        elif sampler in ["hmc", "nuts", "grahmc", "rahmc"]:
            # Use adaptive warmup for HMC/NUTS/GRAHMC
            sampler_kwargs = {}
            if sampler == "hmc":
                sampler_kwargs["num_steps"] = num_steps
            elif sampler == "nuts":
                sampler_kwargs["max_tree_depth"] = 15
            elif sampler in ["grahmc", "rahmc"]:
                sampler_kwargs["num_steps"] = num_steps
                sampler_kwargs["gamma"] = 1.0
                # Steepness from ablation study: lower is better (smoother transitions)
                sampler_kwargs["steepness"] = 0.5 if schedule_type == "tanh" else 2.0
                sampler_kwargs["friction_schedule"] = get_friction_schedule(schedule_type)

            sampler_name_map = {"grahmc": "grahmc", "rahmc": "grahmc"}
            step_size, inv_mass_matrix, warmup_pos, warmup_info = run_adaptive_warmup(
                sampler_name_map.get(sampler, sampler),
                target.log_prob_fn,
                grad_log_prob_fn,
                init_pos,
                key,
                num_warmup=num_warmup,
                target_accept=0.65,
                schedule_type=schedule_type if sampler in ["grahmc", "rahmc"] else None,
                learn_mass_matrix=learn_mass_matrix,
                **sampler_kwargs
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        warmup_time = time.time() - warmup_start
        print(f"  Warmup complete in {warmup_time:.1f}s")
        print(f"  Step size: {step_size:.4f}")
        if learn_mass_matrix and inv_mass_matrix is not None:
            print(f"  Mass matrix range: [{inv_mass_matrix.min():.4f}, {inv_mass_matrix.max():.4f}]")
        else:
            print(f"  Mass matrix: Identity (not learned)")

        # Production sampling phase
        if track_convergence and sampler in ["hmc", "grahmc", "rahmc"]:
            # Batch-based sampling with convergence tracking
            print(f"\n[Phase 2] Convergence Tracking Sampling ({num_samples} samples)...")
            checkpoints = get_log_checkpoints(num_samples, base=convergence_base)
            print(f"  Tracking at {len(checkpoints)} checkpoints: {checkpoints[:5]}{'...' if len(checkpoints) > 5 else ''}")

            sample_start = time.time()
            current_position = warmup_pos
            all_samples_list = []
            all_accept_rates = []
            convergence_trace = []

            for checkpoint_idx, checkpoint in enumerate(checkpoints):
                # Determine batch size
                if checkpoint_idx == 0:
                    batch_size = checkpoint
                    prev_samples = 0
                else:
                    prev_samples = checkpoints[checkpoint_idx - 1]
                    batch_size = checkpoint - prev_samples

                # Run sampler for this batch
                key, sample_key = random.split(key)

                if sampler == "hmc":
                    batch_samples, batch_log_probs, batch_accept_rate, final_state = hmc_run(
                        sample_key, target.log_prob_fn, current_position,
                        step_size=step_size, num_steps=num_steps,
                        num_samples=batch_size, burn_in=0,
                        inv_mass_matrix=inv_mass_matrix
                    )
                elif sampler in ["grahmc", "rahmc"]:
                    tuned_gamma = warmup_info.get("gamma", 1.0)
                    tuned_steepness = warmup_info.get("steepness", 5.0)
                    friction_schedule = get_friction_schedule(schedule_type)

                    batch_samples, batch_log_probs, batch_accept_rate, final_state = rahmc_run(
                        sample_key, target.log_prob_fn, current_position,
                        step_size=step_size, num_steps=num_steps,
                        gamma=tuned_gamma, steepness=tuned_steepness,
                        num_samples=batch_size, burn_in=0,
                        inv_mass_matrix=inv_mass_matrix,
                        friction_schedule=friction_schedule
                    )

                # Accumulate samples and accept rates
                all_samples_list.append(batch_samples)
                all_accept_rates.append(batch_accept_rate)
                current_position = final_state.position

                # Concatenate all samples so far
                cumulative_samples = jnp.concatenate(all_samples_list, axis=0)

                # Compute W2 on cumulative samples
                key, w2_key = random.split(key)
                w2_distance = compute_sliced_w2(
                    cumulative_samples, target_name, target.dim,
                    n_reference=50000, n_projections=500, key=w2_key
                )

                # Compute diagnostics on cumulative samples
                checkpoint_diagnostics = compute_diagnostics(cumulative_samples)

                # Calculate gradient count
                if sampler == "hmc":
                    n_gradients = checkpoint * num_steps * 2  # 2 grads per leapfrog step
                elif sampler in ["grahmc", "rahmc"]:
                    n_gradients = checkpoint * num_steps * 2  # Same as HMC

                # Store convergence data
                convergence_trace.append({
                    "checkpoint": int(checkpoint),
                    "n_gradients": int(n_gradients),
                    "w2_distance": float(w2_distance) if w2_distance is not None else None,
                    "ess_bulk_min": float(checkpoint_diagnostics["ess_bulk_min"]),
                    "ess_tail_min": float(checkpoint_diagnostics["ess_tail_min"]),
                    "rhat_max": float(checkpoint_diagnostics["rhat_max"]),
                })

                if (checkpoint_idx + 1) % 3 == 0 or checkpoint_idx == len(checkpoints) - 1:
                    w2_str = f"{w2_distance:.6f}" if w2_distance is not None else "N/A"
                    print(f"    Checkpoint {checkpoint}/{num_samples}: W2={w2_str}, ESS={checkpoint_diagnostics['ess_bulk_min']:.0f}")

            # Finalize
            samples = cumulative_samples
            accept_rate = jnp.concatenate(all_accept_rates, axis=0)
            all_tree_depths = None

        else:
            # Standard single-batch sampling (no convergence tracking)
            print(f"\n[Phase 2] Fixed Sampling ({num_samples} samples)...")
            sample_start = time.time()
            convergence_trace = None

            current_position = warmup_pos
            key, sample_key = random.split(key)

            if sampler == "rwmh":
                samples, log_probs, accept_rate, final_state = rwMH_run(
                    sample_key, target.log_prob_fn, current_position,
                    num_samples=num_samples, scale=step_size, burn_in=0
                )
                all_tree_depths = None

            elif sampler == "hmc":
                samples, log_probs, accept_rate, final_state = hmc_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, num_steps=num_steps,
                    num_samples=num_samples, burn_in=0,
                    inv_mass_matrix=inv_mass_matrix
                )
                all_tree_depths = None

            elif sampler == "nuts":
                samples, log_probs, accept_rate, final_state, tree_depths, mean_accept_probs = nuts_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, max_tree_depth=10,
                    num_samples=num_samples, burn_in=0,
                    inv_mass_matrix=inv_mass_matrix
                )
                all_tree_depths = tree_depths

            elif sampler in ["grahmc", "rahmc"]:
                tuned_gamma = warmup_info.get("gamma", 1.0)
                tuned_steepness = warmup_info.get("steepness", 5.0)
                friction_schedule = get_friction_schedule(schedule_type)

                samples, log_probs, accept_rate, final_state = rahmc_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, num_steps=num_steps,
                    gamma=tuned_gamma, steepness=tuned_steepness,
                    num_samples=num_samples, burn_in=0,
                    inv_mass_matrix=inv_mass_matrix,
                    friction_schedule=friction_schedule
                )
                all_tree_depths = None

        total_samples = num_samples
        final_accept_rate = float(jnp.mean(accept_rate))

        # Divergence tracking: currently not implemented in samplers
        # Would require sampler modifications to return delta_H
        total_divergences = 0
        total_transitions = num_samples * n_chains
        divergence_rate = 0.0  # Placeholder until samplers return delta_H

        sample_time = time.time() - sample_start
        print(f"  Sampling complete in {sample_time:.1f}s")
        print(f"  Collected {num_samples} samples")
        print(f"  Final acceptance rate: {final_accept_rate:.3f}")
        
        # Note: divergence_rate already set above, no warning needed for placeholder value

        # Compute correct gradient count
        if sampler == "nuts" and all_tree_depths is not None:
            # NUTS: gradients = sum over samples of (2^depth - 1) per chain
            # all_tree_depths shape: (num_samples, n_chains)
            # Each tree of depth d uses 2^d leapfrog steps, each using 1 gradient
            gradients_per_sample = (2 ** all_tree_depths) - 1  # (num_samples, n_chains)
            n_gradients = int(jnp.sum(gradients_per_sample))
            avg_tree_depth = float(jnp.mean(all_tree_depths))
            print(f"  NUTS avg tree depth: {avg_tree_depth:.2f}, total gradients: {n_gradients}")
        elif sampler == "rwmh":
            n_gradients = 0  # RWMH doesn't use gradients
            avg_tree_depth = None
        else:
            # HMC/GRAHMC: num_steps gradients per sample per chain
            n_gradients = num_samples * num_steps * n_chains
            avg_tree_depth = None

        # Set metadata
        if sampler == "rwmh":
            sampler_metadata = {"scale": step_size}
        elif sampler == "hmc":
            sampler_metadata = {"step_size": step_size, "num_steps": num_steps}
        elif sampler == "nuts":
            sampler_metadata = {
                "step_size": step_size, 
                "max_tree_depth": 10,
                "avg_tree_depth": avg_tree_depth
            }
        elif sampler in ["grahmc", "rahmc"]:
            sampler_metadata = {
                "step_size": step_size,
                "num_steps": num_steps,
                "gamma": warmup_info.get("gamma", 1.0),
                "steepness": warmup_info.get("steepness", 5.0),
                "schedule": schedule_type
            }

        # Compute diagnostics
        print("\n[Phase 3] Computing diagnostics...")
        diagnostics = compute_diagnostics(samples)
        
        # Use improved z-score test with Bonferroni correction
        stats_result = check_summary_statistics(diagnostics, target, significance=0.05)
        stats_pass = stats_result["pass"]
        if not stats_pass:
            print(f"  Z-score test FAILED: {stats_result['reason']}")

        # Determine if target has ground truth available
        has_true_mean = target.true_mean is not None and target.true_cov is not None

        # Extract diagnostics
        rhat_max = diagnostics["rhat_max"]
        ess_min = diagnostics["ess_bulk_min"]
        ess_tail_min = diagnostics["ess_tail_min"]

        # Compute efficiency metric
        ess_per_sample = ess_min / num_samples

        # Hard gate (usable): relaxed thresholds for comparative analysis
        usable = (
            rhat_max < 1.05 and                          # Relaxed R-hat
            ess_min >= MIN_ESS_HARD_GATE and             # Bulk ESS >= 400
            ess_tail_min >= MIN_ESS_TAIL_HARD_GATE and   # Tail ESS >= 100
            divergence_rate < 0.05                        # Max 5% divergences
        )

        # Quality check (only computed if usable)
        if usable:
            quality_pass = (
                rhat_max < 1.01 and                      # Strict R-hat
                ess_min >= MIN_ESS_QUALITY and           # Bulk ESS >= 400
                ess_tail_min >= MIN_ESS_TAIL_QUALITY and # Tail ESS >= 400
                divergence_rate < 0.01 and               # Stricter divergence threshold
                (not has_true_mean or stats_pass)
            )

            # Efficiency flags
            is_inefficient = (ess_per_sample < INEFFICIENT_THRESHOLD)
            is_high_efficiency = (ess_per_sample > HIGH_EFFICIENCY_THRESHOLD)
        else:
            quality_pass = False
            is_inefficient = False
            is_high_efficiency = False

        # Final overall_pass (same as quality_pass for now)
        overall_pass = quality_pass

        # Compute total time BEFORE diagnostics (W2) to exclude overhead
        total_time = time.time() - start_time

        # Compute Sliced W2 distance to ground truth (universal for targets with reference)
        # Check if reference sampler exists for this target
        ref_sampler = get_reference_sampler(target_name, target.dim)

        if ref_sampler is not None:
            # Reference sampler exists - compute W2
            print("[Phase 4] Computing Sliced W2 distance...")
            key, w2_key = random.split(key)
            sliced_w2 = compute_sliced_w2(
                samples, target_name, target.dim,
                n_reference=50000, n_projections=500, key=w2_key
            )
            if sliced_w2 is not None:
                print(f"  Sliced W2: {sliced_w2:.6f}")
            else:
                print(f"  Sliced W2: Computation failed")
        else:
            # No reference sampler available
            sliced_w2 = None
            print(f"[Phase 4] Sliced W2: N/A (no reference sampler for {target_name} {target.dim}D)")

        # Compile results
        results = {
            "sampler": sampler,
            "target": target.name,
            "schedule": schedule_type if sampler in ["grahmc", "rahmc"] else None,
            "dim": target.dim,
            "num_steps": num_steps if sampler in ["hmc", "grahmc", "rahmc"] else None,  # Only HMC/GRAHMC use fixed trajectory length
            "n_chains": n_chains,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "total_samples": total_samples,
            "warmup_time": warmup_time,
            "sample_time": sample_time,
            "total_time": total_time,
            "accept_rate": final_accept_rate,
            # Diagnostics
            "rhat_max": diagnostics["rhat_max"],
            "rhat_mean": diagnostics["rhat_mean"],
            "ess_bulk_min": diagnostics["ess_bulk_min"],
            "ess_bulk_mean": diagnostics["ess_bulk_mean"],
            "ess_tail_min": diagnostics["ess_tail_min"],
            "ess_tail_mean": diagnostics["ess_tail_mean"],
            # Efficiency metrics
            "ess_per_sample": ess_per_sample,
            "ess_per_gradient": ess_min / n_gradients if n_gradients > 0 else 0,
            # Divergence tracking
            "divergence_rate": divergence_rate,
            "total_divergences": total_divergences,
            # Gradient count
            "n_gradients": n_gradients,
            # Pass/fail criteria
            "rhat_pass": diagnostics["rhat_max"] < 1.01,
            "ess_pass": diagnostics["ess_bulk_min"] >= MIN_ESS_QUALITY,
            "ess_tail_pass": diagnostics["ess_tail_min"] >= MIN_ESS_TAIL_QUALITY,
            "stats_pass": stats_pass,
            # Include z-score details
            "z_score_max": stats_result.get("max_z"),
            "z_score_threshold": stats_result.get("threshold"),
            # Convergence gates
            "usable": usable,
            "quality_pass": quality_pass,
            # Efficiency flags
            "is_inefficient": is_inefficient,
            "is_high_efficiency": is_high_efficiency,
            # Distributional distance (None if not usable, except Neal's Funnel/Rosenbrock always computed)
            "sliced_w2": sliced_w2,
            # Convergence tracking (None if not tracked)
            "convergence_trace": convergence_trace if track_convergence else None,
        }

        # Add sampler-specific metadata
        results.update(sampler_metadata)

        # Add mass matrix info
        results["mass_matrix_learned"] = learn_mass_matrix
        if learn_mass_matrix and inv_mass_matrix is not None:
            results["mass_matrix_min"] = float(inv_mass_matrix.min())
            results["mass_matrix_max"] = float(inv_mass_matrix.max())
            results["mass_matrix_mean"] = float(inv_mass_matrix.mean())

        # Determine status display
        if results["quality_pass"]:
            status = "[PASS]"
        elif results["usable"]:
            status = "[USABLE]"
        else:
            status = "[FAIL]"

        print(f"\n{status}")
        print(f"  R-hat: {results['rhat_max']:.4f} | ESS: {results['ess_bulk_min']:.0f} | "
              f"Tail ESS: {results['ess_tail_min']:.0f} | Div: {divergence_rate:.1%}")

        # Quality gates info
        eff_flag = ""
        if is_high_efficiency:
            eff_flag = " [HIGH EFFICIENCY]"
        elif is_inefficient:
            eff_flag = " [INEFFICIENT]"

        print(f"  Usable: {results['usable']} | Quality: {results['quality_pass']} | "
              f"Efficiency: {ess_per_sample:.1%}{eff_flag}")
        print(f"  Time: {total_time:.1f}s")

        return results

    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "sampler": sampler,
            "target": target.name,
            "schedule": schedule_type if sampler in ["grahmc", "rahmc"] else None,
            "dim": target.dim,
            "num_steps": num_steps,
            "total_samples": 0,
            "ess_bulk_min": 0.0,
            "n_gradients": 0,
            "divergence_rate": None,
            "error": str(e),
            "total_time": total_time,
            "usable": False,
            "quality_pass": False,
        }


def save_result_incremental(result: Dict, output_dir: str, is_first: bool = False):
    """Save a single result to CSV and JSON files incrementally.

    Args:
        result: Result dictionary from a single benchmark run
        output_dir: Output directory path
        is_first: If True, create new files with headers; if False, append
    """
    csv_path = Path(output_dir) / "benchmark_results.csv"
    json_path = Path(output_dir) / "benchmark_results.json"

    # Helper to round floats in nested structures
    def round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 4)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, (int, str, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: round_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(x) for x in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    # Round result
    rounded_result = round_floats(result)

    # Prepare CSV-compatible version (deep copy to avoid mutating original)
    import copy
    csv_result = copy.deepcopy(rounded_result)

    # Convert nested dicts to JSON strings to avoid column expansion issues
    for key in ['grid_search_info', 'convergence_trace']:
        if key in csv_result and csv_result[key] is not None:
            csv_result[key] = json.dumps(csv_result[key])

    # Save to CSV
    df_row = pd.DataFrame([csv_result])

    if is_first:
        # Create new file with header - store column order for future appends
        df_row.to_csv(csv_path, index=False, mode='w')
        # Save column order to ensure consistency
        col_order_path = Path(output_dir) / ".csv_columns.json"
        with open(col_order_path, 'w') as f:
            json.dump(df_row.columns.tolist(), f)
    else:
        # Read expected column order
        col_order_path = Path(output_dir) / ".csv_columns.json"
        if col_order_path.exists():
            with open(col_order_path, 'r') as f:
                expected_cols = json.load(f)

            # Reindex DataFrame to match expected column order
            # Missing columns will be NaN, extra columns will be dropped
            df_row = df_row.reindex(columns=expected_cols)
        else:
            # Fallback: if column order file is missing, recreate it
            # This shouldn't happen in normal operation but provides safety
            print(f"WARNING: Column order file missing, recreating from current result")
            with open(col_order_path, 'w') as f:
                json.dump(df_row.columns.tolist(), f)

        # Append without header
        df_row.to_csv(csv_path, index=False, mode='a', header=False)

    # Save to JSON (read existing, append, write)
    if is_first:
        all_results_json = [rounded_result]
    else:
        # Load existing results
        if json_path.exists():
            with open(json_path, 'r') as f:
                all_results_json = json.load(f)
        else:
            all_results_json = []
        all_results_json.append(rounded_result)

    # Write updated JSON
    with open(json_path, 'w') as f:
        json.dump(all_results_json, f, indent=2)


def run_all_benchmarks(
    samplers: List[str],
    targets: List[str],
    grahmc_schedules: List[str],
    dim: int,
    n_chains: int,
    num_warmup: int,
    num_samples: int,
    seed: int,
    output_dir: str,
    num_steps_grid: List[int] = None,
    mass_matrix_modes: List[bool] = None,
    track_convergence: bool = False,
    convergence_base: float = 1.5,
) -> pd.DataFrame:
    """Run all sampler-target combinations and save results incrementally.

    Results are saved after each benchmark completes, so interrupted runs
    don't lose all progress.
    """

    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(seed)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Default grid for trajectory length if not specified
    if num_steps_grid is None:
        num_steps_grid = [8, 16, 24, 32, 48, 64, 96]

    # Check if resuming from existing results
    csv_path = Path(output_dir) / "benchmark_results.csv"
    json_path = Path(output_dir) / "benchmark_results.json"
    col_order_path = Path(output_dir) / ".csv_columns.json"

    if json_path.exists():
        # Resume mode: load existing results to skip already-completed runs
        print(f"\n{'='*80}")
        print(f"RESUMING FROM EXISTING RESULTS")
        print(f"{'='*80}")
        with open(json_path, 'r') as f:
            all_results = json.load(f)
        print(f"Found {len(all_results)} existing results")

        # Build set of completed (sampler, target, schedule, mass_matrix) tuples
        completed_runs = set()
        for r in all_results:
            run_signature = (
                r.get("sampler"),
                r.get("target"),
                r.get("schedule"),  # None for non-GRAHMC
                r.get("mass_matrix_learned"),
            )
            completed_runs.add(run_signature)

        print(f"Will skip {len(completed_runs)} already-completed configurations")
        is_first_result = False  # Append mode
        print(f"{'='*80}\n")
    else:
        # Fresh start
        all_results = []
        completed_runs = set()
        is_first_result = True

    # Iterate over all combinations
    for target_name in targets:
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target_name.upper()} (dim={dim})")
        print(f"{'#'*80}")

        target = get_target(target_name, dim=dim)

        for sampler in samplers:
            for learn_mass_matrix in mass_matrix_modes:
                if sampler in ["grahmc", "rahmc"]:
                    # Test each schedule for GRAHMC with grid search over L
                    for schedule in grahmc_schedules:
                        # Check if already completed
                        run_key = (sampler, target.name, schedule, learn_mass_matrix)
                        if run_key in completed_runs:
                            print(f"  [SKIP] {sampler}/{target.name}/{schedule}/mass={learn_mass_matrix} (already completed)")
                            continue

                        key, subkey = random.split(key)
                        results = run_trajectory_length_grid_search(
                            sampler=sampler,
                            target=target,
                            target_name=target_name,
                            key=subkey,
                            n_chains=n_chains,
                            num_warmup=num_warmup,
                            num_samples=num_samples,
                            schedule_type=schedule,
                            num_steps_grid=num_steps_grid,
                            learn_mass_matrix=learn_mass_matrix,
                            track_convergence=track_convergence,
                            convergence_base=convergence_base,
                        )
                        all_results.append(results)
                        # Save incrementally
                        save_result_incremental(results, output_dir, is_first=is_first_result)
                        is_first_result = False
                elif sampler == "hmc":
                    # HMC uses grid search over L
                    # Check if already completed
                    run_key = (sampler, target.name, None, learn_mass_matrix)
                    if run_key in completed_runs:
                        print(f"  [SKIP] {sampler}/{target.name}/mass={learn_mass_matrix} (already completed)")
                        continue

                    key, subkey = random.split(key)
                    results = run_trajectory_length_grid_search(
                        sampler=sampler,
                        target=target,
                        target_name=target_name,
                        key=subkey,
                        n_chains=n_chains,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        schedule_type="constant",  # Unused for HMC
                        num_steps_grid=num_steps_grid,
                        learn_mass_matrix=learn_mass_matrix,
                        track_convergence=track_convergence,
                        convergence_base=convergence_base,
                    )
                    all_results.append(results)
                    # Save incrementally
                    save_result_incremental(results, output_dir, is_first=is_first_result)
                    is_first_result = False
                else:
                    # RWMH and NUTS don't use trajectory length grid search
                    # Check if already completed
                    run_key = (sampler, target.name, None, learn_mass_matrix)
                    if run_key in completed_runs:
                        print(f"  [SKIP] {sampler}/{target.name}/mass={learn_mass_matrix} (already completed)")
                        continue

                    key, subkey = random.split(key)
                    results = run_single_benchmark_with_L(
                        sampler=sampler,
                        target=target,
                        target_name=target_name,
                        key=subkey,
                        n_chains=n_chains,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        schedule_type="constant",  # Unused for RWMH/NUTS
                        num_steps=20,  # Unused for RWMH/NUTS
                        learn_mass_matrix=learn_mass_matrix,
                    )
                    all_results.append(results)
                    # Save incrementally
                    save_result_incremental(results, output_dir, is_first=is_first_result)
                    is_first_result = False

    # Convert to DataFrame for return value and summary
    df = pd.DataFrame(all_results)

    # Results already saved incrementally during execution
    csv_path = Path(output_dir) / "benchmark_results.csv"
    json_path = Path(output_dir) / "benchmark_results.json"

    # Count new vs resumed results
    num_existing = len(completed_runs)
    num_new = len(all_results) - num_existing

    print(f"\n\n[OK] All results saved incrementally to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    if num_existing > 0:
        print(f"  Total experiments: {len(all_results)} ({num_existing} resumed, {num_new} new)")
    else:
        print(f"  Total experiments: {len(all_results)}")

    return df


def print_summary(df: pd.DataFrame):
    """Print a comprehensive summary of benchmark results including failure analysis."""

    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal experiments: {len(df)}")
    
    # Count by status
    passed = df['quality_pass'].sum() if 'quality_pass' in df.columns else 0
    usable = df['usable'].sum() if 'usable' in df.columns else 0
    failed = len(df) - usable
    grid_failed = df['grid_search_failed'].sum() if 'grid_search_failed' in df.columns else 0

    print(f"High Quality (quality_pass): {passed}/{len(df)} ({100*passed/len(df):.1f}%)")
    print(f"Usable: {usable}/{len(df)} ({100*usable/len(df):.1f}%)")
    print(f"Failed: {failed}/{len(df)} ({100*failed/len(df):.1f}%)")
    if grid_failed > 0:
        print(f"Grid Search Failures: {grid_failed}")

    # Efficiency flags
    if 'is_inefficient' in df.columns and 'is_high_efficiency' in df.columns:
        inefficient_count = df['is_inefficient'].sum()
        high_efficiency_count = df['is_high_efficiency'].sum()
        print(f"\nEfficiency Flags:")
        print(f"  Inefficient (ESS/N < 1%): {inefficient_count}/{len(df)}")
        print(f"  High Efficiency (ESS/N > 10%): {high_efficiency_count}/{len(df)}")

    # Breakdown by sampler
    print(f"\nBy Sampler:")
    for sampler in df['sampler'].unique():
        sampler_df = df[df['sampler'] == sampler]
        passed = sampler_df['quality_pass'].sum() if 'quality_pass' in sampler_df.columns else 0
        usable = sampler_df['usable'].sum() if 'usable' in sampler_df.columns else 0
        total = len(sampler_df)
        print(f"  {sampler:10s}: pass={passed}/{total}, usable={usable}/{total}")

    # Breakdown by target
    print(f"\nBy Target:")
    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        passed = target_df['quality_pass'].sum() if 'quality_pass' in target_df.columns else 0
        usable = target_df['usable'].sum() if 'usable' in target_df.columns else 0
        total = len(target_df)
        print(f"  {target:30s}: pass={passed}/{total}, usable={usable}/{total}")

    # Detailed failure analysis

    if 'grid_search_failed' in df.columns:
        failed_df = df[df['grid_search_failed'] == True]
        if len(failed_df) > 0:
            print(f"\n{'='*80}")
            print("FAILURE ANALYSIS (Grid Search Failures)")
            print(f"{'='*80}")
            print(f"\n{len(failed_df)} sampler-target combinations failed all trajectory lengths.\n")
            
            # Show each failure with diagnostics
            for _, row in failed_df.iterrows():
                sampler_name = row['sampler']
                if row.get('schedule'):
                    sampler_name = f"{sampler_name}-{row['schedule']}"
                
                print(f"{sampler_name:20s} on {row['target']:25s}")
                
                # Show why it failed (closest metrics)
                rhat = row.get('rhat_max')
                ess = row.get('ess_bulk_min')
                ess_tail = row.get('ess_tail_min')
                div_rate = row.get('divergence_rate')

                issues = []
                if rhat is not None and rhat > 1.05:
                    issues.append(f"R-hat={rhat:.3f} (>1.05)")
                if ess is not None and ess < MIN_ESS_HARD_GATE:
                    issues.append(f"ESS={ess:.0f} (<{MIN_ESS_HARD_GATE})")
                if ess_tail is not None and ess_tail < MIN_ESS_TAIL_HARD_GATE:
                    issues.append(f"TailESS={ess_tail:.0f} (<{MIN_ESS_TAIL_HARD_GATE})")
                if div_rate is not None and div_rate > 0.05:
                    issues.append(f"Div={div_rate:.1%} (>5%)")
                
                if issues:
                    print(f"  Issues: {', '.join(issues)}")
                else:
                    print(f"  Issues: Unknown (check detailed logs)")
                
                # Show the best L tried
                best_L = row.get('num_steps')
                if best_L:
                    print(f"  Best L tried: {best_L}")
                print()

    # =========================================================================
    # Efficiency ranking (usable runs only)
    # =========================================================================
    if 'usable' in df.columns and 'ess_per_gradient' in df.columns:
        usable_df = df[df['usable'] == True].copy()
        if len(usable_df) > 0:
            print(f"\n{'='*80}")
            print("EFFICIENCY RANKING (Usable Runs Only)")
            print(f"{'='*80}")
            
            # Sort by ESS per gradient
            usable_df = usable_df.sort_values('ess_per_gradient', ascending=False)
            
            print(f"\nTop 10 by ESS/Gradient:")
            print(f"{'Sampler':<25s} {'Target':<25s} {'ESS/Grad':>12s} {'ESS':>8s} {'R-hat':>8s}")
            print("-" * 80)
            
            for i, (_, row) in enumerate(usable_df.head(10).iterrows()):
                sampler_name = row['sampler']
                if row.get('schedule'):
                    sampler_name = f"{sampler_name}-{row['schedule']}"

                # Add efficiency flag
                eff_flag = ""
                if row.get('is_high_efficiency'):
                    eff_flag = " [HIGH EFF]"
                elif row.get('is_inefficient'):
                    eff_flag = " [INEFFICIENT]"

                print(f"{sampler_name:<25s} {row['target']:<25s} "
                      f"{row['ess_per_gradient']:>12.6f} "
                      f"{row.get('ess_bulk_min', 0):>8.0f} "
                      f"{row.get('rhat_max', 0):>8.4f}{eff_flag}")
            
            # Best sampler per target
            print(f"\nBest Sampler per Target:")
            print(f"{'Target':<30s} {'Best Sampler':<25s} {'ESS/Grad':>12s}")
            print("-" * 70)
            
            for target in usable_df['target'].unique():
                target_df = usable_df[usable_df['target'] == target]
                if len(target_df) > 0:
                    best = target_df.loc[target_df['ess_per_gradient'].idxmax()]
                    sampler_name = best['sampler']
                    if best.get('schedule'):
                        sampler_name = f"{sampler_name}-{best['schedule']}"
                    print(f"{target:<30s} {sampler_name:<25s} {best['ess_per_gradient']:>12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Run MCMC benchmarks with adaptive warmup")

    # Target selection
    parser.add_argument("--targets", nargs="+", default=None,
                       help="List of targets to benchmark")
    parser.add_argument("--all-targets", action="store_true",
                       help="Run all available targets")

    # Sampler selection
    parser.add_argument("--samplers", nargs="+",
                       default=["rwmh", "hmc", "nuts", "grahmc"],
                       help="List of samplers to benchmark")
    parser.add_argument("--schedules", nargs="+",
                       default=["constant", "tanh", "sigmoid", "linear", "sine"],
                       help="GRAHMC friction schedules to test")

    # Parameters
    parser.add_argument("--dim", type=int, default=10,
                       help="Dimensionality of targets")
    parser.add_argument("--n-chains", type=int, default=4,
                       help="Number of parallel chains")
    parser.add_argument("--num-warmup", type=int, default=2500,
                       help="Number of warmup steps (default: 500 exploration + 1875 adaptation + 125 cooldown)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Fixed number of samples to collect (default: 10k for dim>=20, 4k otherwise)")
    parser.add_argument("--num-steps-grid", nargs="+", type=int, default=None,
                       help="Grid of trajectory lengths to test for HMC/GRAHMC")
    parser.add_argument("--mass-matrix-mode", type=str,
                       choices=["mass", "no-mass", "both"], default="mass",
                       help="Mass matrix configuration: 'mass' (learn), 'no-mass' (identity), 'both' (run both)")

    # Convergence tracking
    parser.add_argument("--track-convergence", action="store_true",
                       help="Track W2 convergence at log-spaced checkpoints (adds ~30-75s overhead per run)")
    parser.add_argument("--convergence-base", type=float, default=1.5,
                       help="Log spacing base for convergence checkpoints (default: 1.5)")

    # Output
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Set up targets
    if args.all_targets:
        targets = ["standard_normal", "correlated_gaussian", "ill_conditioned_gaussian",
                  "student_t", "log_gamma", "rosenbrock", "neals_funnel", "gaussian_mixture"]
    elif args.targets:
        targets = args.targets
    else:
        print("Error: Must specify --targets or --all-targets")
        return

    # Parse mass matrix modes
    mass_matrix_modes = []
    if args.mass_matrix_mode == "mass":
        mass_matrix_modes = [True]
    elif args.mass_matrix_mode == "no-mass":
        mass_matrix_modes = [False]
    elif args.mass_matrix_mode == "both":
        mass_matrix_modes = [True, False]

    # Dimension-dependent default for num_samples
    if args.num_samples is None:
        if args.dim >= 20:
            num_samples = 10000
        else:
            num_samples = 10000
        print(f"Using default num_samples={num_samples} for dim={args.dim}")
    else:
        num_samples = args.num_samples

    print(f"\n{'='*80}")
    print("MCMC BENCHMARK SUITE (Fixed-Budget with Adaptive Warmup)")
    print(f"{'='*80}")
    print(f"FIXED-BUDGET BENCHMARKING")
    print(f"{'='*80}")
    print(f"Targets: {', '.join(targets)}")
    print(f"Samplers: {', '.join(args.samplers)}")
    print(f"Dimension: {args.dim}")
    print(f"Chains: {args.n_chains}")
    print(f"Warmup: {args.num_warmup} steps")
    print(f"Samples per run: {num_samples}")
    if len(mass_matrix_modes) == 2:
        print(f"Mass matrix: Both (learned + identity) - 2x runs per config")
    elif mass_matrix_modes[0]:
        print(f"Mass matrix: Learned")
    else:
        print(f"Mass matrix: Identity (disabled)")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Run benchmarks
    df = run_all_benchmarks(
        samplers=args.samplers,
        targets=targets,
        grahmc_schedules=args.schedules,
        dim=args.dim,
        n_chains=args.n_chains,
        num_warmup=args.num_warmup,
        num_samples=num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        num_steps_grid=args.num_steps_grid,
        mass_matrix_modes=mass_matrix_modes,
        track_convergence=args.track_convergence,
        convergence_base=args.convergence_base,
    )

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()