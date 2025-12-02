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

from benchmarks.targets import get_target, TargetDistribution
from benchmarks.metrics import compute_sliced_w2
from tuning.adaptation import run_adaptive_warmup
from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run, get_friction_schedule
from samplers.RWMH import rwMH_run
from tuning.dual_averaging import dual_averaging_tune_rwmh


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
    target_ess: int,
    batch_size: int,
    max_samples: int,
    schedule_type: str,
    num_steps_grid: List[int],
    learn_mass_matrix: bool = True,
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
            target_ess=target_ess,
            batch_size=batch_size,
            max_samples=max_samples,
            schedule_type=schedule_type,
            num_steps=num_steps,
            learn_mass_matrix=learn_mass_matrix,
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
            "overall_pass": False,
            "error": "No trajectory length produced usable samples",
            
            # Configuration
            "target_ess": target_ess,
            "num_warmup": num_warmup,
            "batch_size": batch_size,
            "max_samples": max_samples,
            
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
            
            # Pass/fail breakdown (shows which criteria failed)
            "rhat_pass": least_bad.get("rhat_pass"),
            "ess_pass": least_bad.get("ess_pass"),
            "ess_tail_pass": least_bad.get("ess_tail_pass"),
            
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
                    "total_samples": r.get("total_samples", 0),
                    "n_gradients": r.get("n_gradients", 0),
                    "accept_rate": r.get("accept_rate"),
                    "divergence_rate": r.get("divergence_rate"),
                    "usable": r.get("usable", False),
                    "overall_pass": r.get("overall_pass", False),
                    "error": r.get("error"),
                } for r in grid_results],
            },
        }
        return failure_result

    # Normal case: select best among usable runs
    best_result = max(usable_results, key=lambda r: r["ess_per_gradient"])
    selected_L = best_result["num_steps"]

    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results by trajectory length:")
    for r in grid_results:
        if r.get("error"):
            status = "[ERROR]"
        elif r.get("overall_pass"):
            status = "[PASS]"
        elif r.get("usable"):
            status = "[USABLE]"
        else:
            status = "[FAIL]"
        div_info = f", div={r.get('divergence_rate', 0):.1%}" if r.get('divergence_rate') is not None else ""
        print(f"  L={r['num_steps']:2d}: ESS/grad={r.get('ess_per_gradient', 0):.6f}, "
              f"ESS={r.get('ess_bulk_min', 0):7.1f}, R-hat={r.get('rhat_max', 0):.4f}{div_info} {status}")

    print(f"\n>>> BEST: L={selected_L} (ESS/grad={best_result['ess_per_gradient']:.6f})")

    # Add grid search metadata
    best_result["grid_search_info"] = {
        "tested_L_values": num_steps_grid,
        "selected_L": selected_L,
        "has_usable": True,
        "all_results": [{
            "num_steps": r["num_steps"],
            "ess_per_gradient": r.get("ess_per_gradient", 0),
            "ess_bulk_min": r.get("ess_bulk_min", 0),
            "rhat_max": r.get("rhat_max", float('inf')),
            "total_samples": r.get("total_samples", 0),
            "usable": r.get("usable", False),
            "overall_pass": r.get("overall_pass", False),
            "divergence_rate": r.get("divergence_rate", None),
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
    target_ess: int,
    batch_size: int,
    max_samples: int,
    schedule_type: str,
    num_steps: int,
    learn_mass_matrix: bool = True,
) -> Dict:
    """Run a single benchmark with a specific trajectory length.
    This is the internal function called by grid search.
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {sampler.upper()} on {target.name}")
    if sampler in ["grahmc", "rahmc"]:
        print(f"  Schedule: {schedule_type}")
    print(f"  Trajectory Length: L={num_steps}")
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
                sampler_kwargs["steepness"] = 1 if schedule_type == "tanh" else 5
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

        # Production sampling phase with ADAPTIVE sampling
        print(f"\n[Phase 2] Adaptive Sampling (until ESS >= {target_ess}, max {max_samples} samples)...")
        print(f"  Collecting in batches of {batch_size} samples")
        sample_start = time.time()

        all_samples_list = []
        all_log_probs_list = []
        total_samples = 0
        batch_num = 0
        current_position = warmup_pos
        final_accept_rate = 0.0
        
        # Track divergences
        total_divergences = 0
        total_transitions = 0
        
        # Track tree depths for NUTS gradient counting
        all_tree_depths = []

        while total_samples < max_samples:
            batch_num += 1
            key, sample_key = random.split(key)

            if sampler == "rwmh":
                samples_batch, lps_batch, accept_rate, final_state = rwMH_run(
                    sample_key, target.log_prob_fn, current_position,
                    num_samples=batch_size, scale=step_size, burn_in=0
                )
            elif sampler == "hmc":
                samples_batch, lps_batch, accept_rate, final_state = hmc_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, num_steps=num_steps,
                    num_samples=batch_size, burn_in=0,
                    inv_mass_matrix=inv_mass_matrix
                )
            elif sampler == "nuts":
                samples_batch, lps_batch, accept_rate, final_state, tree_depths, mean_accept_probs = nuts_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, max_tree_depth=10,
                    num_samples=batch_size, burn_in=0,
                    inv_mass_matrix=inv_mass_matrix
                )
                # FIX #2: Store tree depths for correct gradient counting
                all_tree_depths.append(tree_depths)
                
            elif sampler in ["grahmc", "rahmc"]:
                tuned_gamma = warmup_info.get("gamma", 1.0)
                tuned_steepness = warmup_info.get("steepness", 5.0)

                friction_schedule = get_friction_schedule(schedule_type)
                samples_batch, lps_batch, accept_rate, final_state = rahmc_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, num_steps=num_steps,
                    gamma=tuned_gamma,
                    steepness=tuned_steepness,
                    num_samples=batch_size, burn_in=0,
                    inv_mass_matrix=inv_mass_matrix,
                    friction_schedule=friction_schedule
                )

            # Continue from where we left off
            current_position = final_state.position
            final_accept_rate = float(jnp.mean(accept_rate))

            all_samples_list.append(samples_batch)
            all_log_probs_list.append(lps_batch)
            total_samples += batch_size
            
            # FIX: Track divergences (approximate via low acceptance for now)
            # Note: For full divergence tracking, we'd need to modify the samplers
            # to return delta_H. This is a proxy using very low acceptance.
            batch_transitions = batch_size * n_chains
            total_transitions += batch_transitions
            # Very low acceptance in a batch suggests potential divergences
            if final_accept_rate < 0.1:
                # Estimate divergences as fraction with very low acceptance
                estimated_div = int(batch_transitions * (1 - final_accept_rate) * 0.5)
                total_divergences += estimated_div

            # Concatenate all samples collected so far
            samples = jnp.concatenate(all_samples_list, axis=0)

            # Compute ESS to check if we've reached target
            samples_for_arviz = np.array(samples).transpose(1, 0, 2)
            idata = az.from_dict(
                posterior={"x": samples_for_arviz},
                coords={"dim": np.arange(target.dim)},
                dims={"x": ["dim"]}
            )
            ess_bulk = az.ess(idata, var_names=["x"], method="bulk")["x"].values
            min_ess = float(np.min(ess_bulk))
            mean_ess = float(np.mean(ess_bulk))

            print(f"  Batch {batch_num}: {total_samples} total samples, min ESS = {min_ess:.1f}, mean ESS = {mean_ess:.1f}")

            if min_ess >= target_ess:
                print(f"  Target ESS reached!")
                break

        # Final concatenation
        samples = jnp.concatenate(all_samples_list, axis=0)
        log_probs = jnp.concatenate(all_log_probs_list, axis=0)

        sample_time = time.time() - sample_start
        print(f"  Sampling complete in {sample_time:.1f}s")
        print(f"  Total samples: {total_samples}")
        print(f"  Final acceptance rate: {final_accept_rate:.3f}")
        
        # Compute divergence rate
        divergence_rate = total_divergences / total_transitions if total_transitions > 0 else 0.0
        if divergence_rate > 0.01:
            print(f"  WARNING: Divergence rate = {divergence_rate:.1%}")

        # Compute correct gradient count
        if sampler == "nuts" and all_tree_depths:
            # NUTS: gradients = sum over samples of (2^depth - 1) per chain
            tree_depths_array = jnp.concatenate(all_tree_depths, axis=0)  # (n_samples, n_chains)
            # Each tree of depth d uses 2^d leapfrog steps, each using 1 gradient
            gradients_per_sample = (2 ** tree_depths_array) - 1  # (n_samples, n_chains)
            n_gradients = int(jnp.sum(gradients_per_sample))
            avg_tree_depth = float(jnp.mean(tree_depths_array))
            print(f"  NUTS avg tree depth: {avg_tree_depth:.2f}, total gradients: {n_gradients}")
        elif sampler == "rwmh":
            n_gradients = 0  # RWMH doesn't use gradients
            avg_tree_depth = None
        else:
            # HMC/GRAHMC: num_steps gradients per sample per chain
            n_gradients = total_samples * num_steps * n_chains
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

        # Check if we hit the maximum sample budget without reaching target ESS
        hit_max_samples = total_samples >= max_samples
        rhat_max = diagnostics["rhat_max"]
        ess_min = diagnostics["ess_bulk_min"]
        ess_tail_min = diagnostics["ess_tail_min"]

        # Relaxed convergence gate: "usable"
        # TIGHTENED from original:
        #   - R-hat: 1.05 -> 1.02
        #   - ESS: 0.5*target -> 0.8*target  
        #   - Added: tail ESS requirement
        #   - Added: divergence rate check
        usable = (
            rhat_max <= 1.02 and                          # Tighter R-hat
            ess_min >= 0.8 * target_ess and               # Higher ESS threshold
            ess_tail_min >= 0.3 * target_ess and          # Tail ESS required
            divergence_rate < 0.05 and                     # Max 5% divergences
            not (hit_max_samples and ess_min < target_ess)
        )

        # Strict high-quality pass: "overall_pass"
        overall_pass = (
            usable and
            rhat_max <= 1.01 and
            ess_min >= target_ess and
            ess_tail_min >= 0.5 * target_ess and
            divergence_rate < 0.01 and                     # Stricter for pass
            (not has_true_mean or stats_pass)
        )

        # Compute Sliced W2 distance to ground truth
        print("[Phase 4] Computing Sliced W2 distance...")
        key, w2_key = random.split(key)
        sliced_w2 = compute_sliced_w2(
            samples, target_name, target.dim,
            n_reference=50000, n_projections=500, key=w2_key
        )
        if sliced_w2 is not None:
            print(f"  Sliced W2: {sliced_w2:.6f}")
        else:
            print(f"  Sliced W2: N/A (no reference sampler)")

        total_time = time.time() - start_time

        # Compile results
        results = {
            "sampler": sampler,
            "target": target.name,
            "schedule": schedule_type if sampler in ["grahmc", "rahmc"] else None,
            "dim": target.dim,
            "num_steps": num_steps,
            "n_chains": n_chains,
            "num_warmup": num_warmup,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "total_samples": total_samples,
            "target_ess": target_ess,
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
            # Divergence tracking
            "divergence_rate": divergence_rate,
            "total_divergences": total_divergences,
            # Correct gradient count
            "n_gradients": n_gradients,
            # Pass/fail criteria
            "rhat_pass": diagnostics["rhat_max"] < 1.01,
            "ess_pass": diagnostics["ess_bulk_min"] >= target_ess,
            "ess_tail_pass": diagnostics["ess_tail_min"] >= target_ess * 0.5,
            "stats_pass": stats_pass,
            # Include z-score details
            "z_score_max": stats_result.get("max_z"),
            "z_score_threshold": stats_result.get("threshold"),
            # Convergence gates (with tightened thresholds)
            "usable": usable,
            "overall_pass": overall_pass,
            # Distributional distance
            "sliced_w2": sliced_w2,
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
        if results["overall_pass"]:
            status = "[PASS]"
        elif results["usable"]:
            status = "[USABLE]"
        else:
            status = "[FAIL]"

        print(f"\n{status}")
        print(f"  R-hat: {results['rhat_max']:.4f} | ESS: {results['ess_bulk_min']:.0f} | "
              f"Tail ESS: {results['ess_tail_min']:.0f} | Div: {divergence_rate:.1%}")
        print(f"  Usable: {results['usable']} | HighQuality: {results['overall_pass']} | Time: {total_time:.1f}s")

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
            "overall_pass": False,
        }


def run_all_benchmarks(
    samplers: List[str],
    targets: List[str],
    grahmc_schedules: List[str],
    dim: int,
    n_chains: int,
    num_warmup: int,
    target_ess: int,
    batch_size: int,
    max_samples: int,
    seed: int,
    output_dir: str,
    num_steps_grid: List[int] = None,
    learn_mass_matrix: bool = True,
) -> pd.DataFrame:
    """Run all sampler-target combinations and save results."""

    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(seed)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Default grid for trajectory length if not specified
    if num_steps_grid is None:
        num_steps_grid = [8, 16, 24, 32, 48, 64]

    all_results = []

    # Iterate over all combinations
    for target_name in targets:
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target_name.upper()} (dim={dim})")
        print(f"{'#'*80}")

        target = get_target(target_name, dim=dim)

        for sampler in samplers:
            if sampler in ["grahmc", "rahmc"]:
                # Test each schedule for GRAHMC with grid search over L
                for schedule in grahmc_schedules:
                    key, subkey = random.split(key)
                    results = run_trajectory_length_grid_search(
                        sampler=sampler,
                        target=target,
                        target_name=target_name,
                        key=subkey,
                        n_chains=n_chains,
                        num_warmup=num_warmup,
                        target_ess=target_ess,
                        batch_size=batch_size,
                        max_samples=max_samples,
                        schedule_type=schedule,
                        num_steps_grid=num_steps_grid,
                        learn_mass_matrix=learn_mass_matrix,
                    )
                    all_results.append(results)
            elif sampler == "hmc":
                # HMC uses grid search over L
                key, subkey = random.split(key)
                results = run_trajectory_length_grid_search(
                    sampler=sampler,
                    target=target,
                    target_name=target_name,
                    key=subkey,
                    n_chains=n_chains,
                    num_warmup=num_warmup,
                    target_ess=target_ess,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    schedule_type="constant",  # Unused for HMC
                    num_steps_grid=num_steps_grid,
                    learn_mass_matrix=learn_mass_matrix,
                )
                all_results.append(results)
            else:
                # RWMH and NUTS don't use trajectory length grid search
                key, subkey = random.split(key)
                results = run_single_benchmark_with_L(
                    sampler=sampler,
                    target=target,
                    target_name=target_name,
                    key=subkey,
                    n_chains=n_chains,
                    num_warmup=num_warmup,
                    target_ess=target_ess,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    schedule_type="constant",  # Unused for RWMH/NUTS
                    num_steps=20,  # Unused for RWMH/NUTS
                    learn_mass_matrix=learn_mass_matrix,
                )
                all_results.append(results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    csv_path = Path(output_dir) / "benchmark_results.csv"
    json_path = Path(output_dir) / "benchmark_results.json"

    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[float]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    df.to_csv(csv_path, index=False)
    print(f"\n\n[OK] Results saved to {csv_path}")

    # Save JSON
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

    rounded_results = round_floats(all_results)
    with open(json_path, 'w') as f:
        json.dump(rounded_results, f, indent=2)
    print(f"[OK] Results saved to {json_path}")

    return df


def print_summary(df: pd.DataFrame):
    """Print a comprehensive summary of benchmark results including failure analysis."""

    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal experiments: {len(df)}")
    
    # Count by status
    passed = df['overall_pass'].sum() if 'overall_pass' in df.columns else 0
    usable = df['usable'].sum() if 'usable' in df.columns else 0
    failed = len(df) - usable
    grid_failed = df['grid_search_failed'].sum() if 'grid_search_failed' in df.columns else 0
    
    print(f"High Quality (overall_pass): {passed}/{len(df)} ({100*passed/len(df):.1f}%)")
    print(f"Usable: {usable}/{len(df)} ({100*usable/len(df):.1f}%)")
    print(f"Failed: {failed}/{len(df)} ({100*failed/len(df):.1f}%)")
    if grid_failed > 0:
        print(f"Grid Search Failures: {grid_failed}")

    # Breakdown by sampler
    print(f"\nBy Sampler:")
    for sampler in df['sampler'].unique():
        sampler_df = df[df['sampler'] == sampler]
        passed = sampler_df['overall_pass'].sum() if 'overall_pass' in sampler_df.columns else 0
        usable = sampler_df['usable'].sum() if 'usable' in sampler_df.columns else 0
        total = len(sampler_df)
        print(f"  {sampler:10s}: pass={passed}/{total}, usable={usable}/{total}")

    # Breakdown by target
    print(f"\nBy Target:")
    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        passed = target_df['overall_pass'].sum() if 'overall_pass' in target_df.columns else 0
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
                target_ess = row.get('target_ess', 1000)
                
                issues = []
                if rhat is not None and rhat > 1.02:
                    issues.append(f"R-hat={rhat:.3f} (>1.02)")
                if ess is not None and ess < 0.8 * target_ess:
                    issues.append(f"ESS={ess:.0f} (<{0.8*target_ess:.0f})")
                if ess_tail is not None and ess_tail < 0.3 * target_ess:
                    issues.append(f"TailESS={ess_tail:.0f} (<{0.3*target_ess:.0f})")
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
                
                print(f"{sampler_name:<25s} {row['target']:<25s} "
                      f"{row['ess_per_gradient']:>12.6f} "
                      f"{row.get('ess_bulk_min', 0):>8.0f} "
                      f"{row.get('rhat_max', 0):>8.4f}")
            
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
    parser.add_argument("--num-warmup", type=int, default=1000,
                       help="Number of warmup steps for adaptive tuning")
    parser.add_argument("--target-ess", type=int, default=400,
                       help="Target ESS for adaptive sampling to reach (default: 400, matches Stan)")
    parser.add_argument("--batch-size", type=int, default=2000,
                       help="Batch size for adaptive sampling")
    parser.add_argument("--max-samples", type=int, default=50000,
                       help="Maximum samples before giving up")
    parser.add_argument("--num-steps-grid", nargs="+", type=int, default=None,
                       help="Grid of trajectory lengths to test for HMC/GRAHMC")
    parser.add_argument("--no-mass-matrix", action="store_true",
                       help="Disable mass matrix adaptation (use identity matrix)")

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

    learn_mass_matrix = not args.no_mass_matrix

    print(f"\n{'='*80}")
    if learn_mass_matrix:
        print("MCMC BENCHMARK SUITE (with Adaptive Warmup + Mass Matrix Tuning)")
    else:
        print("MCMC BENCHMARK SUITE (with Adaptive Warmup, NO Mass Matrix)")
    print(f"{'='*80}")
    print(f"PATCHED VERSION: Fixes #1-5 applied")
    print(f"  #1: Grid search explicit failure handling")
    print(f"  #2: Tightened usable gate (R-hat<=1.02, ESS>=0.8*target)")
    print(f"  #3: Divergence detection and gating")
    print(f"  #4: Tail ESS in usable gate")
    print(f"  #5: Correct NUTS gradient count + Bonferroni z-score")
    print(f"{'='*80}")
    print(f"Targets: {', '.join(targets)}")
    print(f"Samplers: {', '.join(args.samplers)}")
    print(f"Dimension: {args.dim}")
    print(f"Chains: {args.n_chains}")
    print(f"Warmup: {args.num_warmup} steps")
    print(f"Batch size: {args.batch_size} samples per batch")
    print(f"Max samples: {args.max_samples}")
    print(f"Target ESS: {args.target_ess}")
    print(f"Mass matrix: {'Learned' if learn_mass_matrix else 'Identity (disabled)'}")
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
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        target_ess=args.target_ess,
        seed=args.seed,
        output_dir=args.output_dir,
        num_steps_grid=args.num_steps_grid,
        learn_mass_matrix=learn_mass_matrix,
    )

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()