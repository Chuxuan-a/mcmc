"""
Comprehensive benchmarking script with adaptive warmup (mass matrix tuning).

This script runs all sampler-target combinations using the new adaptive warmup
framework with mass matrix learning.

Usage:
    python run_benchmarks_v2.py --dim 20 --targets standard_normal ill_conditioned_gaussian
    python run_benchmarks_v2.py --dim 10 --all-targets --output-dir results
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, grad
import arviz as az

from benchmarks.targets import get_target, TargetDistribution
from tuning.adaptation import run_adaptive_warmup
from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run, get_friction_schedule
from samplers.RWMH import rwMH_run
from tuning.dual_averaging import dual_averaging_tune_rwmh


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
        "summary": summary,  # Add summary for z-score test
    }


def check_summary_statistics(diagnostics: Dict, target: TargetDistribution, z_threshold: float = 5.0) -> bool:
    """Check if estimated mean matches true values using z-score test.

    Uses Monte Carlo Standard Error (MCSE) to compute z-scores for mean estimates.
    A z-score > threshold indicates the estimate is inconsistent with the true value
    beyond what Monte Carlo noise would explain.

    Args:
        diagnostics: Dictionary containing summary statistics with 'summary' from ArviZ
        target: TargetDistribution with true mean/covariance
        z_threshold: Maximum acceptable z-score (default: 5.0, ~5-sigma test)

    Returns:
        True if z-score test passes, False otherwise
    """
    if target.true_mean is None or target.true_cov is None:
        return True  # Skip check if true values unknown

    summary = diagnostics["summary"]
    means = summary["mean"].values
    mcse = summary["mcse_mean"].values
    true_mean = np.array(target.true_mean)

    # Calculate Z-scores: (estimate - truth) / MCSE
    z_scores = (means - true_mean) / (mcse + 1e-16)
    max_z = np.max(np.abs(z_scores))

    # Pass if all z-scores are within threshold
    return max_z < z_threshold


def run_trajectory_length_grid_search(
    sampler: str,
    target: TargetDistribution,
    key: jnp.ndarray,
    n_chains: int,
    num_warmup: int,
    target_ess: int,
    batch_size: int,
    max_samples: int,
    schedule_type: str,
    num_steps_grid: List[int],
) -> Dict:
    """Run grid search over trajectory lengths and return best result.

    For HMC and GRAHMC, we test multiple trajectory lengths (L) and select
    the one that achieves the best ESS per gradient evaluation.

    Returns:
        Best result dict with additional 'grid_search_info' field
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
            key=subkey,
            n_chains=n_chains,
            num_warmup=num_warmup,
            target_ess=target_ess,
            batch_size=batch_size,
            max_samples=max_samples,
            schedule_type=schedule_type,
            num_steps=num_steps,
        )

        # Calculate ESS per gradient for ranking
        if result.get("error") is None:
            n_gradients = result["total_samples"] * num_steps
            ess_per_grad = result["ess_bulk_min"] / n_gradients if n_gradients > 0 else 0
            result["n_gradients"] = n_gradients
            result["ess_per_gradient"] = ess_per_grad
        else:
            result["n_gradients"] = 0
            result["ess_per_gradient"] = 0

        grid_results.append(result)

    # Select best result based on ESS per gradient
    best_result = max(grid_results, key=lambda r: r["ess_per_gradient"])

    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results by trajectory length:")
    for r in grid_results:
        status = "[ERROR]" if r.get("error") else ("[PASS]" if r.get("overall_pass") else "[FAIL]")
        print(f"  L={r['num_steps']:2d}: ESS/grad={r['ess_per_gradient']:.6f}, "
              f"ESS={r['ess_bulk_min']:7.1f}, samples={r['total_samples']:5d} {status}")

    print(f"\n>>> BEST: L={best_result['num_steps']} (ESS/grad={best_result['ess_per_gradient']:.6f})")

    # Add grid search metadata
    best_result["grid_search_info"] = {
        "tested_L_values": num_steps_grid,
        "all_results": [{
            "num_steps": r["num_steps"],
            "ess_per_gradient": r["ess_per_gradient"],
            "ess_bulk_min": r["ess_bulk_min"],
            "total_samples": r["total_samples"],
            "overall_pass": r.get("overall_pass", False),
        } for r in grid_results],
    }

    return best_result


def run_single_benchmark_with_L(
    sampler: str,
    target: TargetDistribution,
    key: jnp.ndarray,
    n_chains: int,
    num_warmup: int,
    target_ess: int,
    batch_size: int,
    max_samples: int,
    schedule_type: str,
    num_steps: int,
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
                sampler_kwargs["num_steps"] = num_steps  # Use grid search value
            elif sampler == "nuts":
                sampler_kwargs["max_tree_depth"] = 10
            elif sampler in ["grahmc", "rahmc"]:
                sampler_kwargs["num_steps"] = num_steps  # Use grid search value
                sampler_kwargs["gamma"] = 1.0
                sampler_kwargs["steepness"] = 5.0 if schedule_type == "tanh" else 10.0
                sampler_kwargs["friction_schedule"] = get_friction_schedule(schedule_type)

            sampler_name_map = {"grahmc": "grahmc", "rahmc": "grahmc"}
            step_size, inv_mass_matrix, warmup_pos, warmup_info = run_adaptive_warmup(
                sampler_name_map.get(sampler, sampler),
                target.log_prob_fn,
                grad_log_prob_fn,  # Use the gradient function we created above
                init_pos,
                key,
                num_warmup=num_warmup,
                target_accept=0.65,
                schedule_type=schedule_type if sampler in ["grahmc", "rahmc"] else None,
                **sampler_kwargs
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        warmup_time = time.time() - warmup_start
        print(f"  Warmup complete in {warmup_time:.1f}s")
        print(f"  Step size: {step_size:.4f}")
        if inv_mass_matrix is not None:
            print(f"  Mass matrix range: [{inv_mass_matrix.min():.4f}, {inv_mass_matrix.max():.4f}]")

        # Production sampling phase with ADAPTIVE sampling (like original benchmark)
        print(f"\n[Phase 2] Adaptive Sampling (until ESS >= {target_ess}, max {max_samples} samples)...")
        print(f"  Collecting in batches of {batch_size} samples")
        sample_start = time.time()

        all_samples_list = []
        all_log_probs_list = []
        total_samples = 0
        batch_num = 0
        current_position = warmup_pos
        final_accept_rate = 0.0

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
                    step_size=step_size, num_steps=num_steps,  # Use grid search value
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
            elif sampler in ["grahmc", "rahmc"]:
                # EXTRACT TUNED VALUES from warmup_info
                tuned_gamma = warmup_info.get("gamma", 1.0)
                tuned_steepness = warmup_info.get("steepness", 5.0)

                friction_schedule = get_friction_schedule(schedule_type)
                samples_batch, lps_batch, accept_rate, final_state = rahmc_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, num_steps=num_steps,  # Use grid search value
                    gamma=tuned_gamma,  # Use tuned values, not hardcoded!
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

        # Set metadata
        if sampler == "rwmh":
            sampler_metadata = {"scale": step_size}
        elif sampler == "hmc":
            sampler_metadata = {"step_size": step_size, "num_steps": num_steps}
        elif sampler == "nuts":
            sampler_metadata = {"step_size": step_size, "max_tree_depth": 10}
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
        stats_pass = check_summary_statistics(diagnostics, target, z_threshold=5.0)

        total_time = time.time() - start_time

        # Compile results
        results = {
            "sampler": sampler,
            "target": target.name,
            "schedule": schedule_type if sampler in ["grahmc", "rahmc"] else None,
            "dim": target.dim,
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
            # Pass/fail criteria
            "rhat_pass": diagnostics["rhat_max"] < 1.01,
            "ess_pass": diagnostics["ess_bulk_min"] >= target_ess,
            "ess_tail_pass": diagnostics["ess_tail_min"] >= target_ess * 0.5,
            "stats_pass": stats_pass,
            "overall_pass": (
                diagnostics["rhat_max"] < 1.01 and
                diagnostics["ess_bulk_min"] >= target_ess and
                diagnostics["ess_tail_min"] >= target_ess * 0.5 and
                stats_pass
            ),
        }

        # Add sampler-specific metadata
        results.update(sampler_metadata)

        # Add mass matrix info
        if inv_mass_matrix is not None:
            results["mass_matrix_min"] = float(inv_mass_matrix.min())
            results["mass_matrix_max"] = float(inv_mass_matrix.max())
            results["mass_matrix_mean"] = float(inv_mass_matrix.mean())

        status = "[PASS]" if results["overall_pass"] else "[FAIL]"
        print(f"\n{status}")
        print(f"  R-hat: {results['rhat_max']:.4f} | ESS: {results['ess_bulk_min']:.0f} | Time: {total_time:.1f}s")

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
            "error": str(e),
            "total_time": total_time,
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
) -> pd.DataFrame:
    """Run all sampler-target combinations and save results.

    Args:
        num_steps_grid: Grid of trajectory lengths to test for HMC/GRAHMC.
                       If None, uses default [8, 16, 32, 64].
    """

    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(seed)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Default grid for trajectory length if not specified
    if num_steps_grid is None:
        num_steps_grid = [8, 16, 32, 64]

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
                        key=subkey,
                        n_chains=n_chains,
                        num_warmup=num_warmup,
                        target_ess=target_ess,
                        batch_size=batch_size,
                        max_samples=max_samples,
                        schedule_type=schedule,
                        num_steps_grid=num_steps_grid,
                    )
                    all_results.append(results)
            elif sampler == "hmc":
                # HMC uses grid search over L
                key, subkey = random.split(key)
                results = run_trajectory_length_grid_search(
                    sampler=sampler,
                    target=target,
                    key=subkey,
                    n_chains=n_chains,
                    num_warmup=num_warmup,
                    target_ess=target_ess,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    schedule_type="constant",  # Unused for HMC
                    num_steps_grid=num_steps_grid,
                )
                all_results.append(results)
            else:
                # RWMH and NUTS don't use trajectory length grid search
                key, subkey = random.split(key)
                results = run_single_benchmark_with_L(
                    sampler=sampler,
                    target=target,
                    key=subkey,
                    n_chains=n_chains,
                    num_warmup=num_warmup,
                    target_ess=target_ess,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    schedule_type="constant",  # Unused for RWMH/NUTS
                    num_steps=20,  # Unused for RWMH/NUTS
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
    """Print a summary of benchmark results."""

    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal experiments: {len(df)}")
    passed = df['overall_pass'].sum()
    total = len(df)
    print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"Failed: {total - passed}")

    # Breakdown by sampler
    print(f"\nBy Sampler:")
    for sampler in df['sampler'].unique():
        sampler_df = df[df['sampler'] == sampler]
        passed = sampler_df['overall_pass'].sum()
        total = len(sampler_df)
        print(f"  {sampler:10s}: {passed}/{total} ({100*passed/total:.1f}%)")

    # Breakdown by target
    print(f"\nBy Target:")
    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        passed = target_df['overall_pass'].sum()
        total = len(target_df)
        print(f"  {target:30s}: {passed}/{total} ({100*passed/total:.1f}%)")


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
    parser.add_argument("--target-ess", type=int, default=1000,
                       help="Target ESS for adaptive sampling to reach")
    parser.add_argument("--batch-size", type=int, default=2000,
                       help="Batch size for adaptive sampling")
    parser.add_argument("--max-samples", type=int, default=50000,
                       help="Maximum samples before giving up")
    parser.add_argument("--num-steps-grid", nargs="+", type=int, default=None,
                       help="Grid of trajectory lengths to test for HMC/GRAHMC (default: [8, 16, 32, 64])")

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

    print(f"\n{'='*80}")
    print("MCMC BENCHMARK SUITE (with Adaptive Warmup + Mass Matrix Tuning)")
    print(f"{'='*80}")
    print(f"Targets: {', '.join(targets)}")
    print(f"Samplers: {', '.join(args.samplers)}")
    print(f"Dimension: {args.dim}")
    print(f"Chains: {args.n_chains}")
    print(f"Warmup: {args.num_warmup} steps")
    print(f"Batch size: {args.batch_size} samples per batch")
    print(f"Max samples: {args.max_samples}")
    print(f"Target ESS: {args.target_ess}")
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
    )

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
