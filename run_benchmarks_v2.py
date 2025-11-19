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
from jax import random
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

    # Summary statistics
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
    }


def check_summary_statistics(diagnostics: Dict, target: TargetDistribution, tolerance: float = 0.15) -> bool:
    """Check if estimated mean and std match true values within tolerance."""
    if target.true_mean is None or target.true_cov is None:
        return True  # Skip check if true values unknown

    mean_est = diagnostics["mean_estimate"]
    std_est = diagnostics["std_estimate"]

    true_mean = np.array(target.true_mean)
    true_std = np.sqrt(np.diag(target.true_cov))

    # Check mean (scaled by true std)
    mean_error = np.abs(mean_est - true_mean) / (true_std + 1e-6)
    mean_pass = np.all(mean_error < tolerance)

    # Check std (relative error)
    std_error = np.abs(std_est - true_std) / (true_std + 1e-6)
    std_pass = np.all(std_error < tolerance)

    return mean_pass and std_pass


def run_single_benchmark(
    sampler: str,
    target: TargetDistribution,
    key: jnp.ndarray,
    n_chains: int = 4,
    num_warmup: int = 1000,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    schedule_type: str = "constant",
) -> Dict:
    """Run a single sampler-target benchmark with adaptive warmup."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK: {sampler.upper()} on {target.name}")
    if sampler in ["grahmc", "rahmc"]:
        print(f"  Schedule: {schedule_type}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Initialize positions
        key, init_key = random.split(key)
        if target.init_sampler is not None:
            init_pos = target.init_sampler(init_key, n_chains)
        else:
            init_pos = random.normal(init_key, (n_chains, target.dim)) * 0.1

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
                sampler_kwargs["num_steps"] = 20  # Fixed L for HMC
            elif sampler == "nuts":
                sampler_kwargs["max_tree_depth"] = 10
            elif sampler in ["grahmc", "rahmc"]:
                sampler_kwargs["num_steps"] = 20
                sampler_kwargs["gamma"] = 1.0
                sampler_kwargs["steepness"] = 5.0 if schedule_type == "tanh" else 10.0
                sampler_kwargs["friction_schedule"] = get_friction_schedule(schedule_type)

            sampler_name_map = {"grahmc": "grahmc", "rahmc": "grahmc"}
            step_size, inv_mass_matrix, warmup_pos, warmup_info = run_adaptive_warmup(
                key, sampler_name_map.get(sampler, sampler), target.log_prob_fn, init_pos,
                num_warmup=num_warmup, target_accept=0.65,
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
                    step_size=step_size, num_steps=20,
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
                friction_schedule = get_friction_schedule(schedule_type)
                samples_batch, lps_batch, accept_rate, final_state = rahmc_run(
                    sample_key, target.log_prob_fn, current_position,
                    step_size=step_size, num_steps=20,
                    gamma=1.0, steepness=5.0 if schedule_type == "tanh" else 10.0,
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
            sampler_metadata = {"step_size": step_size, "num_steps": 20}
        elif sampler == "nuts":
            sampler_metadata = {"step_size": step_size, "max_tree_depth": 10}
        elif sampler in ["grahmc", "rahmc"]:
            sampler_metadata = {"step_size": step_size, "num_steps": 20, "gamma": 1.0, "schedule": schedule_type}

        # Compute diagnostics
        print("\n[Phase 3] Computing diagnostics...")
        diagnostics = compute_diagnostics(samples)
        stats_pass = check_summary_statistics(diagnostics, target, tolerance=0.15)

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
) -> pd.DataFrame:
    """Run all sampler-target combinations and save results."""

    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(seed)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    # Iterate over all combinations
    for target_name in targets:
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target_name.upper()} (dim={dim})")
        print(f"{'#'*80}")

        target = get_target(target_name, dim=dim)

        for sampler in samplers:
            if sampler in ["grahmc", "rahmc"]:
                # Test each schedule for GRAHMC
                for schedule in grahmc_schedules:
                    key, subkey = random.split(key)
                    results = run_single_benchmark(
                        sampler=sampler,
                        target=target,
                        key=subkey,
                        n_chains=n_chains,
                        num_warmup=num_warmup,
                        target_ess=target_ess,
                        batch_size=batch_size,
                        max_samples=max_samples,
                        schedule_type=schedule,
                    )
                    all_results.append(results)
            else:
                # Other samplers don't have schedules
                key, subkey = random.split(key)
                results = run_single_benchmark(
                    sampler=sampler,
                    target=target,
                    key=subkey,
                    n_chains=n_chains,
                    num_warmup=num_warmup,
                    target_ess=target_ess,
                    batch_size=batch_size,
                    max_samples=max_samples,
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
    )

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
