"""
Comprehensive benchmarking script for MCMC samplers across multiple targets.

This script runs all sampler-target combinations, collects results, and generates
comparison reports.

Usage:
    python run_benchmarks.py --output-dir results --dim 10
    python run_benchmarks.py --samplers hmc nuts --targets standard_normal ill_conditioned_gaussian
    python run_benchmarks.py --quick  # Fast test with reduced parameters
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random

from benchmarks.targets import get_target, TargetDistribution
from test_samplers import run_sampler, compute_diagnostics, check_summary_statistics


def run_single_benchmark(
    sampler: str,
    target: TargetDistribution,
    key: jnp.ndarray,
    n_chains: int = 4,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    schedule_type: str = "constant",
) -> Dict:
    """Run a single sampler-target benchmark and return results."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK: {sampler.upper()} on {target.name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        samples, log_probs, metadata = run_sampler(
            sampler=sampler,
            key=key,
            target=target,
            n_chains=n_chains,
            target_ess=target_ess,
            batch_size=batch_size,
            max_samples=max_samples,
            schedule_type=schedule_type,
        )

        # Compute diagnostics
        diagnostics = compute_diagnostics(samples)

        # Check summary statistics
        stats_pass = check_summary_statistics(diagnostics, target, tolerance=0.15)

        elapsed_time = time.time() - start_time

        # Compile results
        results = {
            "sampler": sampler,
            "target": target.name,
            "schedule": schedule_type if sampler in ["grahmc", "rahmc"] else None,
            "dim": target.dim,
            "n_chains": n_chains,
            "target_ess": target_ess,
            "total_samples": metadata["total_samples"],
            "accept_rate": metadata["accept_rate"],
            "elapsed_time": elapsed_time,
            # Diagnostics
            "rhat_max": diagnostics["rhat_max"],
            "rhat_mean": diagnostics["rhat_mean"],
            "ess_bulk_min": diagnostics["ess_bulk_min"],
            "ess_bulk_mean": diagnostics["ess_bulk_mean"],
            "ess_tail_min": diagnostics["ess_tail_min"],
            # Pass/fail
            "rhat_pass": diagnostics["rhat_max"] < 1.01,
            "ess_pass": diagnostics["ess_bulk_min"] >= target_ess,
            "stats_pass": stats_pass,
            "overall_pass": (diagnostics["rhat_max"] < 1.01 and
                           diagnostics["ess_bulk_min"] >= target_ess and
                           stats_pass),
        }

        # Add sampler-specific parameters
        if sampler == "rwmh":
            results["scale"] = metadata["scale"]
        elif sampler in ["hmc", "nuts"]:
            results["step_size"] = metadata["step_size"]
            if sampler == "hmc":
                results["num_steps"] = metadata["num_steps"]
            else:
                results["max_tree_depth"] = metadata["max_tree_depth"]
        elif sampler in ["grahmc", "rahmc"]:
            results["step_size"] = metadata["step_size"]
            results["num_steps"] = metadata["num_steps"]
            results["gamma"] = metadata["gamma"]
            if "steepness" in metadata:
                results["steepness"] = metadata.get("steepness")

        print(f"\n[PASS]" if results["overall_pass"] else f"\n[FAIL]")
        print(f"Time: {elapsed_time:.2f}s | ESS: {results['ess_bulk_min']:.1f} | Accept: {results['accept_rate']:.3f}")

        return results

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n[ERROR] {str(e)}")
        print(f"Time: {elapsed_time:.1f}s")

        return {
            "sampler": sampler,
            "target": target.name,
            "schedule": schedule_type if sampler in ["grahmc", "rahmc"] else None,
            "dim": target.dim,
            "error": str(e),
            "elapsed_time": elapsed_time,
            "overall_pass": False,
        }


def run_all_benchmarks(
    samplers: List[str],
    targets: List[str],
    grahmc_schedules: List[str],
    dim: int,
    n_chains: int,
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

    # Round numeric columns to reasonable precision for readability
    numeric_cols = df.select_dtypes(include=[float]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")

    # Round all numeric values in results for JSON output
    def round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 4)
        elif isinstance(obj, dict):
            return {k: round_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(x) for x in obj]
        return obj

    rounded_results = round_floats(all_results)
    with open(json_path, 'w') as f:
        json.dump(rounded_results, f, indent=2)
    print(f"[OK] Results saved to {json_path}")

    return df


def print_summary(df: pd.DataFrame):
    """Print a summary of benchmark results."""

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal experiments: {len(df)}")
    print(f"Passed: {df['overall_pass'].sum()} ({100*df['overall_pass'].mean():.1f}%)")
    print(f"Failed: {(~df['overall_pass']).sum()}")

    if 'error' in df.columns:
        errors = df['error'].notna().sum()
        if errors > 0:
            print(f"Errors: {errors}")

    print(f"\n{'='*80}")
    print("RESULTS BY SAMPLER")
    print(f"{'='*80}")

    sampler_summary = df.groupby('sampler').agg({
        'overall_pass': ['count', 'sum', 'mean'],
        'elapsed_time': 'mean',
        'ess_bulk_min': 'mean',
        'accept_rate': 'mean',
    }).round(3)
    print(sampler_summary)

    print(f"\n{'='*80}")
    print("RESULTS BY TARGET")
    print(f"{'='*80}")

    target_summary = df.groupby('target').agg({
        'overall_pass': ['count', 'sum', 'mean'],
        'elapsed_time': 'mean',
        'ess_bulk_min': 'mean',
    }).round(3)
    print(target_summary)

    # Find best performer per target
    print(f"\n{'='*80}")
    print("BEST SAMPLER PER TARGET (by min ESS)")
    print(f"{'='*80}")

    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        if not target_df.empty:
            best = target_df.loc[target_df['ess_bulk_min'].idxmax()]
            sampler_id = best['sampler']
            if pd.notna(best.get('schedule')):
                sampler_id += f"-{best['schedule']}"
            print(f"{target:40s} -> {sampler_id:20s} (ESS={best['ess_bulk_min']:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive MCMC benchmarks across samplers and targets"
    )

    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["rwmh", "hmc", "nuts", "grahmc"],
        choices=["rwmh", "hmc", "nuts", "grahmc", "rahmc"],
        help="Samplers to benchmark (default: all)"
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=["standard_normal", "correlated_gaussian", "ill_conditioned_gaussian", "neals_funnel", "rosenbrock"],
        choices=["standard_normal", "correlated_gaussian", "ill_conditioned_gaussian", "neals_funnel", "rosenbrock"],
        help="Target distributions (default: all)"
    )

    parser.add_argument(
        "--grahmc-schedules",
        nargs="+",
        default=["constant", "tanh", "sigmoid", "linear", "sine"],
        choices=["constant", "tanh", "sigmoid", "linear", "sine"],
        help="GRAHMC friction schedules to test (default: all)"
    )

    parser.add_argument("--dim", type=int, default=10, help="Dimensionality (default: 10)")
    parser.add_argument("--chains", type=int, default=4, help="Number of chains (default: 4)")
    parser.add_argument("--target-ess", type=int, default=1000, help="Target ESS (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size (default: 2000)")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples (default: 50000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (reduced parameters for fast testing)"
    )

    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt (useful for automation)"
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        print("\n[QUICK MODE] Using reduced parameters for fast testing")
        args.dim = 5
        args.chains = 2
        args.target_ess = 200
        args.batch_size = 500
        args.max_samples = 3000
        args.samplers = ["hmc", "nuts"]  # Faster samplers only
        args.targets = ["standard_normal", "ill_conditioned_gaussian"]  # Subset
        args.grahmc_schedules = ["constant"]  # Just one schedule

    print(f"\n{'='*80}")
    print("MCMC BENCHMARK SUITE")
    print(f"{'='*80}")
    print(f"Samplers: {', '.join(args.samplers)}")
    print(f"Targets: {', '.join(args.targets)}")
    if "grahmc" in args.samplers or "rahmc" in args.samplers:
        print(f"GRAHMC schedules: {', '.join(args.grahmc_schedules)}")
    print(f"Dimensionality: {args.dim}")
    print(f"Chains: {args.chains}")
    print(f"Target ESS: {args.target_ess}")
    print(f"Output: {args.output_dir}")

    # Calculate total experiments
    n_grahmc = args.samplers.count("grahmc") + args.samplers.count("rahmc")
    n_other = len(args.samplers) - n_grahmc
    total_experiments = len(args.targets) * (n_other + n_grahmc * len(args.grahmc_schedules))
    print(f"\nTotal experiments to run: {total_experiments}")

    if not args.no_confirm:
        input("\nPress Enter to start benchmarking (or Ctrl+C to cancel)...")

    # Run benchmarks
    start_time = time.time()
    df = run_all_benchmarks(
        samplers=args.samplers,
        targets=args.targets,
        grahmc_schedules=args.grahmc_schedules,
        dim=args.dim,
        n_chains=args.chains,
        target_ess=args.target_ess,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    total_time = time.time() - start_time

    # Print summary
    print_summary(df)

    print(f"\n{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")

    return 0 if df['overall_pass'].all() else 1


if __name__ == "__main__":
    sys.exit(main())
