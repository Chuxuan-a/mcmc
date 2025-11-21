"""
Comprehensive benchmarks WITHOUT mass matrix adaptation.
Uses dual averaging tuning but no mass matrix (identity covariance).
"""
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List

from benchmarks.targets import get_target
from test_samplers import (
    dual_averaging_tune_rwmh,
    dual_averaging_tune_hmc,
    dual_averaging_tune_nuts,
    dual_averaging_tune_grahmc,
    compute_diagnostics,
    check_summary_statistics
)
from samplers.RWMH import rwMH_run
from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run, get_friction_schedule

jax.config.update("jax_enable_x64", True)

def run_single_benchmark_no_mass(
    sampler: str,
    target_name: str,
    dim: int,
    schedule_type: str = None,
    n_chains: int = 4,
    target_ess: int = 1000,
    max_samples: int = 50000,
    num_steps_grid: List[int] = None,
    seed: int = 42,
) -> Dict:
    """Run single benchmark without mass matrix."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK: {sampler.upper()} on {target_name}")
    if schedule_type:
        print(f"  Schedule: {schedule_type}")
    print(f"{'='*80}")

    target = get_target(target_name, dim=dim)
    key = random.PRNGKey(seed)

    # Initialize
    key, init_key = random.split(key)
    if target.init_sampler is not None:
        init_pos = target.init_sampler(init_key, n_chains)
    else:
        init_pos = random.normal(init_key, (n_chains, dim)) * 2.0

    start_time = time.time()

    # Grid search for HMC/GRAHMC
    if sampler in ["hmc", "grahmc"] and num_steps_grid:
        print(f"\n[Grid Search] Testing L={num_steps_grid}")

        grid_results = []

        for num_steps in num_steps_grid:
            print(f"\n--- Testing L={num_steps} ---")
            result = _run_with_L_no_mass(
                sampler, target, init_pos, num_steps, schedule_type,
                n_chains, target_ess, max_samples, key, dim
            )
            grid_results.append(result)

        # Select best result based on ESS per gradient (among passing runs only)
        passing_results = [r for r in grid_results if r.get("overall_pass", False)]

        if passing_results:
            # Choose best among passing runs
            best_result = max(passing_results, key=lambda r: r.get("ess_per_gradient", 0))
        else:
            # No passing runs - choose best ESS/grad anyway (mark as failed)
            best_result = max(grid_results, key=lambda r: r.get("ess_per_gradient", 0))
            print("\n!!! WARNING: NO L VALUES PASSED ALL CHECKS !!!")
            print("!!! Selecting best ESS/grad among failed runs !!!")

        best_result["best_L"] = best_result["num_steps"]
        print(f"\n>>> BEST: L={best_result['best_L']} (ESS/grad={best_result['ess_per_gradient']:.6f})")

        # Add grid search metadata (reuse computed results)
        best_result["grid_search_info"] = {
            "tested_L_values": num_steps_grid,
            "all_results": [{
                "num_steps": r.get("num_steps", r.get("best_L")),
                "ess_per_gradient": r["ess_per_gradient"],
                "ess_bulk_min": r["ess_bulk_min"],
                "total_samples": r["total_samples"],
                "overall_pass": r.get("overall_pass", False),
            } for r in grid_results],
        }

        return best_result
    else:
        # RWMH or NUTS (no grid search)
        return _run_with_L_no_mass(
            sampler, target, init_pos, 20, schedule_type,
            n_chains, target_ess, max_samples, key, dim
        )


def _run_with_L_no_mass(sampler, target, init_pos, num_steps, schedule_type,
                        n_chains, target_ess, max_samples, key, dim):
    """Helper to run sampler with specific L, no mass matrix."""

    key, tune_key = random.split(key)

    # Tune parameters
    print(f"  Tuning (no mass matrix)...")
    tune_start = time.time()

    if sampler == "rwmh":
        scale, _ = dual_averaging_tune_rwmh(tune_key, target.log_prob_fn, init_pos)
        params = {"scale": scale}
    elif sampler == "hmc":
        step_size, _ = dual_averaging_tune_hmc(
            tune_key, target.log_prob_fn, init_pos, num_steps=num_steps
        )
        params = {"step_size": step_size, "num_steps": num_steps}
    elif sampler == "nuts":
        step_size, _ = dual_averaging_tune_nuts(tune_key, target.log_prob_fn, init_pos)
        params = {"step_size": step_size}
    elif sampler == "grahmc":
        friction_schedule = get_friction_schedule(schedule_type or "constant")
        tuned = dual_averaging_tune_grahmc(
            tune_key, target.log_prob_fn, init_pos,
            num_steps=num_steps, friction_schedule=friction_schedule
        )
        if len(tuned) == 4:
            step_size, gamma, steepness, _ = tuned
        else:
            step_size, gamma, _ = tuned
            steepness = 1.0
        params = {
            "step_size": step_size, "num_steps": num_steps,
            "gamma": gamma, "steepness": steepness
        }

    tune_time = time.time() - tune_start
    print(f"  Tuned in {tune_time:.1f}s: {params}")

    # Sample (adaptive until target ESS)
    print(f"  Sampling...")
    sample_start = time.time()

    key, sample_key = random.split(key)
    samples_list = []
    total_samples = 0
    current_pos = init_pos

    while total_samples < max_samples:
        batch_size = min(2000, max_samples - total_samples)

        if sampler == "rwmh":
            samples, lps, accept_rate, final_state = rwMH_run(
                sample_key, target.log_prob_fn, current_pos,
                scale=params["scale"], num_samples=batch_size, burn_in=0
            )
        elif sampler == "hmc":
            samples, lps, accept_rate, final_state = hmc_run(
                sample_key, target.log_prob_fn, current_pos,
                step_size=params["step_size"], num_steps=params["num_steps"],
                num_samples=batch_size, burn_in=0,
                inv_mass_matrix=None  # Force identity (no mass matrix)
            )
        elif sampler == "nuts":
            samples, lps, accept_rate, final_state, _, mean_accept = nuts_run(
                sample_key, target.log_prob_fn, current_pos,
                step_size=params["step_size"], num_samples=batch_size, burn_in=0,
                inv_mass_matrix=None  # Force identity
            )
        elif sampler == "grahmc":
            friction_schedule = get_friction_schedule(schedule_type or "constant")
            samples, lps, accept_rate, final_state = rahmc_run(
                sample_key, target.log_prob_fn, current_pos,
                step_size=params["step_size"], num_steps=params["num_steps"],
                gamma=params["gamma"], steepness=params.get("steepness", 1.0),
                num_samples=batch_size, burn_in=0,
                friction_schedule=friction_schedule,
                inv_mass_matrix=None  # Force identity
            )

        samples_list.append(samples)
        total_samples += batch_size
        current_pos = final_state.position

        # Check ESS
        all_samples = jnp.concatenate(samples_list, axis=0)
        diagnostics = compute_diagnostics(all_samples)
        ess_min = diagnostics["ess_bulk_min"]

        print(f"    Batch {len(samples_list)}: {total_samples} samples, ESS={ess_min:.1f}")

        if ess_min >= target_ess:
            print(f"  Reached target ESS!")
            break

        key, sample_key = random.split(key)

    sample_time = time.time() - sample_start

    # Final diagnostics
    all_samples = jnp.concatenate(samples_list, axis=0)
    diagnostics = compute_diagnostics(all_samples)
    stats_pass = check_summary_statistics(diagnostics, target, z_threshold=5.0, total_samples=total_samples)

    # Compute ESS/gradient
    if sampler == "rwmh":
        n_gradients = 0
        ess_per_grad = 0
    elif sampler == "nuts":
        n_gradients = total_samples * 50  # Rough estimate (tree depth varies)
        ess_per_grad = diagnostics["ess_bulk_min"] / n_gradients
    else:
        n_gradients = total_samples * params["num_steps"]
        ess_per_grad = diagnostics["ess_bulk_min"] / n_gradients

    # Checks
    rhat_pass = diagnostics["rhat_max"] <= 1.01
    ess_pass = diagnostics["ess_bulk_min"] >= target_ess
    overall_pass = rhat_pass and ess_pass and stats_pass

    result = {
        "sampler": sampler,
        "target": target.name,
        "schedule": schedule_type,
        "dim": dim,
        "n_chains": n_chains,
        "total_samples": total_samples,
        "target_ess": target_ess,
        "warmup_time": tune_time,
        "sample_time": sample_time,
        "total_time": tune_time + sample_time,
        "accept_rate": float(jnp.mean(accept_rate)),
        "rhat_max": diagnostics["rhat_max"],
        "rhat_mean": diagnostics["rhat_mean"],
        "ess_bulk_min": diagnostics["ess_bulk_min"],
        "ess_bulk_mean": diagnostics["ess_bulk_mean"],
        "ess_tail_min": diagnostics["ess_tail_min"],
        "rhat_pass": rhat_pass,
        "ess_pass": ess_pass,
        "stats_pass": stats_pass,
        "overall_pass": overall_pass,
        "n_gradients": n_gradients,
        "ess_per_gradient": ess_per_grad,
        "mass_matrix_used": False,
        **params
    }

    print(f"\n  Results: ESS={diagnostics['ess_bulk_min']:.1f}, R-hat={diagnostics['rhat_max']:.4f}")
    print(f"  Pass: {overall_pass} (rhat={rhat_pass}, ess={ess_pass}, stats={stats_pass})")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmarks WITHOUT mass matrix")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--targets", nargs="+",
                       default=["neals_funnel", "log_gamma", "rosenbrock", "gaussian_mixture", "student_t"])
    parser.add_argument("--samplers", nargs="+",
                       default=["rwmh", "hmc", "nuts", "grahmc"])
    parser.add_argument("--schedules", nargs="+",
                       default=["constant", "tanh", "sigmoid", "linear", "sine"])
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--target-ess", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--num-steps-grid", nargs="+", type=int, default=[8, 16, 24, 32, 48, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_no_mass_matrix")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE BENCHMARKS - NO MASS MATRIX")
    print("="*80)
    print(f"Targets: {args.targets}")
    print(f"Samplers: {args.samplers}")
    print(f"GRAHMC schedules: {args.schedules}")
    print(f"Grid search L: {args.num_steps_grid}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    all_results = []

    for target_name in args.targets:
        for sampler in args.samplers:
            if sampler == "grahmc":
                # Test all schedules for GRAHMC
                for schedule in args.schedules:
                    result = run_single_benchmark_no_mass(
                        sampler=sampler,
                        target_name=target_name,
                        dim=args.dim,
                        schedule_type=schedule,
                        n_chains=args.n_chains,
                        target_ess=args.target_ess,
                        max_samples=args.max_samples,
                        num_steps_grid=args.num_steps_grid,
                        seed=args.seed
                    )
                    all_results.append(result)
            else:
                # Other samplers don't have schedules
                result = run_single_benchmark_no_mass(
                    sampler=sampler,
                    target_name=target_name,
                    dim=args.dim,
                    schedule_type=None,
                    n_chains=args.n_chains,
                    target_ess=args.target_ess,
                    max_samples=args.max_samples,
                    num_steps_grid=args.num_steps_grid if sampler == "hmc" else None,
                    seed=args.seed
                )
                all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)

    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"

    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[Saved] {json_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configurations: {len(all_results)}")
    print(f"Passed: {sum(r['overall_pass'] for r in all_results)}")
    print(f"Failed: {sum(not r['overall_pass'] for r in all_results)}")
    print("="*80)


if __name__ == "__main__":
    main()
