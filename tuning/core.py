"Core tuning and sampling orchestration."
import argparse
import sys
from typing import Dict

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import arviz as az

# Import samplers
from samplers.RWMH import rwMH_run
from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run, get_friction_schedule

# Import target distributions
from benchmarks.targets import TargetDistribution, get_target

# Import tuning algorithms
from tuning.adaptation import run_adaptive_warmup
from tuning.dual_averaging import (
    dual_averaging_tune_rwmh,
    coordinate_wise_tune_grahmc,
)

# Import plotting functions
from tuning.plots import (
    plot_tuning_history,
    plot_sampling_diagnostics,
    plot_grid_comparison,
    plot_grahmc_grid_comparison,
    plot_coordinate_tuning_history,
)


def compute_diagnostics(samples: jnp.ndarray) -> Dict:
    """Compute convergence diagnostics and summary statistics.

    Args:
        samples: Array of shape (n_samples, n_chains, n_dim)

    Returns:
        Dictionary of diagnostic results
    """
    n_samples, n_chains, n_dim = samples.shape

    # Convert to ArviZ InferenceData format
    # ArviZ expects (chain, draw, *shape)
    samples_for_arviz = np.array(samples).transpose(1, 0, 2)  # (n_chains, n_samples, n_dim)

    idata = az.from_dict(
        posterior={"x": samples_for_arviz},
        coords={"dim": np.arange(n_dim)},
        dims={"x": ["dim"]}
    )

    # Compute split R-hat (rank-normalized)
    rhat = az.rhat(idata, var_names=["x"])
    rhat_values = rhat["x"].values  # Per-dimension R-hat

    # Compute ESS
    ess_bulk = az.ess(idata, var_names=["x"], method="bulk")["x"].values
    ess_tail = az.ess(idata, var_names=["x"], method="tail")["x"].values

    # Compute summary statistics
    summary = az.summary(idata, var_names=["x"])

    diagnostics = {
        "rhat_max": float(np.max(rhat_values)),
        "rhat_mean": float(np.mean(rhat_values)),
        "rhat_per_dim": rhat_values,
        "ess_bulk_min": float(np.min(ess_bulk)),
        "ess_bulk_mean": float(np.mean(ess_bulk)),
        "ess_tail_min": float(np.min(ess_tail)),
        "ess_tail_mean": float(np.mean(ess_tail)),
        "summary": summary,
    }

    return diagnostics


def tune_and_sample_rwmh(
    key: jnp.ndarray,
    target: TargetDistribution,
    n_chains: int = 4,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    warmup_steps: int = 2000,
) -> dict:
    """Tune RWMH parameters and run sampler with adaptive sampling until target ESS.

    Args:
        key: JAX random key
        target: TargetDistribution object
        n_chains: Number of parallel chains
        target_ess: Target minimum bulk ESS per dimension
        batch_size: Number of samples per batch
        max_samples: Maximum total samples to collect
        warmup_steps: Maximum tuning iterations

    Returns:
        Dictionary containing tuned parameters, samples, and diagnostics
    """
    n_dim = target.dim
    log_prob_fn = target.log_prob_fn

    # Initialize chains (use custom sampler if provided, else overdispersed start)
    key, init_key = random.split(key)
    if target.init_sampler is not None:
        init_position = target.init_sampler(init_key, n_chains)
    else:
        init_position = random.normal(init_key, shape=(n_chains, n_dim)) * 2.0

    print(f"\n{'='*60}")
    print(f"TUNING RWMH SAMPLER")
    print(f"{'='*60}")
    print(f"Target: {target.name}")
    print(f"  {target.description}")
    print(f"Chains: {n_chains}")
    print(f"Max tuning iterations: {warmup_steps}")

    # Tune scale parameter
    print("\nTuning proposal scale...")
    key, tune_key = random.split(key)
    scale, history = dual_averaging_tune_rwmh(
        tune_key, log_prob_fn, init_position, max_iter=warmup_steps
    )

    print(f"\n{'='*60}")
    print(f"ADAPTIVE SAMPLING WITH TUNED PARAMETERS")
    print(f"{'='*60}")
    print(f"Tuned scale: {scale:.4f}")
    print(f"Target ESS: {target_ess}")
    print(f"Batch size: {batch_size}")
    print(f"Max samples: {max_samples}")

    # Adaptive sampling until target ESS is reached
    print(f"\nSampling adaptively until ESS >= {target_ess}...")
    all_samples_list = []
    all_log_probs_list = []
    total_samples = 0
    batch_num = 0
    current_position = init_position  # Track chain state across batches

    while total_samples < max_samples:
        batch_num += 1
        key, sample_key = random.split(key)

        samples_batch, lps_batch, accept_rate, final_state = rwMH_run(
            sample_key, log_prob_fn, current_position,
            num_samples=batch_size, scale=scale, burn_in=0
        )

        # Continue from where we left off
        current_position = final_state.position

        all_samples_list.append(samples_batch)
        all_log_probs_list.append(lps_batch)
        total_samples += batch_size

        # Concatenate all samples collected so far
        samples = jnp.concatenate(all_samples_list, axis=0)

        # Compute ESS
        samples_for_arviz = np.array(samples).transpose(1, 0, 2)
        idata = az.from_dict(
            posterior={"x": samples_for_arviz},
            coords={"dim": np.arange(n_dim)},
            dims={"x": ["dim"]}
        )
        ess_bulk = az.ess(idata, var_names=["x"], method="bulk")["x"].values
        min_ess = float(np.min(ess_bulk))
        mean_ess = float(np.mean(ess_bulk))

        print(f"  Batch {batch_num}: {total_samples} total samples, min ESS = {min_ess:.1f}, mean ESS = {mean_ess:.1f}")

        if min_ess >= target_ess:
            print(f"  Target ESS reached!")
            break

    # Final samples
    samples = jnp.concatenate(all_samples_list, axis=0)
    log_probs = jnp.concatenate(all_log_probs_list, axis=0)

    # Compute convergence diagnostics
    print(f"\n{'='*60}")
    print(f"CONVERGENCE DIAGNOSTICS")
    print(f"{'='*60}")
    diagnostics = compute_diagnostics(samples)

    print(f"\nSplit R-hat (rank-normalized):")
    print(f"  Max: {diagnostics['rhat_max']:.4f}")
    print(f"  Mean: {diagnostics['rhat_mean']:.4f}")
    rhat_pass = diagnostics['rhat_max'] < 1.01
    print(f"  Status: {'PASS' if rhat_pass else 'FAIL'} (threshold: 1.01)")

    print(f"\nEffective Sample Size (bulk):")
    print(f"  Min: {diagnostics['ess_bulk_min']:.1f}")
    print(f"  Mean: {diagnostics['ess_bulk_mean']:.1f}")
    ess_pass = diagnostics['ess_bulk_min'] >= target_ess
    print(f"  Status: {'PASS' if ess_pass else 'FAIL'} (threshold: {target_ess})")

    print(f"\nEffective Sample Size (tail):")
    print(f"  Min: {diagnostics['ess_tail_min']:.1f}")
    print(f"  Mean: {diagnostics['ess_tail_mean']:.1f}")

    # Compute summary statistics
    mean_acceptance = float(jnp.mean(accept_rate))
    sample_mean = np.mean(samples, axis=(0, 1))
    sample_std = np.std(samples, axis=(0, 1))
    mean_mean = float(np.mean(sample_mean))
    mean_std = float(np.mean(sample_std))

    print(f"\nSummary Statistics:")
    print(f"  Mean acceptance rate: {mean_acceptance:.3f}")
    print(f"  Mean of sample means (should be ~0): {mean_mean:.4f}")
    print(f"  Mean of sample stds (should be ~1): {mean_std:.4f}")

    return {
        "scale": scale,
        "history": history,
        "samples": samples,
        "log_probs": log_probs,
        "accept_rate": accept_rate,
        "mean_acceptance": mean_acceptance,
        "diagnostics": diagnostics,
        "total_samples": total_samples,
    }


def tune_and_sample_nuts(
    key: jnp.ndarray,
    target: TargetDistribution,
    n_chains: int = 4,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    warmup_steps: int = 1000,
    max_tree_depth: int = 10,
) -> dict:
    """Tune NUTS parameters and run sampler with adaptive sampling until target ESS.

    Args:
        key: JAX random key
        target: TargetDistribution object
        n_chains: Number of parallel chains
        target_ess: Target minimum bulk ESS per dimension
        batch_size: Number of samples per batch
        max_samples: Maximum total samples to collect
        warmup_steps: Number of warmup steps for adaptation.
        max_tree_depth: Maximum tree depth for NUTS

    Returns:
        Dictionary containing tuned parameters, samples, diagnostics, and cost metrics
    """
    n_dim = target.dim
    log_prob_fn = target.log_prob_fn

    # Initialize chains (use custom sampler if provided, else overdispersed start)
    key, init_key = random.split(key)
    if target.init_sampler is not None:
        init_position = target.init_sampler(init_key, n_chains)
    else:
        init_position = random.normal(init_key, shape=(n_chains, n_dim)) * 2.0

    print(f"\n{'='*60}")
    print(f"TUNING NUTS SAMPLER")
    print(f"{'='*60}")
    print(f"Target: {target.name}")
    print(f"  {target.description}")
    print(f"Chains: {n_chains}")
    print(f"Warmup Steps: {warmup_steps}")
    print(f"Max tree depth: {max_tree_depth}")

    # Tune step size and mass matrix
    print("\nRunning adaptive warmup...")
    key, tune_key = random.split(key)
    step_size, inv_mass_matrix, warmup_position, tune_history = run_adaptive_warmup(
        tune_key, "nuts", log_prob_fn, init_position,
        num_warmup=warmup_steps,
        max_tree_depth=max_tree_depth
    )

    print(f"\n{'='*60}")
    print(f"ADAPTIVE SAMPLING WITH TUNED PARAMETERS")
    print(f"{'='*60}")
    print(f"Tuned step size: {step_size:.4f}")
    print(f"Target ESS: {target_ess}")
    print(f"Batch size: {batch_size}")
    print(f"Max samples: {max_samples}")

    # Adaptive sampling until target ESS is reached
    print(f"\nSampling adaptively until ESS >= {target_ess}...")
    all_samples_list = []
    all_log_probs_list = []
    all_tree_depths_list = []
    all_mean_accept_probs_list = []
    total_samples = 0
    batch_num = 0
    current_position = warmup_position  # Start from end of warmup

    while total_samples < max_samples:
        batch_num += 1
        key, sample_key = random.split(key)

        samples_batch, lps_batch, accept_rate, final_state, tree_depths, mean_accept_probs = nuts_run(
            sample_key, log_prob_fn, current_position,
            step_size=step_size,
            inv_mass_matrix=inv_mass_matrix,
            max_tree_depth=max_tree_depth,
            num_samples=batch_size, burn_in=0
        )

        # Continue from where we left off
        current_position = final_state.position

        all_samples_list.append(samples_batch)
        all_log_probs_list.append(lps_batch)
        all_tree_depths_list.append(tree_depths)
        all_mean_accept_probs_list.append(mean_accept_probs)
        total_samples += batch_size

        # Concatenate all samples collected so far
        samples = jnp.concatenate(all_samples_list, axis=0)

        # Compute ESS
        samples_for_arviz = np.array(samples).transpose(1, 0, 2)
        idata = az.from_dict(
            posterior={"x": samples_for_arviz},
            coords={"dim": np.arange(n_dim)},
            dims={"x": ["dim"]}
        )
        ess_bulk = az.ess(idata, var_names=["x"], method="bulk")["x"].values
        min_ess = float(np.min(ess_bulk))
        mean_ess = float(np.mean(ess_bulk))

        # Compute total gradient evaluations so far
        all_tree_depths = jnp.concatenate(all_tree_depths_list, axis=0)
        # For tree depth d: total leapfrog steps = 2^(d+1) - 1
        total_gradient_calls = int(jnp.sum(2**(all_tree_depths + 1) - 1))
        avg_tree_depth = float(jnp.mean(all_tree_depths))

        print(f"  Batch {batch_num}: {total_samples} samples, {total_gradient_calls} grad calls, "
              f"min ESS = {min_ess:.1f}, mean ESS = {mean_ess:.1f}, avg depth = {avg_tree_depth:.1f}")

        if min_ess >= target_ess:
            print(f"  Target ESS reached!")
            break

    # Final samples and metrics
    samples = jnp.concatenate(all_samples_list, axis=0)
    log_probs = jnp.concatenate(all_log_probs_list, axis=0)
    tree_depths = jnp.concatenate(all_tree_depths_list, axis=0)
    mean_accept_probs = jnp.concatenate(all_mean_accept_probs_list, axis=0)

    # Compute final gradient evaluation count
    total_gradient_calls = int(jnp.sum(2**(tree_depths + 1) - 1))
    avg_tree_depth = float(jnp.mean(tree_depths))
    avg_mean_accept = float(jnp.mean(mean_accept_probs))

    # Compute convergence diagnostics
    print(f"\n{'='*60}")
    print(f"CONVERGENCE DIAGNOSTICS")
    print(f"{'='*60}")
    diagnostics = compute_diagnostics(samples)

    print(f"\nSplit R-hat (rank-normalized):")
    print(f"  Max: {diagnostics['rhat_max']:.4f}")
    print(f"  Mean: {diagnostics['rhat_mean']:.4f}")
    rhat_pass = diagnostics['rhat_max'] < 1.01
    print(f"  Status: {'PASS' if rhat_pass else 'FAIL'} (threshold: 1.01)")

    print(f"\nEffective Sample Size (bulk):")
    print(f"  Min: {diagnostics['ess_bulk_min']:.1f}")
    print(f"  Mean: {diagnostics['ess_bulk_mean']:.1f}")
    ess_pass = diagnostics['ess_bulk_min'] >= target_ess
    print(f"  Status: {'PASS' if ess_pass else 'FAIL'} (threshold: {target_ess})")

    print(f"\nEffective Sample Size (tail):")
    print(f"  Min: {diagnostics['ess_tail_min']:.1f}")
    print(f"  Mean: {diagnostics['ess_tail_mean']:.1f}")

    # Compute efficiency metrics
    ess_per_sample = diagnostics['ess_bulk_min'] / total_samples
    ess_per_gradient = diagnostics['ess_bulk_min'] / total_gradient_calls

    print(f"\nComputational Efficiency:")
    print(f"  Total gradient calls: {total_gradient_calls}")
    print(f"  Average tree depth: {avg_tree_depth:.2f}")
    print(f"  ESS per sample: {ess_per_sample:.4f}")
    print(f"  ESS per gradient call: {ess_per_gradient:.6f}")

    # Compute summary statistics
    sample_mean = np.mean(samples, axis=(0, 1))
    sample_std = np.std(samples, axis=(0, 1))
    mean_mean = float(np.mean(sample_mean))
    mean_std = float(np.mean(sample_std))

    print(f"\nSummary Statistics:")
    print(f"  Mean acceptance probability: {avg_mean_accept:.3f}")
    print(f"  Mean of sample means (should be ~0): {mean_mean:.4f}")
    print(f"  Mean of sample stds (should be ~1): {mean_std:.4f}")

    return {
        "step_size": step_size,
        "inv_mass_matrix": inv_mass_matrix,
        "max_tree_depth": max_tree_depth,
        "history": tune_history,
        "samples": samples,
        "log_probs": log_probs,
        "tree_depths": tree_depths,
        "mean_accept_probs": mean_accept_probs,
        "avg_mean_accept": avg_mean_accept,
        "diagnostics": diagnostics,
        "total_samples": total_samples,
        "total_gradient_calls": total_gradient_calls,
        "avg_tree_depth": avg_tree_depth,
        "ess_per_sample": ess_per_sample,
        "ess_per_gradient": ess_per_gradient,
    }


def tune_and_sample_hmc_grid(
    key: jnp.ndarray,
    target: TargetDistribution,
    n_chains: int = 4,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    warmup_steps: int = 1000,
    num_steps_grid: list = None,
) -> dict:
    """Grid search over HMC num_steps, tuning step_size and mass matrix for each.

    Args:
        key: JAX random key
        target: TargetDistribution object
        n_chains: Number of parallel chains
        target_ess: Target minimum bulk ESS per dimension
        batch_size: Number of samples per batch
        max_samples: Maximum total samples to collect
        warmup_steps: Number of warmup steps for adaptation.
        num_steps_grid: List of num_steps values to try (default: [8, 16, 24, 32, 48, 64])

    Returns:
        Dictionary containing best configuration, grid results, and comparison data
    """
    if num_steps_grid is None:
        num_steps_grid = [8, 16, 24, 32, 48, 64]

    n_dim = target.dim
    log_prob_fn = target.log_prob_fn

    # Initialize chains (use custom sampler if provided, else overdispersed start)
    key, init_key = random.split(key)
    if target.init_sampler is not None:
        init_position = target.init_sampler(init_key, n_chains)
    else:
        init_position = random.normal(init_key, shape=(n_chains, n_dim)) * 2.0

    print(f"\n{'='*60}")
    print(f"HMC GRID SEARCH")
    print(f"{'='*60}")
    print(f"Target: {target.name}")
    print(f"  {target.description}")
    print(f"Chains: {n_chains}")
    print(f"Grid: num_steps = {num_steps_grid}")
    print(f"Target ESS: {target_ess}")

    grid_results = []

    for L in num_steps_grid:
        print(f"\n{'='*60}")
        print(f"TUNING HMC WITH NUM_STEPS = {L}")
        print(f"{'='*60}")

        # 1. Tune step_size and mass matrix for this L
        key, tune_key = random.split(key)
        step_size, inv_mass_matrix, warmup_position, tune_history = run_adaptive_warmup(
            tune_key, "hmc", log_prob_fn, init_position,
            num_warmup=warmup_steps,
            num_steps=L
        )

        print(f"\n  ADAPTIVE SAMPLING")
        print(f"  Tuned step_size: {step_size:.4f}")

        # 2. Sample with tuned parameters until target ESS
        all_samples_list = []
        all_log_probs_list = []
        total_samples = 0
        batch_num = 0
        current_position = warmup_position

        while total_samples < max_samples:
            batch_num += 1
            key, sample_key = random.split(key)

            samples_batch, lps_batch, accept_rate, final_state = hmc_run(
                sample_key, log_prob_fn, current_position,
                step_size=step_size, num_steps=L,
                inv_mass_matrix=inv_mass_matrix,
                num_samples=batch_size, burn_in=0
            )

            current_position = final_state.position
            all_samples_list.append(samples_batch)
            all_log_probs_list.append(lps_batch)
            total_samples += batch_size

            # Compute ESS
            samples = jnp.concatenate(all_samples_list, axis=0)
            samples_for_arviz = np.array(samples).transpose(1, 0, 2)
            idata = az.from_dict(
                posterior={"x": samples_for_arviz},
                coords={"dim": np.arange(n_dim)},
                dims={"x": ["dim"]}
            )
            ess_bulk = az.ess(idata, var_names=["x"], method="bulk")["x"].values
            min_ess = float(np.min(ess_bulk))
            mean_ess = float(np.mean(ess_bulk))

            print(f"    Batch {batch_num}: {total_samples} samples, min ESS = {min_ess:.1f}, mean ESS = {mean_ess:.1f}")

            if min_ess >= target_ess:
                print(f"    Target ESS reached!")
                break

        # 3. Final diagnostics
        samples = jnp.concatenate(all_samples_list, axis=0)
        log_probs = jnp.concatenate(all_log_probs_list, axis=0)
        diagnostics = compute_diagnostics(samples)

        # 4. Compute efficiency metrics
        total_gradient_calls = total_samples * L
        ess_per_sample = diagnostics['ess_bulk_min'] / total_samples
        ess_per_gradient = diagnostics['ess_bulk_min'] / total_gradient_calls

        print(f"\n  RESULTS FOR NUM_STEPS = {L}:")
        print(f"    Step size: {step_size:.4f}")
        print(f"    Total samples: {total_samples}")
        print(f"    Total gradient calls: {total_gradient_calls}")
        print(f"    Min ESS: {diagnostics['ess_bulk_min']:.1f}")
        print(f"    ESS per sample: {ess_per_sample:.4f}")
        print(f"    ESS per gradient: {ess_per_gradient:.6f}")
        print(f"    R-hat max: {diagnostics['rhat_max']:.4f}")

        # 5. Store results
        grid_results.append({
            'num_steps': L,
            'step_size': step_size,
            'inv_mass_matrix': inv_mass_matrix,
            'tune_history': tune_history,
            'samples': samples,
            'log_probs': log_probs,
            'accept_rate': accept_rate,
            'diagnostics': diagnostics,
            'total_samples': total_samples,
            'total_gradient_calls': total_gradient_calls,
            'ess_per_sample': ess_per_sample,
            'ess_per_gradient': ess_per_gradient,
        })

    # 6. Select best configuration
    best_config = max(grid_results, key=lambda x: x['ess_per_gradient'])

    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"  num_steps = {best_config['num_steps']}")
    print(f"  step_size = {best_config['step_size']:.4f}")
    print(f"  ESS per gradient = {best_config['ess_per_gradient']:.6f}")
    print(f"  Total samples = {best_config['total_samples']}")
    print(f"  Total gradient calls = {best_config['total_gradient_calls']}")

    return {
        'best_config': best_config,
        'grid_results': grid_results,
        'num_steps_grid': num_steps_grid,
    }


def tune_and_sample_grahmc_grid(
    key: jnp.ndarray,
    target: TargetDistribution,
    n_chains: int = 4,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    warmup_steps: int = 1000,
    max_cycles: int = 10,
    schedule_type: str = 'constant',
    num_steps_grid: list = None,
) -> dict:
    """Grid search over GRAHMC num_steps, coordinate-wise tuning for each.

    Args:
        key: JAX random key
        target: TargetDistribution object
        n_chains: Number of parallel chains
        target_ess: Target minimum bulk ESS per dimension
        batch_size: Number of samples per batch
        max_samples: Maximum total samples to collect
        warmup_steps: Number of warmup steps for adaptation.
        max_cycles: Maximum coordinate-wise tuning cycles
        schedule_type: Friction schedule ('constant', 'tanh', 'sigmoid', 'linear', 'sine')
        num_steps_grid: List of num_steps values to try (default: [8, 16, 24, 32, 48, 64])

    Returns:
        Dictionary containing best configuration, grid results, and comparison data
    """
    if num_steps_grid is None:
        num_steps_grid = [8, 16, 24, 32, 48, 64]

    n_dim = target.dim
    log_prob_fn = target.log_prob_fn

    # Get friction schedule
    friction_schedule = get_friction_schedule(schedule_type)
    has_steepness = schedule_type in ['tanh', 'sigmoid']

    # Initialize chains (use custom sampler if provided, else overdispersed start)
    key, init_key = random.split(key)
    if target.init_sampler is not None:
        init_position = target.init_sampler(init_key, n_chains)
    else:
        init_position = random.normal(init_key, shape=(n_chains, n_dim)) * 2.0

    print(f"\n{'='*60}")
    print(f"GRAHMC GRID SEARCH ({schedule_type.upper()} schedule)")
    print(f"{'='*60}")
    print(f"Target: {target.name}")
    print(f"  {target.description}")
    print(f"Chains: {n_chains}")
    print(f"Grid: num_steps = {num_steps_grid}")
    print(f"Target ESS: {target_ess}")

    grid_results = []

    for L in num_steps_grid:
        print(f"\n{'='*60}")
        print(f"TUNING GRAHMC WITH NUM_STEPS = {L}")
        print(f"{'='*60}")

        # 1. Tune step_size, mass_matrix, and friction parameters
        key, tune_key = random.split(key)
        # First, tune step size and mass matrix using the "hmc" mode of warmup
        step_size, inv_mass_matrix, warmup_position, _ = run_adaptive_warmup(
            tune_key, "hmc", log_prob_fn, init_position,
            num_warmup=warmup_steps,
            num_steps=L
        )

        # Then, fine-tune friction parameters holding step size and mass matrix fixed
        # CRITICAL: Pass the learned mass matrix so friction is tuned on sphered geometry
        key, tune_key2 = random.split(key)
        step_size, gamma, steepness, tune_history = coordinate_wise_tune_grahmc(
            key=tune_key2,
            log_prob_fn=log_prob_fn,
            grad_log_prob_fn=None,  # Not used internally
            init_position=warmup_position,
            num_steps=L,
            schedule_type=schedule_type,
            max_cycles=max_cycles,
            inv_mass_matrix=inv_mass_matrix,
        )

        print(f"\n  ADAPTIVE SAMPLING")
        print(f"  Tuned step_size: {step_size:.4f}")
        print(f"  Tuned gamma: {gamma:.4f}")
        if has_steepness:
            print(f"  Tuned steepness: {steepness:.4f}")

        # 2. Sample with tuned parameters until target ESS
        all_samples_list = []
        all_log_probs_list = []
        total_samples = 0
        batch_num = 0
        current_position = warmup_position

        while total_samples < max_samples:
            batch_num += 1
            key, sample_key = random.split(key)

            samples_batch, lps_batch, accept_rate, final_state = rahmc_run(
                sample_key, log_prob_fn, current_position,
                step_size=step_size, num_steps=L,
                gamma=gamma, steepness=steepness,
                inv_mass_matrix=inv_mass_matrix,
                num_samples=batch_size, burn_in=0,
                friction_schedule=friction_schedule
            )

            current_position = final_state.position
            all_samples_list.append(samples_batch)
            all_log_probs_list.append(lps_batch)
            total_samples += batch_size

            # Compute ESS
            samples = jnp.concatenate(all_samples_list, axis=0)
            samples_for_arviz = np.array(samples).transpose(1, 0, 2)
            idata = az.from_dict(
                posterior={"x": samples_for_arviz},
                coords={"dim": np.arange(n_dim)},
                dims={"x": ["dim"]}
            )
            ess_bulk = az.ess(idata, var_names=["x"], method="bulk")["x"].values
            min_ess = float(np.min(ess_bulk))
            mean_ess = float(np.mean(ess_bulk))

            print(f"    Batch {batch_num}: {total_samples} samples, min ESS = {min_ess:.1f}, mean ESS = {mean_ess:.1f}")

            if min_ess >= target_ess:
                print(f"    Target ESS reached!")
                break

        # 3. Final diagnostics
        samples = jnp.concatenate(all_samples_list, axis=0)
        log_probs = jnp.concatenate(all_log_probs_list, axis=0)
        diagnostics = compute_diagnostics(samples)

        # 4. Compute efficiency metrics
        total_gradient_calls = total_samples * L
        ess_per_sample = diagnostics['ess_bulk_min'] / total_samples
        ess_per_gradient = diagnostics['ess_bulk_min'] / total_gradient_calls

        print(f"\n  RESULTS FOR NUM_STEPS = {L}:")
        print(f"    Step size: {step_size:.4f}")
        print(f"    Gamma: {gamma:.4f}")
        if has_steepness:
            print(f"    Steepness: {steepness:.4f}")
        print(f"    Total samples: {total_samples}")
        print(f"    Total gradient calls: {total_gradient_calls}")
        print(f"    Min ESS: {diagnostics['ess_bulk_min']:.1f}")
        print(f"    ESS per sample: {ess_per_sample:.4f}")
        print(f"    ESS per gradient: {ess_per_gradient:.6f}")
        print(f"    R-hat max: {diagnostics['rhat_max']:.4f}")

        # 5. Store results
        grid_results.append({
            'num_steps': L,
            'step_size': step_size,
            'gamma': gamma,
            'steepness': steepness,
            'inv_mass_matrix': inv_mass_matrix,
            'tune_history': tune_history,
            'samples': samples,
            'log_probs': log_probs,
            'accept_rate': accept_rate,
            'diagnostics': diagnostics,
            'total_samples': total_samples,
            'total_gradient_calls': total_gradient_calls,
            'ess_per_sample': ess_per_sample,
            'ess_per_gradient': ess_per_gradient,
        })

    # 6. Select best configuration
    best_config = max(grid_results, key=lambda x: x['ess_per_gradient'])

    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"  num_steps = {best_config['num_steps']}")
    print(f"  step_size = {best_config['step_size']:.4f}")
    print(f"  gamma = {best_config['gamma']:.4f}")
    if has_steepness:
        print(f"  steepness = {best_config['steepness']:.4f}")
    print(f"  ESS per gradient = {best_config['ess_per_gradient']:.6f}")
    print(f"  Total samples = {best_config['total_samples']}")
    print(f"  Total gradient calls = {best_config['total_gradient_calls']}")

    return {
        'best_config': best_config,
        'grid_results': grid_results,
        'num_steps_grid': num_steps_grid,
        'schedule_type': schedule_type,
        'has_steepness': has_steepness,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune MCMC sampler hyperparameters using dual averaging"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        choices=["rwmh", "hmc", "nuts", "grahmc"],
        help="Sampler to tune"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="standard_normal",
        choices=["standard_normal", "correlated_gaussian", "ill_conditioned_gaussian",
                 "neals_funnel", "rosenbrock"],
        help="Target distribution (default: standard_normal)"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="constant",
        choices=["constant", "tanh", "sigmoid", "linear", "sine"],
        help="Friction schedule for GRAHMC (default: constant)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Dimensionality (default: 10)"
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of chains (default: 4)"
    )
    parser.add_argument(
        "--target-ess",
        type=int,
        default=1000,
        help="Target minimum ESS (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Samples per batch (default: 2000)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum total samples (default: 50000)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps for adaptation (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help="Maximum tree depth for NUTS (default: 10)"
    )
    parser.add_argument(
        "--num-steps-grid",
        type=str,
        default=None,
        help="Comma-separated list of num_steps for grid search (HMC default: 1,2,4,8,16,32,64; GRAHMC default: 8,16,32,64)"
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=10,
        help="Maximum coordinate-wise tuning cycles for GRAHMC (default: 10)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tuning_output",
        help="Directory for output plots (default: ./tuning_output)"
    )

    args = parser.parse_args()

    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(args.seed)

    # Create target distribution
    target = get_target(args.target, dim=args.dim)

    # Run tuning and sampling
    if args.sampler == "rwmh":
        results = tune_and_sample_rwmh(
            key=key,
            target=target,
            n_chains=args.chains,
            target_ess=args.target_ess,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            warmup_steps=args.warmup_steps,
        )
    elif args.sampler == "hmc":
        # Parse num_steps grid (default for HMC)
        if args.num_steps_grid is None:
            num_steps_grid = [8, 16, 24, 32, 48, 64]
        else:
            num_steps_grid = [int(x) for x in args.num_steps_grid.split(',')]
        results = tune_and_sample_hmc_grid(
            key=key,
            target=target,
            n_chains=args.chains,
            target_ess=args.target_ess,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            warmup_steps=args.warmup_steps,
            num_steps_grid=num_steps_grid,
        )
    elif args.sampler == "nuts":
        results = tune_and_sample_nuts(
            key=key,
            target=target,
            n_chains=args.chains,
            target_ess=args.target_ess,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            warmup_steps=args.warmup_steps,
            max_tree_depth=args.max_tree_depth,
        )
    elif args.sampler == "grahmc":
        # Parse num_steps grid (default for GRAHMC)
        if args.num_steps_grid is None:
            num_steps_grid = [8, 16, 24, 32, 48, 64]
        else:
            num_steps_grid = [int(x) for x in args.num_steps_grid.split(',')]
        results = tune_and_sample_grahmc_grid(
            key=key,
            target=target,
            n_chains=args.chains,
            target_ess=args.target_ess,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            warmup_steps=args.warmup_steps,
            max_cycles=args.max_cycles,
            schedule_type=args.schedule,
            num_steps_grid=num_steps_grid,
        )
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Tuned {args.sampler.upper()} parameters:")
    if args.sampler == "rwmh":
        print(f"  scale = {results['scale']:.4f}")
        print(f"Total samples collected: {results['total_samples']}")
    elif args.sampler == "hmc":
        best = results['best_config']
        print(f"  num_steps = {best['num_steps']}")
        print(f"  step_size = {best['step_size']:.4f}")
        print(f"  ESS per gradient call = {best['ess_per_gradient']:.6f}")
        print(f"Total samples collected: {best['total_samples']}")
        print(f"Total gradient calls: {best['total_gradient_calls']}")
    elif args.sampler == "nuts":
        print(f"  step_size = {results['step_size']:.4f}")
        print(f"  max_tree_depth = {results['max_tree_depth']}")
        print(f"  avg_tree_depth = {results['avg_tree_depth']:.2f}")
        print(f"  ESS per gradient call = {results['ess_per_gradient']:.6f}")
        print(f"Total samples collected: {results['total_samples']}")
        print(f"Total gradient calls: {results['total_gradient_calls']}")
    elif args.sampler == "grahmc":
        best = results['best_config']
        print(f"  Schedule: {args.schedule}")
        print(f"  num_steps = {best['num_steps']}")
        print(f"  step_size = {best['step_size']:.4f}")
        print(f"  gamma = {best['gamma']:.4f}")
        if results['has_steepness']:
            print(f"  steepness = {best['steepness']:.4f}")
        print(f"  ESS per gradient call = {best['ess_per_gradient']:.6f}")
        print(f"Total samples collected: {best['total_samples']}")
        print(f"Total gradient calls = {best['total_gradient_calls']}")

    # Generate plots if requested
    # if args.plot:
    #     import os
    #     os.makedirs(args.output_dir, exist_ok=True)

    #     print(f"\n{'='*60}")
    #     print(f"GENERATING DIAGNOSTIC PLOTS")
    #     print(f"{'='*60}")

    #     if args.sampler == "hmc":
    #         # HMC: Plot grid comparison
    #         grid_plot_file = os.path.join(args.output_dir, "hmc_grid_comparison.png")
    #         plot_grid_comparison(results["grid_results"], results["num_steps_grid"],
    #                            output_file=grid_plot_file)

    #         # Plot tuning history for best configuration
    #         best = results['best_config']
    #         tuning_plot_file = os.path.join(args.output_dir, f"hmc_best_L{best['num_steps']}_tuning_history.png")
    #         # plot_tuning_history(best["tune_history"], sampler_name=f"HMC (L={best['num_steps']})",
    #         #                   output_file=tuning_plot_file)

    #         # Plot sampling diagnostics for best configuration
    #         sampling_plot_file = os.path.join(args.output_dir, f"hmc_best_L{best['num_steps']}_sampling_diagnostics.png")
    #         plot_sampling_diagnostics(best["samples"], best["diagnostics"],
    #                                  sampler_name=f"HMC (L={best['num_steps']})",
    #                                  output_file=sampling_plot_file)
    #     elif args.sampler == "grahmc":
    #         # GRAHMC: Plot grid comparison with schedule-specific parameters
    #         schedule_name = args.schedule
    #         grid_plot_file = os.path.join(args.output_dir, f"grahmc_{schedule_name}_grid_comparison.png")
    #         plot_grahmc_grid_comparison(results["grid_results"], results["num_steps_grid"],
    #                                    schedule_type=schedule_name,
    #                                    has_steepness=results['has_steepness'],
    #                                    output_file=grid_plot_file)

    #         # Plot coordinate-wise tuning history for best configuration
    #         best = results['best_config']
    #         tuning_plot_file = os.path.join(args.output_dir,
    #                                        f"grahmc_{schedule_name}_best_L{best['num_steps']}_tuning_history.png")
    #         plot_coordinate_tuning_history(best["tune_history"], output_file=tuning_plot_file)

    #         # Plot sampling diagnostics for best configuration
    #         sampling_plot_file = os.path.join(args.output_dir,
    #                                          f"grahmc_{schedule_name}_best_L{best['num_steps']}_sampling_diagnostics.png")
    #         plot_sampling_diagnostics(best["samples"], best["diagnostics"],
    #                                  sampler_name=f"GRAHMC-{schedule_name.upper()} (L={best['num_steps']})",
    #                                  output_file=sampling_plot_file)
    #     else:
    #         # RWMH/NUTS: Standard plots
    #         if "history" in results:
    #             tuning_plot_file = os.path.join(args.output_dir, f"{args.sampler}_tuning_history.png")
    #             plot_tuning_history(results["history"], sampler_name=args.sampler.upper(),
    #                                output_file=tuning_plot_file)

    #         sampling_plot_file = os.path.join(args.output_dir, f"{args.sampler}_sampling_diagnostics.png")
    #         plot_sampling_diagnostics(results["samples"], results["diagnostics"],
    #                                  sampler_name=args.sampler.upper(),
    #                                  output_file=sampling_plot_file)

    #     print(f"\nPlots saved to {args.output_dir}/")

    # return 0


if __name__ == "__main__":
    sys.exit(main())
