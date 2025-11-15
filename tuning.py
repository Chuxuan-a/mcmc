"""Dual averaging parameter tuning for MCMC samplers.

This module provides command-line tools for tuning sampler hyperparameters
using the dual averaging algorithm from Hoffman & Gelman (2014).

Usage:
    python tuning.py --sampler rwmh --target standard_normal --dim 10
    python tuning.py --sampler hmc --target neals_funnel --dim 10 --num-steps 20
    python tuning.py --sampler nuts --target ill_conditioned_gaussian --dim 10
    python tuning.py --sampler grahmc --target correlated_gaussian --schedule tanh --dim 10
"""
import argparse
import sys
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# Import samplers
from samplers.RWMH import rwMH_run
from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run, get_friction_schedule

# Import target distributions
from benchmarks.targets import TargetDistribution, get_target


def dual_averaging_tune_rwmh(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    target_accept: float = 0.234,  # Optimal for RWMH
    tolerance: float = 0.01,
    max_iter: int = 2000,
    min_iter: int = 100,
    patience: int = 10,
) -> Tuple[float, Dict]:
    """Tune RWMH scale parameter using dual averaging until convergence.

    Based on Hoffman & Gelman (2014) NUTS paper, Section 3.2.

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance (relative change in parameter)
        max_iter: Maximum number of tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Number of consecutive converged iterations required

    Returns:
        Tuple of (tuned_scale, history) where history contains:
        - scale_history: List of scale values over iterations
        - accept_history: List of acceptance rates over iterations
        - converged_iter: Iteration where convergence occurred
    """
    # Dual averaging parameters (from Stan)
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    # Initialize (use 2.38/sqrt(d) as starting point - Roberts & Rosenthal optimal)
    d = init_position.shape[-1]
    initial_scale = 2.38 / jnp.sqrt(d)
    log_scale = jnp.log(initial_scale)
    mu = log_scale  # Target around the initial scale
    log_scale_bar = 0.0
    H_bar = 0.0

    scale = jnp.exp(log_scale)
    prev_scale_bar = scale

    # Track history for visualization
    scale_history = []
    accept_history = []

    # Run tuning iterations until convergence
    n_samples_per_tune = 100  # More samples for better acceptance estimate
    converged_count = 0
    converged_iter = max_iter

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        _, _, accept_rate, _ = rwMH_run(
            subkey, log_prob_fn, init_position, num_samples=n_samples_per_tune, scale=float(scale), burn_in=0
        )
        alpha = float(jnp.mean(accept_rate))

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_scale = mu - (jnp.sqrt(m) / gamma) * H_bar
        m_kappa = m ** (-kappa)
        log_scale_bar = m_kappa * log_scale + (1 - m_kappa) * log_scale_bar

        scale = jnp.exp(log_scale)
        current_scale_bar = float(jnp.exp(log_scale_bar))

        # Record history
        scale_history.append(current_scale_bar)
        accept_history.append(alpha)

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_scale_bar - prev_scale_bar) / (abs(prev_scale_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: scale={current_scale_bar:.4f}, accept={alpha:.3f}")
                converged_iter = m
                history = {
                    "scale_history": scale_history,
                    "accept_history": accept_history,
                    "converged_iter": converged_iter,
                    "target_accept": target_accept,
                }
                return current_scale_bar, history

        prev_scale_bar = current_scale_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: scale={current_scale_bar:.4f}, accept={alpha:.3f}")

    final_scale = float(jnp.exp(log_scale_bar))
    print(f"  Reached max iterations ({max_iter}): scale={final_scale:.4f}")
    history = {
        "scale_history": scale_history,
        "accept_history": accept_history,
        "converged_iter": converged_iter,
        "target_accept": target_accept,
    }
    return final_scale, history


def dual_averaging_tune_hmc(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    num_steps: int,
    target_accept: float = 0.65,
    tolerance: float = 0.01,
    max_iter: int = 2000,
    min_iter: int = 100,
    patience: int = 10,
) -> Tuple[float, Dict]:
    """Tune HMC step size for fixed num_steps using dual averaging until convergence.

    Based on Hoffman & Gelman (2014) NUTS paper, Section 3.2.

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        num_steps: Fixed number of leapfrog steps (not tuned)
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance (relative change in parameter)
        max_iter: Maximum number of tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Number of consecutive converged iterations required

    Returns:
        Tuple of (tuned_step_size, history) where history contains:
        - step_size_history: List of step size values over iterations
        - accept_history: List of acceptance rates over iterations
        - converged_iter: Iteration where convergence occurred
    """
    # Dual averaging parameters (from Stan)
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    # Initialize with d-dependent starting point
    d = init_position.shape[-1]
    initial_step_size = 0.5 / jnp.sqrt(d)  # Scale inversely with sqrt(dim)
    log_step_size = jnp.log(initial_step_size)
    mu = log_step_size  # Set target around initial
    log_step_size_bar = 0.0
    H_bar = 0.0

    step_size = jnp.exp(log_step_size)
    prev_step_size_bar = float(step_size)

    # Track history for visualization
    step_size_history = []
    accept_history = []

    # Run tuning iterations until convergence
    n_samples_per_tune = 100  # More samples for better acceptance estimate
    converged_count = 0
    converged_iter = max_iter

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        # Run HMC to collect samples and get acceptance statistics
        _, _, accept_rate, _ = hmc_run(
            subkey, log_prob_fn, init_position,
            step_size=float(step_size), num_steps=num_steps,
            num_samples=n_samples_per_tune, burn_in=0
        )
        # Use mean acceptance rate across chains
        alpha = float(jnp.mean(accept_rate))

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_step_size = mu - (jnp.sqrt(m) / gamma) * H_bar
        m_kappa = m ** (-kappa)
        log_step_size_bar = m_kappa * log_step_size + (1 - m_kappa) * log_step_size_bar

        step_size = jnp.exp(log_step_size)
        current_step_size_bar = float(jnp.exp(log_step_size_bar))

        # Record history
        step_size_history.append(current_step_size_bar)
        accept_history.append(alpha)

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_step_size_bar - prev_step_size_bar) / (abs(prev_step_size_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"    Converged after {m} iterations: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")
                converged_iter = m
                history = {
                    "step_size_history": step_size_history,
                    "accept_history": accept_history,
                    "converged_iter": converged_iter,
                    "target_accept": target_accept,
                    "num_steps": num_steps,
                }
                return current_step_size_bar, history

        prev_step_size_bar = current_step_size_bar

        if m % 200 == 0:
            print(f"    Tuning iteration {m}: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")

    final_step_size = float(jnp.exp(log_step_size_bar))
    print(f"    Reached max iterations ({max_iter}): step_size={final_step_size:.4f}")
    history = {
        "step_size_history": step_size_history,
        "accept_history": accept_history,
        "converged_iter": converged_iter,
        "target_accept": target_accept,
        "num_steps": num_steps,
    }
    return final_step_size, history


def dual_averaging_tune_nuts(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    max_tree_depth: int = 10,
    target_accept: float = 0.65,  # Same as HMC
    tolerance: float = 0.01,
    max_iter: int = 2000,
    min_iter: int = 100,
    patience: int = 10,
) -> Tuple[float, Dict]:
    """Tune NUTS step size using dual averaging until convergence.

    Based on Hoffman & Gelman (2014) NUTS paper, Section 3.2.
    Note: NUTS automatically selects trajectory length, so only step size is tuned.

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        max_tree_depth: Maximum tree depth (fixed, not tuned)
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance (relative change in parameter)
        max_iter: Maximum number of tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Number of consecutive converged iterations required

    Returns:
        Tuple of (tuned_step_size, history) where history contains:
        - step_size_history: List of step size values over iterations
        - accept_history: List of mean acceptance probabilities over iterations
        - tree_depth_history: List of average tree depths over iterations
        - converged_iter: Iteration where convergence occurred
    """
    # Dual averaging parameters (from Stan)
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    # Initialize with d-dependent starting point
    d = init_position.shape[-1]
    initial_step_size = 0.5 / jnp.sqrt(d)  # Scale inversely with sqrt(dim)
    log_step_size = jnp.log(initial_step_size)
    mu = log_step_size  # Set target around initial
    log_step_size_bar = 0.0
    H_bar = 0.0

    step_size = jnp.exp(log_step_size)
    prev_step_size_bar = float(step_size)

    # Track history for visualization
    step_size_history = []
    accept_history = []
    tree_depth_history = []

    # Run tuning iterations until convergence
    n_samples_per_tune = 100  # More samples for better acceptance estimate
    converged_count = 0
    converged_iter = max_iter

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        # Run NUTS to collect samples and get acceptance statistics
        _, _, _, _, tree_depths, mean_accept_probs = nuts_run(
            subkey, log_prob_fn, init_position,
            step_size=float(step_size), max_tree_depth=max_tree_depth,
            num_samples=n_samples_per_tune, burn_in=0
        )
        # Use the mean of Metropolis acceptance probabilities from leapfrog trajectories
        # This is the standard statistic for NUTS dual averaging (Hoffman & Gelman 2014)
        alpha = float(jnp.mean(mean_accept_probs))
        avg_tree_depth = float(jnp.mean(tree_depths))

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_step_size = mu - (jnp.sqrt(m) / gamma) * H_bar
        m_kappa = m ** (-kappa)
        log_step_size_bar = m_kappa * log_step_size + (1 - m_kappa) * log_step_size_bar

        step_size = jnp.exp(log_step_size)
        current_step_size_bar = float(jnp.exp(log_step_size_bar))

        # Record history
        step_size_history.append(current_step_size_bar)
        accept_history.append(alpha)
        tree_depth_history.append(avg_tree_depth)

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_step_size_bar - prev_step_size_bar) / (abs(prev_step_size_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}, avg_depth={avg_tree_depth:.1f}")
                converged_iter = m
                history = {
                    "step_size_history": step_size_history,
                    "accept_history": accept_history,
                    "tree_depth_history": tree_depth_history,
                    "converged_iter": converged_iter,
                    "target_accept": target_accept,
                    "max_tree_depth": max_tree_depth,
                }
                return current_step_size_bar, history

        prev_step_size_bar = current_step_size_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}, avg_depth={avg_tree_depth:.1f}")

    final_step_size = float(jnp.exp(log_step_size_bar))
    print(f"  Reached max iterations ({max_iter}): step_size={final_step_size:.4f}")
    history = {
        "step_size_history": step_size_history,
        "accept_history": accept_history,
        "tree_depth_history": tree_depth_history,
        "converged_iter": converged_iter,
        "target_accept": target_accept,
        "max_tree_depth": max_tree_depth,
    }
    return final_step_size, history


def coordinate_wise_tune_grahmc(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    num_steps: int,
    schedule_type: str = 'constant',
    target_accept: float = 0.65,
    tolerance: float = 0.02,
    max_cycles: int = 110,
    min_cycles: int = 3,
    patience: int = 2,
    max_iter_per_param: int = 400,
) -> Tuple[float, float, float, Dict]:
    """Tune GRAHMC hyperparameters using coordinate-wise dual averaging.

    Cycles through optimizing step_size, gamma, and steepness (if applicable)
    until convergence. Parameters are optimized one at a time using dual averaging
    while holding others fixed.

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        num_steps: Fixed number of leapfrog steps
        schedule_type: Friction schedule type ('constant', 'tanh', 'sigmoid', 'linear', 'sine')
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance (relative change)
        max_cycles: Maximum number of coordinate cycles
        min_cycles: Minimum cycles before checking convergence
        patience: Number of converged cycles required
        max_iter_per_param: Maximum dual averaging iterations per parameter

    Returns:
        Tuple of (step_size, gamma, steepness, history) where history contains:
        - cycle_history: List of (step_size, gamma, steepness, accept_rate) per cycle
        - converged_cycle: Cycle where convergence occurred
    """
    # Get friction schedule and determine if it uses steepness
    friction_schedule = get_friction_schedule(schedule_type)
    has_steepness = schedule_type in ['tanh', 'sigmoid']

    # Dual averaging parameters
    gamma_da = 0.05
    t0 = 10.0
    kappa = 0.75

    # Initialize parameters
    d = init_position.shape[-1]
    step_size = 0.5 / jnp.sqrt(d)
    gamma = 1.0
    steepness = 5.0 if schedule_type == 'tanh' else (10.0 if schedule_type == 'sigmoid' else 1.0)

    # Track history
    cycle_history = []
    converged_count = 0
    converged_cycle = max_cycles

    # Previous values for convergence check
    prev_step_size = step_size
    prev_gamma = gamma
    prev_steepness = steepness

    n_samples_per_tune = 100

    for cycle in range(max_cycles):
        print(f"    Cycle {cycle + 1}/{max_cycles}")

        # ===== 1. Tune step_size (hold gamma, steepness fixed) =====
        log_step_size = jnp.log(step_size)
        mu_step = log_step_size
        log_step_size_bar = 0.0
        H_bar_step = 0.0

        for m in range(1, max_iter_per_param + 1):
            key, subkey = random.split(key)
            _, _, accept_rate, _ = rahmc_run(
                subkey, log_prob_fn, init_position,
                step_size=float(jnp.exp(log_step_size)), num_steps=num_steps,
                gamma=float(gamma), steepness=float(steepness),
                num_samples=n_samples_per_tune, burn_in=0,
                friction_schedule=friction_schedule
            )
            alpha = float(jnp.mean(accept_rate))

            eta_m = 1.0 / (m + t0)
            H_bar_step = (1 - eta_m) * H_bar_step + eta_m * (target_accept - alpha)
            log_step_size = mu_step - (jnp.sqrt(m) / gamma_da) * H_bar_step
            m_kappa = m ** (-kappa)
            log_step_size_bar = m_kappa * log_step_size + (1 - m_kappa) * log_step_size_bar

        step_size = float(jnp.exp(log_step_size_bar))

        # ===== 2. Tune gamma (hold step_size, steepness fixed) =====
        log_gamma = jnp.log(gamma)
        mu_gamma = log_gamma
        log_gamma_bar = 0.0
        H_bar_gamma = 0.0

        for m in range(1, max_iter_per_param + 1):
            key, subkey = random.split(key)
            _, _, accept_rate, _ = rahmc_run(
                subkey, log_prob_fn, init_position,
                step_size=float(step_size), num_steps=num_steps,
                gamma=float(jnp.exp(log_gamma)), steepness=float(steepness),
                num_samples=n_samples_per_tune, burn_in=0,
                friction_schedule=friction_schedule
            )
            alpha = float(jnp.mean(accept_rate))

            eta_m = 1.0 / (m + t0)
            H_bar_gamma = (1 - eta_m) * H_bar_gamma + eta_m * (target_accept - alpha)
            log_gamma = mu_gamma - (jnp.sqrt(m) / gamma_da) * H_bar_gamma
            m_kappa = m ** (-kappa)
            log_gamma_bar = m_kappa * log_gamma + (1 - m_kappa) * log_gamma_bar

        gamma = float(jnp.exp(log_gamma_bar))

        # ===== 3. Tune steepness (hold step_size, gamma fixed) - if applicable =====
        if has_steepness:
            log_steepness = jnp.log(steepness)
            mu_steepness = log_steepness
            log_steepness_bar = 0.0
            H_bar_steepness = 0.0

            for m in range(1, max_iter_per_param + 1):
                key, subkey = random.split(key)
                _, _, accept_rate, _ = rahmc_run(
                    subkey, log_prob_fn, init_position,
                    step_size=float(step_size), num_steps=num_steps,
                    gamma=float(gamma), steepness=float(jnp.exp(log_steepness)),
                    num_samples=n_samples_per_tune, burn_in=0,
                    friction_schedule=friction_schedule
                )
                alpha = float(jnp.mean(accept_rate))

                eta_m = 1.0 / (m + t0)
                H_bar_steepness = (1 - eta_m) * H_bar_steepness + eta_m * (target_accept - alpha)
                log_steepness = mu_steepness - (jnp.sqrt(m) / gamma_da) * H_bar_steepness
                m_kappa = m ** (-kappa)
                log_steepness_bar = m_kappa * log_steepness + (1 - m_kappa) * log_steepness_bar

            steepness = float(jnp.exp(log_steepness_bar))

        # Get final acceptance rate for this cycle
        key, subkey = random.split(key)
        _, _, accept_rate, _ = rahmc_run(
            subkey, log_prob_fn, init_position,
            step_size=float(step_size), num_steps=num_steps,
            gamma=float(gamma), steepness=float(steepness),
            num_samples=n_samples_per_tune, burn_in=0,
            friction_schedule=friction_schedule
        )
        final_accept = float(jnp.mean(accept_rate))

        cycle_history.append({
            'step_size': step_size,
            'gamma': gamma,
            'steepness': steepness,
            'accept_rate': final_accept,
        })

        print(f"      step_size={step_size:.4f}, gamma={gamma:.4f}, steepness={steepness:.4f}, accept={final_accept:.3f}")

        # Check convergence after minimum cycles
        if cycle >= min_cycles:
            rel_change_step = abs(step_size - prev_step_size) / (abs(prev_step_size) + 1e-10)
            rel_change_gamma = abs(gamma - prev_gamma) / (abs(prev_gamma) + 1e-10)
            rel_change_steep = abs(steepness - prev_steepness) / (abs(prev_steepness) + 1e-10) if has_steepness else 0.0

            max_change = max(rel_change_step, rel_change_gamma, rel_change_steep)

            if max_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"      Converged after {cycle + 1} cycles")
                converged_cycle = cycle + 1
                break

        prev_step_size = step_size
        prev_gamma = gamma
        prev_steepness = steepness

    history = {
        'cycle_history': cycle_history,
        'converged_cycle': converged_cycle,
        'schedule_type': schedule_type,
        'has_steepness': has_steepness,
        'num_steps': num_steps,
        'target_accept': target_accept,
    }

    return step_size, gamma, steepness, history


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


def plot_tuning_history(history: Dict, sampler_name: str = "RWMH", output_file: str = None):
    """Plot dual averaging tuning history.

    Args:
        history: Dictionary containing tuning history
        sampler_name: Name of sampler for plot title
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")

    # Check if this is NUTS (has tree_depth_history)
    is_nuts = "tree_depth_history" in history
    n_plots = 3 if is_nuts else 2

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)

    # Get parameter name (scale for RWMH, step_size for NUTS)
    if "scale_history" in history:
        param_history = history["scale_history"]
        param_name = "Scale Parameter"
    else:
        param_history = history["step_size_history"]
        param_name = "Step Size"

    iterations = np.arange(1, len(param_history) + 1)
    converged_iter = history["converged_iter"]

    # Plot parameter evolution (top panel)
    axes[0].plot(iterations, param_history, 'b-', linewidth=1.5, label=param_name)
    axes[0].axvline(converged_iter, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (iter {converged_iter})')
    axes[0].set_ylabel(param_name, fontsize=12)
    axes[0].set_title(f'{sampler_name} Dual Averaging Tuning History', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot acceptance rate evolution (middle panel)
    axes[1].plot(iterations, history["accept_history"], 'g-', linewidth=1.5, label='Acceptance rate')
    axes[1].axhline(history["target_accept"], color='orange', linestyle='--', linewidth=1.5,
                    label=f'Target ({history["target_accept"]:.3f})')
    axes[1].axvline(converged_iter, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (iter {converged_iter})')
    if not is_nuts:
        axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Acceptance Rate', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Plot tree depth evolution for NUTS (bottom panel)
    if is_nuts:
        axes[2].plot(iterations, history["tree_depth_history"], 'm-', linewidth=1.5, label='Avg tree depth')
        axes[2].axhline(history["max_tree_depth"], color='red', linestyle='--', linewidth=1.5,
                        label=f'Max depth ({history["max_tree_depth"]})')
        axes[2].axvline(converged_iter, color='r', linestyle='--', linewidth=1.5,
                        label=f'Converged (iter {converged_iter})')
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('Average Tree Depth', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved tuning history plot to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_sampling_diagnostics(samples: jnp.ndarray, diagnostics: Dict,
                               sampler_name: str = "RWMH", output_file: str = None):
    """Plot sampling diagnostics including trace plots and marginals.

    Args:
        samples: Array of shape (n_samples, n_chains, n_dim)
        diagnostics: Dictionary containing diagnostic results
        sampler_name: Name of sampler for plot title
        output_file: Path to save plot (if None, displays interactively)
    """
    n_samples, n_chains, n_dim = samples.shape

    # Plot only first 4 dimensions to keep it manageable
    plot_dims = min(4, n_dim)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(plot_dims, 2, figsize=(12, 3 * plot_dims))
    if plot_dims == 1:
        axes = axes.reshape(1, -1)

    samples_np = np.array(samples)

    for i in range(plot_dims):
        # Trace plots (left column)
        for chain in range(n_chains):
            axes[i, 0].plot(samples_np[:, chain, i], alpha=0.6, linewidth=0.5, label=f'Chain {chain+1}')
        axes[i, 0].set_ylabel(f'x[{i}]', fontsize=10)
        axes[i, 0].set_title(f'Trace Plot (dim {i})', fontsize=10)
        if i == 0:
            axes[i, 0].legend(loc='upper right', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        # Marginal distributions (right column)
        for chain in range(n_chains):
            axes[i, 1].hist(samples_np[:, chain, i], bins=30, alpha=0.4, density=True, label=f'Chain {chain+1}')

        # Overlay true standard normal
        x_range = np.linspace(-4, 4, 100)
        axes[i, 1].plot(x_range, np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi),
                        'k--', linewidth=2, label='True N(0,1)')
        axes[i, 1].set_xlabel(f'x[{i}]', fontsize=10)
        axes[i, 1].set_ylabel('Density', fontsize=10)
        axes[i, 1].set_title(f'Marginal (dim {i})', fontsize=10)
        if i == 0:
            axes[i, 1].legend(loc='upper right', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

    if plot_dims > 1:
        axes[-1, 0].set_xlabel('Iteration', fontsize=10)
    else:
        axes[0, 0].set_xlabel('Iteration', fontsize=10)

    plt.suptitle(f'{sampler_name} Sampling Diagnostics', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved sampling diagnostics plot to {output_file}")
    else:
        plt.show()
    plt.close()


def tune_and_sample_rwmh(
    key: jnp.ndarray,
    target: TargetDistribution,
    n_chains: int = 4,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    max_iter: int = 2000,
) -> dict:
    """Tune RWMH parameters and run sampler with adaptive sampling until target ESS.

    Args:
        key: JAX random key
        target: TargetDistribution object
        n_chains: Number of parallel chains
        target_ess: Target minimum bulk ESS per dimension
        batch_size: Number of samples per batch
        max_samples: Maximum total samples to collect
        max_iter: Maximum tuning iterations

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
    print(f"Max tuning iterations: {max_iter}")

    # Tune scale parameter
    print("\nTuning proposal scale...")
    key, tune_key = random.split(key)
    scale, history = dual_averaging_tune_rwmh(
        tune_key, log_prob_fn, init_position, max_iter=max_iter
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
    max_iter: int = 2000,
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
        max_iter: Maximum tuning iterations
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
    print(f"Max tuning iterations: {max_iter}")
    print(f"Max tree depth: {max_tree_depth}")

    # Tune step size parameter
    print("\nTuning step size...")
    key, tune_key = random.split(key)
    step_size, history = dual_averaging_tune_nuts(
        tune_key, log_prob_fn, init_position,
        max_tree_depth=max_tree_depth, max_iter=max_iter
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
    current_position = init_position  # Track chain state across batches

    while total_samples < max_samples:
        batch_num += 1
        key, sample_key = random.split(key)

        samples_batch, lps_batch, accept_rate, final_state, tree_depths, mean_accept_probs = nuts_run(
            sample_key, log_prob_fn, current_position,
            step_size=step_size, max_tree_depth=max_tree_depth,
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
        "max_tree_depth": max_tree_depth,
        "history": history,
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
    max_iter: int = 2000,
    num_steps_grid: list = None,
) -> dict:
    """Grid search over HMC num_steps, tuning step_size for each.

    Args:
        key: JAX random key
        target: TargetDistribution object
        n_chains: Number of parallel chains
        target_ess: Target minimum bulk ESS per dimension
        batch_size: Number of samples per batch
        max_samples: Maximum total samples to collect
        max_iter: Maximum tuning iterations
        num_steps_grid: List of num_steps values to try (default: [8, 16, 32, 64])

    Returns:
        Dictionary containing best configuration, grid results, and comparison data
    """
    if num_steps_grid is None:
        num_steps_grid = [8, 16, 32, 64]

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

        # 1. Tune step_size for this L
        key, tune_key = random.split(key)
        step_size, tune_history = dual_averaging_tune_hmc(
            tune_key, log_prob_fn, init_position,
            num_steps=L, max_iter=max_iter
        )

        print(f"\n  ADAPTIVE SAMPLING")
        print(f"  Tuned step_size: {step_size:.4f}")

        # 2. Sample with tuned parameters until target ESS
        all_samples_list = []
        all_log_probs_list = []
        total_samples = 0
        batch_num = 0
        current_position = init_position

        while total_samples < max_samples:
            batch_num += 1
            key, sample_key = random.split(key)

            samples_batch, lps_batch, accept_rate, final_state = hmc_run(
                sample_key, log_prob_fn, current_position,
                step_size=step_size, num_steps=L,
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
        max_cycles: Maximum coordinate-wise tuning cycles
        schedule_type: Friction schedule ('constant', 'tanh', 'sigmoid', 'linear', 'sine')
        num_steps_grid: List of num_steps values to try (default: [8, 16, 32, 64])

    Returns:
        Dictionary containing best configuration, grid results, and comparison data
    """
    if num_steps_grid is None:
        num_steps_grid = [8, 16, 32, 64]

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

        # 1. Coordinate-wise tune (step_size, gamma, steepness)
        key, tune_key = random.split(key)
        step_size, gamma, steepness, tune_history = coordinate_wise_tune_grahmc(
            tune_key, log_prob_fn, init_position,
            num_steps=L, schedule_type=schedule_type, max_cycles=max_cycles
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
        current_position = init_position

        while total_samples < max_samples:
            batch_num += 1
            key, sample_key = random.split(key)

            samples_batch, lps_batch, accept_rate, final_state = rahmc_run(
                sample_key, log_prob_fn, current_position,
                step_size=step_size, num_steps=L,
                gamma=gamma, steepness=steepness,
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


def plot_grid_comparison(grid_results: list, num_steps_grid: list, output_file: str = None):
    """Plot grid search comparison of ESS per gradient vs num_steps.

    Args:
        grid_results: List of result dictionaries from grid search
        num_steps_grid: List of num_steps values that were tried
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract metrics
    ess_per_grad = [r['ess_per_gradient'] for r in grid_results]
    ess_per_sample = [r['ess_per_sample'] for r in grid_results]
    total_grad_calls = [r['total_gradient_calls'] for r in grid_results]
    step_sizes = [r['step_size'] for r in grid_results]

    # Find best
    best_idx = np.argmax(ess_per_grad)
    best_L = num_steps_grid[best_idx]

    # Plot 1: ESS per gradient (KEY METRIC)
    axes[0, 0].plot(num_steps_grid, ess_per_grad, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 0].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[0, 0].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[0, 0].set_ylabel('ESS per Gradient Call', fontsize=12)
    axes[0, 0].set_title('Computational Efficiency (KEY METRIC)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: ESS per sample (for comparison)
    axes[0, 1].plot(num_steps_grid, ess_per_sample, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 1].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[0, 1].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[0, 1].set_ylabel('ESS per Sample', fontsize=12)
    axes[0, 1].set_title('ESS per Sample (ignores cost)', fontsize=12)
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Total gradient calls to reach target ESS
    axes[1, 0].bar(range(len(num_steps_grid)), total_grad_calls, color='purple', alpha=0.6)
    axes[1, 0].set_xticks(range(len(num_steps_grid)))
    axes[1, 0].set_xticklabels(num_steps_grid)
    axes[1, 0].axvline(best_idx, color='r', linestyle='--', linewidth=2, label=f'Best: L={best_L}')
    axes[1, 0].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[1, 0].set_ylabel('Total Gradient Calls', fontsize=12)
    axes[1, 0].set_title('Computational Cost to Reach Target ESS', fontsize=12)
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Tuned step sizes
    axes[1, 1].plot(num_steps_grid, step_sizes, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 1].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[1, 1].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[1, 1].set_ylabel('Tuned Step Size', fontsize=12)
    axes[1, 1].set_title('Step Size vs Trajectory Length', fontsize=12)
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('HMC Grid Search: Trajectory Length Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved grid comparison plot to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_grahmc_grid_comparison(grid_results: list, num_steps_grid: list,
                                  schedule_type: str = 'constant',
                                  has_steepness: bool = False,
                                  output_file: str = None):
    """Plot GRAHMC grid search comparison with schedule-specific parameters.

    Args:
        grid_results: List of result dictionaries from grid search
        num_steps_grid: List of num_steps values that were tried
        schedule_type: Friction schedule type
        has_steepness: Whether schedule uses steepness parameter
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")

    # 2x3 layout for schedules with steepness, 2x2 for schedules without
    if has_steepness:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

    # Extract metrics
    ess_per_grad = [r['ess_per_gradient'] for r in grid_results]
    ess_per_sample = [r['ess_per_sample'] for r in grid_results]
    total_grad_calls = [r['total_gradient_calls'] for r in grid_results]
    step_sizes = [r['step_size'] for r in grid_results]
    gammas = [r['gamma'] for r in grid_results]
    if has_steepness:
        steepnesses = [r['steepness'] for r in grid_results]

    # Find best
    best_idx = np.argmax(ess_per_grad)
    best_L = num_steps_grid[best_idx]

    # Plot 1: ESS per gradient (KEY METRIC)
    axes[0].plot(num_steps_grid, ess_per_grad, 'o-', linewidth=2, markersize=8, color='green')
    axes[0].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[0].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[0].set_ylabel('ESS per Gradient Call', fontsize=12)
    axes[0].set_title('Computational Efficiency (KEY METRIC)', fontsize=12, fontweight='bold')
    axes[0].set_xscale('log', base=2)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: ESS per sample
    axes[1].plot(num_steps_grid, ess_per_sample, 'o-', linewidth=2, markersize=8, color='blue')
    axes[1].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[1].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[1].set_ylabel('ESS per Sample', fontsize=12)
    axes[1].set_title('ESS per Sample (ignores cost)', fontsize=12)
    axes[1].set_xscale('log', base=2)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Total gradient calls
    axes[2].bar(range(len(num_steps_grid)), total_grad_calls, color='purple', alpha=0.6)
    axes[2].set_xticks(range(len(num_steps_grid)))
    axes[2].set_xticklabels(num_steps_grid)
    axes[2].axvline(best_idx, color='r', linestyle='--', linewidth=2, label=f'Best: L={best_L}')
    axes[2].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[2].set_ylabel('Total Gradient Calls', fontsize=12)
    axes[2].set_title('Computational Cost to Reach Target ESS', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3, axis='y')

    # Plot 4: Tuned step sizes
    axes[3].plot(num_steps_grid, step_sizes, 'o-', linewidth=2, markersize=8, color='orange')
    axes[3].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[3].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[3].set_ylabel('Tuned Step Size', fontsize=12)
    axes[3].set_title('Step Size vs Trajectory Length', fontsize=12)
    axes[3].set_xscale('log', base=2)
    axes[3].legend(loc='best')
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Tuned gamma values
    if has_steepness:
        ax_gamma = axes[4]
    else:
        ax_gamma = axes[2] if len(axes) > 4 else None
        if ax_gamma is None:
            # Create new subplot position
            ax_gamma = fig.add_subplot(2, 2, 3)

    if ax_gamma is not None:
        ax_gamma.plot(num_steps_grid, gammas, 'o-', linewidth=2, markersize=8, color='cyan')
        ax_gamma.axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
        ax_gamma.set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
        ax_gamma.set_ylabel('Tuned Gamma (friction)', fontsize=12)
        ax_gamma.set_title('Gamma vs Trajectory Length', fontsize=12)
        ax_gamma.set_xscale('log', base=2)
        ax_gamma.legend(loc='best')
        ax_gamma.grid(True, alpha=0.3)

    # Plot 6: Tuned steepness (if applicable)
    if has_steepness:
        axes[5].plot(num_steps_grid, steepnesses, 'o-', linewidth=2, markersize=8, color='magenta')
        axes[5].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
        axes[5].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
        axes[5].set_ylabel('Tuned Steepness', fontsize=12)
        axes[5].set_title('Steepness vs Trajectory Length', fontsize=12)
        axes[5].set_xscale('log', base=2)
        axes[5].legend(loc='best')
        axes[5].grid(True, alpha=0.3)

    plt.suptitle(f'GRAHMC ({schedule_type.upper()}) Grid Search: Trajectory Length Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved grid comparison plot to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_coordinate_tuning_history(history: Dict, output_file: str = None):
    """Plot coordinate-wise tuning history for GRAHMC.

    Args:
        history: Dictionary containing cycle history from coordinate_wise_tune_grahmc
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")

    cycle_history = history['cycle_history']
    schedule_type = history['schedule_type']
    has_steepness = history['has_steepness']
    num_steps = history['num_steps']
    converged_cycle = history['converged_cycle']

    # Extract data
    cycles = np.arange(1, len(cycle_history) + 1)
    step_sizes = [c['step_size'] for c in cycle_history]
    gammas = [c['gamma'] for c in cycle_history]
    steepnesses = [c['steepness'] for c in cycle_history]
    accept_rates = [c['accept_rate'] for c in cycle_history]

    # Create subplots
    n_plots = 4 if has_steepness else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)

    # Plot 1: Step size evolution
    axes[0].plot(cycles, step_sizes, 'o-', linewidth=2, markersize=6, color='orange')
    axes[0].axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (cycle {converged_cycle})')
    axes[0].set_ylabel('Step Size', fontsize=12)
    axes[0].set_title(f'GRAHMC ({schedule_type.upper()}, L={num_steps}) Coordinate-wise Tuning History',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Gamma evolution
    axes[1].plot(cycles, gammas, 'o-', linewidth=2, markersize=6, color='cyan')
    axes[1].axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (cycle {converged_cycle})')
    axes[1].set_ylabel('Gamma (friction)', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Steepness evolution (if applicable)
    if has_steepness:
        axes[2].plot(cycles, steepnesses, 'o-', linewidth=2, markersize=6, color='magenta')
        axes[2].axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                        label=f'Converged (cycle {converged_cycle})')
        axes[2].set_ylabel('Steepness', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

    # Plot 4 (or 3): Acceptance rate evolution
    ax_accept = axes[3] if has_steepness else axes[2]
    ax_accept.plot(cycles, accept_rates, 'o-', linewidth=2, markersize=6, color='green')
    ax_accept.axhline(history['target_accept'], color='orange', linestyle='--', linewidth=1.5,
                      label=f'Target ({history["target_accept"]:.3f})')
    ax_accept.axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                      label=f'Converged (cycle {converged_cycle})')
    ax_accept.set_xlabel('Cycle', fontsize=12)
    ax_accept.set_ylabel('Acceptance Rate', fontsize=12)
    ax_accept.legend(loc='best')
    ax_accept.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved coordinate tuning history plot to {output_file}")
    else:
        plt.show()
    plt.close()


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
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum tuning iterations (default: 2000)"
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
        default=110,
        help="Maximum coordinate-wise tuning cycles for GRAHMC (default: 110)"
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
            max_iter=args.max_iter,
        )
    elif args.sampler == "hmc":
        # Parse num_steps grid (default for HMC)
        if args.num_steps_grid is None:
            num_steps_grid = [8, 16, 32, 64]
        else:
            num_steps_grid = [int(x) for x in args.num_steps_grid.split(',')]
        results = tune_and_sample_hmc_grid(
            key=key,
            target=target,
            n_chains=args.chains,
            target_ess=args.target_ess,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_iter=args.max_iter,
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
            max_iter=args.max_iter,
            max_tree_depth=args.max_tree_depth,
        )
    elif args.sampler == "grahmc":
        # Parse num_steps grid (default for GRAHMC - smaller than HMC)
        if args.num_steps_grid is None:
            num_steps_grid = [8, 16, 32, 64]
        else:
            num_steps_grid = [int(x) for x in args.num_steps_grid.split(',')]
        results = tune_and_sample_grahmc_grid(
            key=key,
            target=target,
            n_chains=args.chains,
            target_ess=args.target_ess,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
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
        print(f"Total gradient calls: {best['total_gradient_calls']}")

    # Generate plots if requested
    if args.plot:
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"GENERATING DIAGNOSTIC PLOTS")
        print(f"{'='*60}")

        if args.sampler == "hmc":
            # HMC: Plot grid comparison
            grid_plot_file = os.path.join(args.output_dir, "hmc_grid_comparison.png")
            plot_grid_comparison(results["grid_results"], results["num_steps_grid"],
                               output_file=grid_plot_file)

            # Plot tuning history for best configuration
            best = results['best_config']
            tuning_plot_file = os.path.join(args.output_dir, f"hmc_best_L{best['num_steps']}_tuning_history.png")
            plot_tuning_history(best["tune_history"], sampler_name=f"HMC (L={best['num_steps']})",
                               output_file=tuning_plot_file)

            # Plot sampling diagnostics for best configuration
            sampling_plot_file = os.path.join(args.output_dir, f"hmc_best_L{best['num_steps']}_sampling_diagnostics.png")
            plot_sampling_diagnostics(best["samples"], best["diagnostics"],
                                     sampler_name=f"HMC (L={best['num_steps']})",
                                     output_file=sampling_plot_file)
        elif args.sampler == "grahmc":
            # GRAHMC: Plot grid comparison with schedule-specific parameters
            schedule_name = args.schedule
            grid_plot_file = os.path.join(args.output_dir, f"grahmc_{schedule_name}_grid_comparison.png")
            plot_grahmc_grid_comparison(results["grid_results"], results["num_steps_grid"],
                                       schedule_type=schedule_name,
                                       has_steepness=results['has_steepness'],
                                       output_file=grid_plot_file)

            # Plot coordinate-wise tuning history for best configuration
            best = results['best_config']
            tuning_plot_file = os.path.join(args.output_dir,
                                           f"grahmc_{schedule_name}_best_L{best['num_steps']}_tuning_history.png")
            plot_coordinate_tuning_history(best["tune_history"], output_file=tuning_plot_file)

            # Plot sampling diagnostics for best configuration
            sampling_plot_file = os.path.join(args.output_dir,
                                             f"grahmc_{schedule_name}_best_L{best['num_steps']}_sampling_diagnostics.png")
            plot_sampling_diagnostics(best["samples"], best["diagnostics"],
                                     sampler_name=f"GRAHMC-{schedule_name.upper()} (L={best['num_steps']})",
                                     output_file=sampling_plot_file)
        else:
            # RWMH/NUTS: Standard plots
            tuning_plot_file = os.path.join(args.output_dir, f"{args.sampler}_tuning_history.png")
            plot_tuning_history(results["history"], sampler_name=args.sampler.upper(),
                               output_file=tuning_plot_file)

            sampling_plot_file = os.path.join(args.output_dir, f"{args.sampler}_sampling_diagnostics.png")
            plot_sampling_diagnostics(results["samples"], results["diagnostics"],
                                     sampler_name=args.sampler.upper(),
                                     output_file=sampling_plot_file)

        print(f"\nPlots saved to {args.output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
