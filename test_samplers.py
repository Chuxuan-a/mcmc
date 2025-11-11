"""Test suite for MCMC samplers.

Tests samplers on a 10D standard normal distribution with:
- Dual averaging for automatic step size tuning
- Split rank-normalized R-hat convergence diagnostics
- Summary statistics validation against known true values
"""
import argparse
import sys
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import arviz as az

# Import samplers
from samplers.RWMH import rwMH_run
from samplers.HMC import hmc_run
from samplers.GRAHMC import (
    rahmc_run,
    constant_schedule,
    tanh_schedule,
    sigmoid_schedule,
    linear_schedule,
    sine_schedule,
)
from samplers.NUTS import nuts_run, nuts_init, nuts_step


def standard_normal_log_prob(x: jnp.ndarray) -> jnp.ndarray:
    """Log probability of standard normal N(0, I)."""
    D = x.shape[-1]
    return -0.5 * (jnp.sum(x**2, axis=-1) + D * jnp.log(2.0 * jnp.pi))


def dual_averaging_tune_rwmh(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    target_accept: float = 0.234,  # Optimal for RWMH
    tolerance: float = 0.01,
    max_iter: int = 2000,
    min_iter: int = 100,
    patience: int = 10,
) -> float:
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
        Tuned scale parameter
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

    # Run tuning iterations until convergence
    n_samples_per_tune = 20  # More samples for better acceptance estimate
    converged_count = 0

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

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_scale_bar - prev_scale_bar) / (abs(prev_scale_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: scale={current_scale_bar:.4f}, accept={alpha:.3f}")
                return current_scale_bar

        prev_scale_bar = current_scale_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: scale={current_scale_bar:.4f}, accept={alpha:.3f}")

    final_scale = float(jnp.exp(log_scale_bar))
    print(f"  Reached max iterations ({max_iter}): scale={final_scale:.4f}")
    return final_scale


def dual_averaging_tune_grahmc_step_size(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    num_steps: int,
    gamma: float,
    steepness: float,
    friction_schedule,
    target_accept: float = 0.65,
    tolerance: float = 0.01,
    max_iter: int = 500,
    min_iter: int = 50,
    patience: int = 10,
) -> float:
    """Tune GRAHMC step size via dual averaging (gamma, steepness fixed).

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        num_steps: Number of leapfrog steps (fixed)
        gamma: Friction amplitude (fixed)
        steepness: Transition sharpness parameter (fixed, can be None)
        friction_schedule: Friction schedule function
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance
        max_iter: Maximum tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Consecutive converged iterations required

    Returns:
        Tuned step size
    """
    # Dual averaging parameters (from Stan)
    gamma_da = 0.05  # renamed to avoid confusion with friction gamma
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

    # Run tuning iterations until convergence
    n_samples_per_tune = 20  # More samples for better acceptance estimate
    converged_count = 0

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        _, _, accept_rate, _ = rahmc_run(
            subkey, log_prob_fn, init_position,
            step_size=float(step_size), num_steps=num_steps,
            gamma=gamma, steepness=steepness,
            num_samples=n_samples_per_tune, burn_in=0,
            friction_schedule=friction_schedule
        )
        alpha = float(jnp.mean(accept_rate))

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_step_size = mu - (jnp.sqrt(m) / gamma_da) * H_bar
        m_kappa = m ** (-kappa)
        log_step_size_bar = m_kappa * log_step_size + (1 - m_kappa) * log_step_size_bar

        step_size = jnp.exp(log_step_size)
        current_step_size_bar = float(jnp.exp(log_step_size_bar))

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_step_size_bar - prev_step_size_bar) / (abs(prev_step_size_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")
                return current_step_size_bar

        prev_step_size_bar = current_step_size_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")

    final_step_size = float(jnp.exp(log_step_size_bar))
    print(f"  Reached max iterations ({max_iter}): step_size={final_step_size:.4f}")
    return final_step_size


def dual_averaging_tune_grahmc_steepness(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    num_steps: int,
    step_size: float,
    gamma: float,
    friction_schedule,
    target_accept: float = 0.65,
    tolerance: float = 0.01,
    max_iter: int = 500,
    min_iter: int = 50,
    patience: int = 10,
) -> float:
    """Tune GRAHMC steepness parameter via dual averaging (step_size, gamma fixed).

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        num_steps: Number of leapfrog steps (fixed)
        step_size: Integration step size (fixed)
        gamma: Friction amplitude (fixed)
        friction_schedule: Friction schedule function
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance
        max_iter: Maximum tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Consecutive converged iterations required

    Returns:
        Tuned steepness parameter
    """
    # Dual averaging parameters (from Stan)
    gamma_da = 0.05
    t0 = 10.0
    kappa = 0.75

    # Initialize steepness around 5.0 for tanh, 10.0 for sigmoid
    # Use a reasonable default
    initial_steepness = 5.0
    log_steepness = jnp.log(initial_steepness)
    mu = log_steepness
    log_steepness_bar = 0.0
    H_bar = 0.0

    steepness = jnp.exp(log_steepness)
    prev_steepness_bar = float(steepness)

    # Run tuning iterations until convergence
    n_samples_per_tune = 20
    converged_count = 0

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        _, _, accept_rate, _ = rahmc_run(
            subkey, log_prob_fn, init_position,
            step_size=step_size, num_steps=num_steps,
            gamma=gamma, steepness=float(steepness),
            num_samples=n_samples_per_tune, burn_in=0,
            friction_schedule=friction_schedule
        )
        alpha = float(jnp.mean(accept_rate))

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_steepness = mu - (jnp.sqrt(m) / gamma_da) * H_bar
        m_kappa = m ** (-kappa)
        log_steepness_bar = m_kappa * log_steepness + (1 - m_kappa) * log_steepness_bar

        steepness = jnp.exp(log_steepness)
        current_steepness_bar = float(jnp.exp(log_steepness_bar))

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_steepness_bar - prev_steepness_bar) / (abs(prev_steepness_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: steepness={current_steepness_bar:.4f}, accept={alpha:.3f}")
                return current_steepness_bar

        prev_steepness_bar = current_steepness_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: steepness={current_steepness_bar:.4f}, accept={alpha:.3f}")

    final_steepness = float(jnp.exp(log_steepness_bar))
    print(f"  Reached max iterations ({max_iter}): steepness={final_steepness:.4f}")
    return final_steepness


def dual_averaging_tune_grahmc_gamma(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    num_steps: int,
    step_size: float,
    steepness: float,
    friction_schedule,
    target_accept: float = 0.65,
    tolerance: float = 0.01,
    max_iter: int = 500,
    min_iter: int = 50,
    patience: int = 10,
) -> float:
    """Tune GRAHMC gamma (friction amplitude) via dual averaging (step_size, steepness fixed).

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        num_steps: Number of leapfrog steps (fixed)
        step_size: Integration step size (fixed)
        steepness: Transition sharpness parameter (fixed, can be None)
        friction_schedule: Friction schedule function
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance
        max_iter: Maximum tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Consecutive converged iterations required

    Returns:
        Tuned gamma parameter
    """
    # Dual averaging parameters (from Stan)
    gamma_da = 0.05
    t0 = 10.0
    kappa = 0.75

    # Initialize gamma around 0.5 (reasonable for many problems)
    initial_gamma = 0.5
    log_gamma = jnp.log(initial_gamma)
    mu = log_gamma  # Target around initial
    log_gamma_bar = 0.0
    H_bar = 0.0

    gamma = jnp.exp(log_gamma)
    prev_gamma_bar = float(gamma)

    # Run tuning iterations until convergence
    n_samples_per_tune = 20
    converged_count = 0

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        _, _, accept_rate, _ = rahmc_run(
            subkey, log_prob_fn, init_position,
            step_size=step_size, num_steps=num_steps,
            gamma=float(gamma), steepness=steepness,
            num_samples=n_samples_per_tune, burn_in=0,
            friction_schedule=friction_schedule
        )
        alpha = float(jnp.mean(accept_rate))

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_gamma = mu - (jnp.sqrt(m) / gamma_da) * H_bar
        m_kappa = m ** (-kappa)
        log_gamma_bar = m_kappa * log_gamma + (1 - m_kappa) * log_gamma_bar

        gamma = jnp.exp(log_gamma)
        current_gamma_bar = float(jnp.exp(log_gamma_bar))

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_gamma_bar - prev_gamma_bar) / (abs(prev_gamma_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: gamma={current_gamma_bar:.4f}, accept={alpha:.3f}")
                return current_gamma_bar

        prev_gamma_bar = current_gamma_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: gamma={current_gamma_bar:.4f}, accept={alpha:.3f}")

    final_gamma = float(jnp.exp(log_gamma_bar))
    print(f"  Reached max iterations ({max_iter}): gamma={final_gamma:.4f}")
    return final_gamma


def dual_averaging_tune_grahmc(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    num_steps: int = 20,
    friction_schedule = constant_schedule,
    target_accept: float = 0.65,
    tolerance: float = 0.01,
    max_cycles: int = 10,
    min_cycles: int = 2,
    patience: int = 2,
) -> Tuple[float, float] | Tuple[float, float, float]:
    """Tune GRAHMC parameters via coordinate-wise dual averaging until convergence.

    For schedules with steepness (tanh, sigmoid): tunes (step_size, gamma, steepness)
    For schedules without steepness (constant, linear, sine): tunes (step_size, gamma)

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        num_steps: Number of leapfrog steps (fixed, not tuned)
        friction_schedule: Friction schedule function
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance (relative change in parameters)
        max_cycles: Maximum number of alternating cycles
        min_cycles: Minimum cycles before checking convergence
        patience: Number of consecutive converged cycles required

    Returns:
        If schedule uses steepness: (step_size, gamma, steepness)
        Otherwise: (step_size, gamma)
    """
    # Check which schedule type we're using
    schedule_uses_steepness = friction_schedule in [tanh_schedule, sigmoid_schedule]

    # Initialize parameters
    gamma = 0.5
    d = init_position.shape[-1]
    step_size = 0.5 / jnp.sqrt(d)
    steepness = 5.0 if friction_schedule == tanh_schedule else 10.0 if friction_schedule == sigmoid_schedule else None

    if schedule_uses_steepness:
        print(f"  Starting coordinate-wise tuning (will iterate until convergence)...")
        print(f"  Initial: step_size={step_size:.4f}, gamma={gamma:.4f}, steepness={steepness:.4f}")
    else:
        print(f"  Starting coordinate-wise tuning (will iterate until convergence)...")
        print(f"  Initial: step_size={step_size:.4f}, gamma={gamma:.4f}")

    prev_step_size = step_size
    prev_gamma = gamma
    prev_steepness = steepness
    converged_count = 0

    for cycle in range(max_cycles):
        print(f"\n  --- Cycle {cycle + 1} ---")

        # Tune step_size with fixed gamma and steepness
        if schedule_uses_steepness:
            print(f"  Tuning step_size (gamma={gamma:.4f}, steepness={steepness:.4f} fixed)...")
        else:
            print(f"  Tuning step_size (gamma={gamma:.4f} fixed)...")
        key, subkey = random.split(key)
        step_size = dual_averaging_tune_grahmc_step_size(
            subkey, log_prob_fn, init_position,
            num_steps=num_steps, gamma=gamma,
            steepness=steepness, friction_schedule=friction_schedule,
            target_accept=target_accept
        )

        # Tune gamma with fixed step_size and steepness
        if schedule_uses_steepness:
            print(f"  Tuning gamma (step_size={step_size:.4f}, steepness={steepness:.4f} fixed)...")
        else:
            print(f"  Tuning gamma (step_size={step_size:.4f} fixed)...")
        key, subkey = random.split(key)
        gamma = dual_averaging_tune_grahmc_gamma(
            subkey, log_prob_fn, init_position,
            num_steps=num_steps, step_size=step_size,
            steepness=steepness, friction_schedule=friction_schedule,
            target_accept=target_accept
        )

        # Tune steepness with fixed step_size and gamma (only for tanh/sigmoid)
        if schedule_uses_steepness:
            print(f"  Tuning steepness (step_size={step_size:.4f}, gamma={gamma:.4f} fixed)...")
            key, subkey = random.split(key)
            steepness = dual_averaging_tune_grahmc_steepness(
                subkey, log_prob_fn, init_position,
                num_steps=num_steps, step_size=step_size,
                gamma=gamma, friction_schedule=friction_schedule,
                target_accept=target_accept
            )

        # Check convergence after minimum cycles
        if cycle >= min_cycles - 1:
            step_size_change = abs(step_size - prev_step_size) / (abs(prev_step_size) + 1e-10)
            gamma_change = abs(gamma - prev_gamma) / (abs(prev_gamma) + 1e-10)

            if schedule_uses_steepness:
                steepness_change = abs(steepness - prev_steepness) / (abs(prev_steepness) + 1e-10)
                print(f"  Parameter changes: step_size={step_size_change:.6f}, gamma={gamma_change:.6f}, steepness={steepness_change:.6f}")
                all_stable = (step_size_change < tolerance and
                            gamma_change < tolerance and
                            steepness_change < tolerance)
            else:
                print(f"  Parameter changes: step_size={step_size_change:.6f}, gamma={gamma_change:.6f}")
                all_stable = step_size_change < tolerance and gamma_change < tolerance

            if all_stable:
                converged_count += 1
                print(f"  All parameters stable ({converged_count}/{patience})")
            else:
                converged_count = 0
                print(f"  Parameters still changing, continuing...")

            if converged_count >= patience:
                print(f"\n  Converged after {cycle + 1} cycles!")
                if schedule_uses_steepness:
                    print(f"  Final: step_size={step_size:.4f}, gamma={gamma:.4f}, steepness={steepness:.4f}")
                    return step_size, gamma, steepness
                else:
                    print(f"  Final: step_size={step_size:.4f}, gamma={gamma:.4f}")
                    return step_size, gamma

        prev_step_size = step_size
        prev_gamma = gamma
        prev_steepness = steepness

    print(f"\n  Reached max cycles ({max_cycles})")
    if schedule_uses_steepness:
        print(f"  Final: step_size={step_size:.4f}, gamma={gamma:.4f}, steepness={steepness:.4f}")
        return step_size, gamma, steepness
    else:
        print(f"  Final: step_size={step_size:.4f}, gamma={gamma:.4f}")
        return step_size, gamma


def dual_averaging_tune_hmc(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    num_steps: int = 20,
    target_accept: float = 0.65,  # Optimal for HMC
    tolerance: float = 0.01,
    max_iter: int = 2000,
    min_iter: int = 100,
    patience: int = 10,
) -> float:
    """Tune HMC step size using dual averaging until convergence.

    Based on Hoffman & Gelman (2014) NUTS paper, Section 3.2.
    Note: Integration length (num_steps) is kept fixed as it requires grid search,
    not continuous optimization.

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        num_steps: Number of leapfrog steps (fixed, not tuned)
        target_accept: Target acceptance rate
        tolerance: Convergence tolerance (relative change in parameter)
        max_iter: Maximum number of tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Number of consecutive converged iterations required

    Returns:
        Tuned step size parameter
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

    # Run tuning iterations until convergence
    n_samples_per_tune = 20  # More samples for better acceptance estimate
    converged_count = 0

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        _, _, accept_rate, _ = hmc_run(
            subkey, log_prob_fn, init_position,
            step_size=float(step_size), num_steps=num_steps,
            num_samples=n_samples_per_tune, burn_in=0
        )
        alpha = float(jnp.mean(accept_rate))

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_step_size = mu - (jnp.sqrt(m) / gamma) * H_bar
        m_kappa = m ** (-kappa)
        log_step_size_bar = m_kappa * log_step_size + (1 - m_kappa) * log_step_size_bar

        step_size = jnp.exp(log_step_size)
        current_step_size_bar = float(jnp.exp(log_step_size_bar))

        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_step_size_bar - prev_step_size_bar) / (abs(prev_step_size_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")
                return current_step_size_bar

        prev_step_size_bar = current_step_size_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")

    final_step_size = float(jnp.exp(log_step_size_bar))
    print(f"  Reached max iterations ({max_iter}): step_size={final_step_size:.4f}")
    return final_step_size


def dual_averaging_tune_nuts(
    key: jnp.ndarray,
    log_prob_fn,
    init_position: jnp.ndarray,
    max_tree_depth: int = 10,
    target_accept: float = 0.65,  # Target acceptance rate (same as HMC)
    tolerance: float = 0.01,
    max_iter: int = 2000,
    min_iter: int = 100,
    patience: int = 10,
) -> float:
    """Tune NUTS step size using dual averaging until convergence.

    Based on Hoffman & Gelman (2014) NUTS paper, Section 3.2.
    Note: NUTS automatically selects trajectory length, so only step size is tuned.

    Args:
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        max_tree_depth: Maximum tree depth (default: 10)
        target_accept: Target acceptance rate (NUTS aims higher due to slice sampling)
        tolerance: Convergence tolerance (relative change in parameter)
        max_iter: Maximum number of tuning iterations
        min_iter: Minimum iterations before checking convergence
        patience: Number of consecutive converged iterations required

    Returns:
        Tuned step size parameter
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

    # Run tuning iterations until convergence
    n_samples_per_tune = 20  # More samples for better acceptance estimate
    converged_count = 0

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

        # Dual averaging update
        eta_m = 1.0 / (m + t0)
        H_bar = (1 - eta_m) * H_bar + eta_m * (target_accept - alpha)
        log_step_size = mu - (jnp.sqrt(m) / gamma) * H_bar
        m_kappa = m ** (-kappa)
        log_step_size_bar = m_kappa * log_step_size + (1 - m_kappa) * log_step_size_bar

        step_size = jnp.exp(log_step_size)
        current_step_size_bar = float(jnp.exp(log_step_size_bar))


        # Check convergence after minimum iterations
        if m >= min_iter:
            relative_change = abs(current_step_size_bar - prev_step_size_bar) / (abs(prev_step_size_bar) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                print(f"  Converged after {m} iterations: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")
                return current_step_size_bar

        prev_step_size_bar = current_step_size_bar

        if m % 200 == 0:
            print(f"  Tuning iteration {m}: step_size={current_step_size_bar:.4f}, accept={alpha:.3f}")

    final_step_size = float(jnp.exp(log_step_size_bar))
    print(f"  Reached max iterations ({max_iter}): step_size={final_step_size:.4f}")
    return final_step_size


def run_sampler(
    sampler: str,
    key: jnp.ndarray,
    n_dim: int = 10,
    n_chains: int = 4,
    n_tune: int = 1000,
    target_ess: int = 1000,
    batch_size: int = 2000,
    max_samples: int = 50000,
    schedule_type: str = "constant",
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Run specified sampler with adaptive sampling until target ESS is reached.

    Args:
        sampler: Name of sampler ('rwmh', 'hmc', 'rahmc', 'grahmc', 'nuts')
        key: JAX random key
        n_dim: Dimensionality of target distribution
        n_chains: Number of parallel chains
        n_tune: Number of tuning iterations
        target_ess: Target minimum bulk ESS per dimension
        batch_size: Number of samples per batch
        max_samples: Maximum total samples to collect
        schedule_type: Friction schedule type for GRAHMC
                      ('constant', 'tanh', 'sigmoid', 'linear', 'sine')

    Returns:
        Tuple of (samples, log_probs, metadata)
    """
    # Initialize chains (overdispersed start)
    key, init_key = random.split(key)
    init_position = random.normal(init_key, shape=(n_chains, n_dim)) * 2.0

    print(f"\n{'='*60}")
    print(f"Testing {sampler.upper()} sampler")
    print(f"{'='*60}")
    print(f"Target: {n_dim}D Standard Normal")
    print(f"Chains: {n_chains}")
    print(f"Tuning iterations: {n_tune}")
    print(f"Target ESS: {target_ess}")

    metadata = {"sampler": sampler, "n_dim": n_dim, "n_chains": n_chains}

    # Tune parameters
    if sampler == "rwmh":
        print("\nTuning proposal scale...")
        key, tune_key = random.split(key)
        scale = dual_averaging_tune_rwmh(
            tune_key, standard_normal_log_prob, init_position
        )
        metadata["scale"] = scale
        param_str = f"scale={scale:.4f}"
    elif sampler == "hmc":
        num_steps = 20  # Fixed trajectory length (not tuned - requires grid search)
        print(f"\nTuning step size (L={num_steps} fixed)...")
        key, tune_key = random.split(key)
        step_size = dual_averaging_tune_hmc(
            tune_key, standard_normal_log_prob, init_position,
            num_steps=num_steps
        )
        metadata["step_size"] = step_size
        metadata["num_steps"] = num_steps
        param_str = f"step_size={step_size:.4f}, L={num_steps}"
    elif sampler == "nuts":
        max_tree_depth = 10  # Maximum tree depth (max trajectory = 2^10 = 1024 steps)
        print(f"\nTuning step size (max_tree_depth={max_tree_depth} fixed)...")
        key, tune_key = random.split(key)
        step_size = dual_averaging_tune_nuts(
            tune_key, standard_normal_log_prob, init_position,
            max_tree_depth=max_tree_depth
        )
        metadata["step_size"] = step_size
        metadata["max_tree_depth"] = max_tree_depth
        param_str = f"step_size={step_size:.4f}, max_depth={max_tree_depth}"
    elif sampler in ["grahmc", "rahmc"]:
        # Map schedule type to schedule function
        schedule_map = {
            "constant": constant_schedule,
            "tanh": tanh_schedule,
            "sigmoid": sigmoid_schedule,
            "linear": linear_schedule,
            "sine": sine_schedule,
        }
        friction_schedule = schedule_map[schedule_type]

        num_steps = 20  # Fixed trajectory length
        print(f"\nTuning parameters for {schedule_type} friction schedule via coordinate-wise dual averaging (L={num_steps} fixed)...")
        key, tune_key = random.split(key)

        # Tune parameters (returns 2 or 3 values depending on schedule)
        tuned_params = dual_averaging_tune_grahmc(
            tune_key, standard_normal_log_prob, init_position,
            num_steps=num_steps, friction_schedule=friction_schedule
        )

        # Unpack based on number of returned values
        if len(tuned_params) == 3:
            step_size, gamma, steepness = tuned_params
            metadata["step_size"] = step_size
            metadata["num_steps"] = num_steps
            metadata["gamma"] = gamma
            metadata["steepness"] = steepness
            metadata["schedule_type"] = schedule_type
            param_str = f"step_size={step_size:.4f}, L={num_steps}, gamma={gamma:.4f}, steepness={steepness:.4f}"
        else:
            step_size, gamma = tuned_params
            steepness = None
            metadata["step_size"] = step_size
            metadata["num_steps"] = num_steps
            metadata["gamma"] = gamma
            metadata["schedule_type"] = schedule_type
            param_str = f"step_size={step_size:.4f}, L={num_steps}, gamma={gamma:.4f}"
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Adaptive sampling until target ESS is reached
    print(f"\nSampling adaptively until ESS >= {target_ess}...")
    print(f"(collecting in batches of {batch_size} samples)")

    all_samples_list = []
    all_log_probs_list = []
    total_samples = 0
    batch_num = 0
    current_position = init_position  # Track chain state across batches

    while total_samples < max_samples:
        batch_num += 1
        key, sample_key = random.split(key)

        if sampler == "rwmh":
            samples_batch, lps_batch, accept_rate, final_state = rwMH_run(
                sample_key, standard_normal_log_prob, current_position,
                num_samples=batch_size, scale=scale, burn_in=0
            )
        elif sampler == "hmc":
            samples_batch, lps_batch, accept_rate, final_state = hmc_run(
                sample_key, standard_normal_log_prob, current_position,
                step_size=step_size, num_steps=num_steps,
                num_samples=batch_size, burn_in=0
            )
        elif sampler == "nuts":
            samples_batch, lps_batch, accept_rate, final_state, _, _ = nuts_run(
                sample_key, standard_normal_log_prob, current_position,
                step_size=step_size, max_tree_depth=max_tree_depth,
                num_samples=batch_size, burn_in=0
            )
        elif sampler in ["grahmc", "rahmc"]:
            samples_batch, lps_batch, accept_rate, final_state = rahmc_run(
                sample_key, standard_normal_log_prob, current_position,
                step_size=step_size, num_steps=num_steps,
                gamma=gamma, steepness=steepness,
                num_samples=batch_size, burn_in=0,
                friction_schedule=friction_schedule
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

    # Final results
    samples = jnp.concatenate(all_samples_list, axis=0)
    log_probs = jnp.concatenate(all_log_probs_list, axis=0)
    metadata["total_samples"] = total_samples
    metadata["accept_rate"] = float(jnp.mean(accept_rate))  # Last batch accept rate

    print(f"\nFinal: {total_samples} samples collected with {param_str}")
    print(f"Mean acceptance rate (last batch): {metadata['accept_rate']:.3f}")

    return samples, log_probs, metadata


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
        "summary": summary,
    }

    return diagnostics


def check_summary_statistics(diagnostics: Dict, n_dim: int, tolerance: float = 0.1) -> bool:
    """Check if inferred statistics match true values.

    For standard normal: true mean = 0, true std = 1

    Args:
        diagnostics: Dictionary containing summary statistics
        n_dim: Dimensionality
        tolerance: Acceptable deviation from true values

    Returns:
        True if all checks pass, False otherwise
    """
    summary = diagnostics["summary"]

    print("\n" + "="*60)
    print("SUMMARY STATISTICS CHECK")
    print("="*60)
    print(f"True values: mean=0.0, std=1.0")
    print(f"Tolerance: +/-{tolerance}")

    all_pass = True

    # Check means
    means = summary["mean"].values
    mean_errors = np.abs(means)
    max_mean_error = np.max(mean_errors)
    mean_check = max_mean_error < tolerance

    print(f"\nMean errors: max={max_mean_error:.4f}")
    print(f"  Status: {'PASS' if mean_check else 'FAIL'}")
    if not mean_check:
        all_pass = False
        bad_dims = np.where(mean_errors >= tolerance)[0]
        print(f"  Failed dimensions: {bad_dims.tolist()}")

    # Check standard deviations
    stds = summary["sd"].values
    std_errors = np.abs(stds - 1.0)
    max_std_error = np.max(std_errors)
    std_check = max_std_error < tolerance

    print(f"\nStd dev errors: max={max_std_error:.4f}")
    print(f"  Status: {'PASS' if std_check else 'FAIL'}")
    if not std_check:
        all_pass = False
        bad_dims = np.where(std_errors >= tolerance)[0]
        print(f"  Failed dimensions: {bad_dims.tolist()}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Test MCMC samplers on standard normal")
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        choices=["rwmh", "hmc", "rahmc", "grahmc", "nuts"],
        help="Sampler to test"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="constant",
        choices=["constant", "tanh", "sigmoid", "linear", "sine"],
        help="Friction schedule for GRAHMC/RAHMC (default: constant)"
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionality (default: 10)")
    parser.add_argument("--chains", type=int, default=4, help="Number of chains (default: 4)")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning iterations (default: 1000)")
    parser.add_argument("--target-ess", type=int, default=1000, help="Target minimum ESS (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Samples per batch (default: 2000)")
    parser.add_argument("--max-samples", type=int, default=50000, help="Maximum total samples (default: 50000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(args.seed)

    # Run sampler
    samples, log_probs, metadata = run_sampler(
        sampler=args.sampler,
        key=key,
        n_dim=args.dim,
        n_chains=args.chains,
        n_tune=args.tune,
        target_ess=args.target_ess,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        schedule_type=args.schedule,
    )

    # Compute diagnostics
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)
    diagnostics = compute_diagnostics(samples)

    print(f"\nSplit R-hat (rank-normalized):")
    print(f"  Max: {diagnostics['rhat_max']:.4f}")
    print(f"  Mean: {diagnostics['rhat_mean']:.4f}")
    rhat_pass = diagnostics['rhat_max'] < 1.01  # Standard threshold
    print(f"  Status: {'PASS' if rhat_pass else 'FAIL'} (threshold: 1.01)")

    print(f"\nEffective Sample Size (bulk):")
    print(f"  Min: {diagnostics['ess_bulk_min']:.1f}")
    print(f"  Mean: {diagnostics['ess_bulk_mean']:.1f}")
    # ESS should be at least the target we sampled for
    ess_pass = diagnostics['ess_bulk_min'] >= args.target_ess
    print(f"  Status: {'PASS' if ess_pass else 'FAIL'} (threshold: {args.target_ess})")

    print(f"\nEffective Sample Size (tail):")
    print(f"  Min: {diagnostics['ess_tail_min']:.1f}")
    ess_tail_pass = diagnostics['ess_tail_min'] >= args.target_ess  # Same threshold
    print(f"  Status: {'PASS' if ess_tail_pass else 'FAIL'} (threshold: {args.target_ess})")

    # Check summary statistics
    stats_pass = check_summary_statistics(diagnostics, args.dim, tolerance=0.10)

    # Overall result
    print("\n" + "="*60)
    print("OVERALL RESULT")
    print("="*60)
    all_pass = rhat_pass and ess_pass and ess_tail_pass and stats_pass
    if all_pass:
        print("[PASS] ALL TESTS PASSED")
        return 0
    else:
        print("[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
