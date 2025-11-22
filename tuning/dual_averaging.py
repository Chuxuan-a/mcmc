"""Dual averaging parameter tuning for MCMC samplers."""
from typing import Tuple, Dict, NamedTuple, Optional, List

import jax
import jax.numpy as jnp
from jax import random

# Import samplers
from samplers.RWMH import rwMH_run
from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run, get_friction_schedule


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
    log_scale_bar = log_scale  # Initialize to initial value, not 0.0
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

    # Track evolving chain position (don't always restart from init_position)
    current_position = init_position

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        _, _, accept_rate, final_state = rwMH_run(
            subkey, log_prob_fn, current_position, num_samples=n_samples_per_tune, scale=float(scale), burn_in=0
        )
        current_position = final_state.position  # Evolve chain
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
    log_step_size_bar = log_step_size  # Initialize to initial value, not 0.0
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

    # Track evolving chain position (don't always restart from init_position)
    current_position = init_position

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        # Run HMC to collect samples and get acceptance statistics
        _, _, accept_rate, final_state = hmc_run(
            subkey, log_prob_fn, current_position,
            step_size=float(step_size), num_steps=num_steps,
            num_samples=n_samples_per_tune, burn_in=0
        )
        current_position = final_state.position  # Evolve chain
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
    log_step_size_bar = log_step_size  # Initialize to initial value, not 0.0
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

    # Track evolving chain position (don't always restart from init_position)
    current_position = init_position

    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        # Run NUTS to collect samples and get acceptance statistics
        _, _, _, final_state, tree_depths, mean_accept_probs = nuts_run(
            subkey, log_prob_fn, current_position,
            step_size=float(step_size), max_tree_depth=max_tree_depth,
            num_samples=n_samples_per_tune, burn_in=0
        )
        current_position = final_state.position  # Evolve chain
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


# def coordinate_wise_tune_grahmc(
#     key: jnp.ndarray,
#     log_prob_fn,
#     grad_log_prob_fn,
#     init_position: jnp.ndarray,
#     num_steps: int,
#     schedule_type: str = 'constant',
#     target_accept: float = 0.65,
#     tolerance: float = 0.02,
#     max_cycles: int = 110,
#     min_cycles: int = 3,
#     patience: int = 2,
#     max_iter_per_param: int = 400,
#     inv_mass_matrix: jnp.ndarray = None,
# ) -> Tuple[float, float, float, Dict]:
#     """Tune GRAHMC hyperparameters using coordinate-wise dual averaging.

#     Cycles through optimizing step_size, gamma, and steepness (if applicable)
#     until convergence. Parameters are optimized one at a time using dual averaging
#     while holding others fixed.

#     Args:
#         key: JAX random key
#         log_prob_fn: Target log probability function
#         grad_log_prob_fn: Gradient of log probability function
#         init_position: Initial positions (n_chains, n_dim)
#         num_steps: Fixed number of leapfrog steps
#         schedule_type: Friction schedule type ('constant', 'tanh', 'sigmoid', 'linear', 'sine')
#         target_accept: Target acceptance rate
#         tolerance: Convergence tolerance (relative change)
#         max_cycles: Maximum number of coordinate cycles
#         min_cycles: Minimum cycles before checking convergence
#         patience: Number of converged cycles required
#         max_iter_per_param: Maximum dual averaging iterations per parameter
#         inv_mass_matrix: Optional inverse mass matrix (diagonal)

#     Returns:
#         Tuple of (step_size, gamma, steepness, history) where history contains:
#         - cycle_history: List of (step_size, gamma, steepness, accept_rate) per cycle
#         - converged_cycle: Cycle where convergence occurred
#     """
#     # Get friction schedule function and determine if it uses steepness
#     friction_schedule = get_friction_schedule(schedule_type)
#     has_steepness = schedule_type in ['tanh', 'sigmoid']

#     # Dual averaging parameters
#     gamma_da = 0.05
#     t0 = 10.0
#     kappa = 0.75

#     # Initialize parameters
#     d = init_position.shape[-1]
#     step_size = 0.5 / jnp.sqrt(d)
#     gamma = 1.0
#     steepness = 5.0 if schedule_type == 'tanh' else (10.0 if schedule_type == 'sigmoid' else 1.0)

#     # Track history
#     cycle_history = []
#     converged_count = 0
#     converged_cycle = max_cycles

#     # Previous values for convergence check
#     prev_step_size = step_size
#     prev_gamma = gamma
#     prev_steepness = steepness

#     n_samples_per_tune = 100

#     # Track evolving chain position (don't always restart from init_position)
#     current_position = init_position

#     for cycle in range(max_cycles):
#         print(f"    Cycle {cycle + 1}/{max_cycles}")

#         # ===== 1. Tune step_size (hold gamma, steepness fixed) =====
#         log_step_size = jnp.log(step_size)
#         mu_step = log_step_size
#         log_step_size_bar = log_step_size  # Initialize to current value, not 0.0
#         H_bar_step = 0.0

#         for m in range(1, max_iter_per_param + 1):
#             key, subkey = random.split(key)
#             _, _, accept_rate, final_state = rahmc_run(
#                 subkey, log_prob_fn, current_position,
#                 step_size=float(jnp.exp(log_step_size)), num_steps=num_steps,
#                 gamma=float(gamma), steepness=float(steepness),
#                 num_samples=n_samples_per_tune, burn_in=0,
#                 friction_schedule=friction_schedule,
#                 inv_mass_matrix=inv_mass_matrix
#             )
#             current_position = final_state.position  # Evolve chain
#             alpha = float(jnp.mean(accept_rate))

#             eta_m = 1.0 / (m + t0)
#             H_bar_step = (1 - eta_m) * H_bar_step + eta_m * (target_accept - alpha)
#             log_step_size = mu_step - (jnp.sqrt(m) / gamma_da) * H_bar_step
#             m_kappa = m ** (-kappa)
#             log_step_size_bar = m_kappa * log_step_size + (1 - m_kappa) * log_step_size_bar

#         step_size = float(jnp.exp(log_step_size_bar))

#         # ===== 2. Tune gamma (hold step_size, steepness fixed) =====
#         log_gamma = jnp.log(gamma)
#         mu_gamma = log_gamma
#         log_gamma_bar = log_gamma  # Initialize to current value, not 0.0
#         H_bar_gamma = 0.0

#         for m in range(1, max_iter_per_param + 1):
#             key, subkey = random.split(key)
#             _, _, accept_rate, final_state = rahmc_run(
#                 subkey, log_prob_fn, current_position,
#                 step_size=float(step_size), num_steps=num_steps,
#                 gamma=float(jnp.exp(log_gamma)), steepness=float(steepness),
#                 num_samples=n_samples_per_tune, burn_in=0,
#                 friction_schedule=friction_schedule,
#                 inv_mass_matrix=inv_mass_matrix
#             )
#             current_position = final_state.position  # Evolve chain
#             alpha = float(jnp.mean(accept_rate))

#             eta_m = 1.0 / (m + t0)
#             H_bar_gamma = (1 - eta_m) * H_bar_gamma + eta_m * (target_accept - alpha)
#             log_gamma = mu_gamma - (jnp.sqrt(m) / gamma_da) * H_bar_gamma
#             # Clip to prevent explosion: gamma in [0.01, 50.0]
#             log_gamma = jnp.clip(log_gamma, jnp.log(0.01), jnp.log(50.0))
#             m_kappa = m ** (-kappa)
#             log_gamma_bar = m_kappa * log_gamma + (1 - m_kappa) * log_gamma_bar

#         gamma = float(jnp.exp(log_gamma_bar))

#         # ===== 3. Steepness: use fixed value, not DA tuning =====
#         # DA tuning for steepness disabled because acceptance rate is not the right
#         # objective - it was driving steepness to extreme values (step-function behavior).
#         # Steepness remains at its initialized value (5.0 for tanh, 10.0 for sigmoid).
#         #
#         # if has_steepness:
#         #     log_steepness = jnp.log(steepness)
#         #     mu_steepness = log_steepness
#         #     log_steepness_bar = log_steepness
#         #     H_bar_steepness = 0.0
#         #
#         #     for m in range(1, max_iter_per_param + 1):
#         #         key, subkey = random.split(key)
#         #         _, _, accept_rate, final_state = rahmc_run(
#         #             subkey, log_prob_fn, current_position,
#         #             step_size=float(step_size), num_steps=num_steps,
#         #             gamma=float(gamma), steepness=float(jnp.exp(log_steepness)),
#         #             num_samples=n_samples_per_tune, burn_in=0,
#         #             friction_schedule=friction_schedule,
#         #             inv_mass_matrix=inv_mass_matrix
#         #         )
#         #         current_position = final_state.position
#         #         alpha = float(jnp.mean(accept_rate))
#         #
#         #         eta_m = 1.0 / (m + t0)
#         #         H_bar_steepness = (1 - eta_m) * H_bar_steepness + eta_m * (target_accept - alpha)
#         #         log_steepness = mu_steepness - (jnp.sqrt(m) / gamma_da) * H_bar_steepness
#         #         log_steepness = jnp.clip(log_steepness, jnp.log(0.5), jnp.log(20.0))
#         #         m_kappa = m ** (-kappa)
#         #         log_steepness_bar = m_kappa * log_steepness + (1 - m_kappa) * log_steepness_bar
#         #
#         #     steepness = float(jnp.exp(log_steepness_bar))

#         # Get final acceptance rate for this cycle
#         key, subkey = random.split(key)
#         _, _, accept_rate, final_state = rahmc_run(
#             subkey, log_prob_fn, current_position,
#             step_size=float(step_size), num_steps=num_steps,
#             gamma=float(gamma), steepness=float(steepness),
#             num_samples=n_samples_per_tune, burn_in=0,
#             friction_schedule=friction_schedule,
#             inv_mass_matrix=inv_mass_matrix
#         )
#         current_position = final_state.position  # Evolve chain
#         final_accept = float(jnp.mean(accept_rate))

#         cycle_history.append({
#             'step_size': step_size,
#             'gamma': gamma,
#             'steepness': steepness,
#             'accept_rate': final_accept,
#         })

#         print(f"      step_size={step_size:.4f}, gamma={gamma:.4f}, steepness={steepness:.4f}, accept={final_accept:.3f}")

#         # Check convergence after minimum cycles
#         if cycle >= min_cycles:
#             rel_change_step = abs(step_size - prev_step_size) / (abs(prev_step_size) + 1e-10)
#             rel_change_gamma = abs(gamma - prev_gamma) / (abs(prev_gamma) + 1e-10)
#             rel_change_steep = abs(steepness - prev_steepness) / (abs(prev_steepness) + 1e-10) if has_steepness else 0.0

#             max_change = max(rel_change_step, rel_change_gamma, rel_change_steep)

#             if max_change < tolerance:
#                 converged_count += 1
#             else:
#                 converged_count = 0

#             if converged_count >= patience:
#                 print(f"      Converged after {cycle + 1} cycles")
#                 converged_cycle = cycle + 1
#                 break

#         prev_step_size = step_size
#         prev_gamma = gamma
#         prev_steepness = steepness

#     history = {
#         'cycle_history': cycle_history,
#         'converged_cycle': converged_cycle,
#         'schedule_type': schedule_type,
#         'has_steepness': has_steepness,
#         'num_steps': num_steps,
#         'target_accept': target_accept,
#     }

#     return step_size, gamma, steepness, history


# =========================================================================
# Joint Tuning Logic for GRAHMC
# =========================================================================

class JointDualAveragingState(NamedTuple):
    """State for vector-based dual averaging (e.g., [log_step, log_gamma])."""
    log_params: jnp.ndarray      # Vector: [log_step, log_gamma]
    log_params_bar: jnp.ndarray  # Vector: Smoothed estimates
    H_bar: float                 # Scalar: Acceptance error is still scalar
    mu: jnp.ndarray              # Vector: Reference point
    count: int
    gamma: float = 0.05
    t0: float = 10.0
    kappa: float = 0.75

def joint_da_init(initial_params: jnp.ndarray) -> JointDualAveragingState:
    """Initialize joint dual averaging state."""
    log_params = jnp.log(initial_params)
    return JointDualAveragingState(
        log_params=log_params,
        log_params_bar=log_params, # Initialize to current
        H_bar=0.0,
        mu=log_params,
        count=0
    )

def joint_da_update(state: JointDualAveragingState, accept_stat: float, target_accept: float) -> JointDualAveragingState:
    """Update vector of parameters based on scalar acceptance error."""
    m = state.count + 1
    
    # 1. Update Error Accumulator (H_bar)
    # Note: A low acceptance (accept < target) results in POSITIVE (target - accept).
    eta_m = 1.0 / (m + state.t0)
    H_bar = (1 - eta_m) * state.H_bar + eta_m * (target_accept - accept_stat)
    
    # 2. Update Parameters
    # We subtract H_bar.
    # If acceptance is LOW -> H_bar is POSITIVE -> Params DECREASE.
    #   - Step size decreases (improves integration accuracy)
    #   - Gamma decreases (reduces energy drift)
    # This direction is correct for both parameters.
    log_params = state.mu - (jnp.sqrt(m) / state.gamma) * H_bar

    # Clip gamma to prevent extreme values (step_size left unconstrained)
    log_params = jnp.array([
        log_params[0],  # step_size: no explicit bounds
        jnp.clip(log_params[1], jnp.log(0.01), jnp.log(20.0))
    ])

    # 3. Smoothing
    m_kappa = m ** (-state.kappa)
    log_params_bar = m_kappa * log_params + (1 - m_kappa) * state.log_params_bar
    
    return JointDualAveragingState(
        log_params=log_params,
        log_params_bar=log_params_bar,
        H_bar=float(H_bar),
        mu=state.mu,
        count=m,
        gamma=state.gamma,
        t0=state.t0,
        kappa=state.kappa
    )

def joint_tune_grahmc(
    key: jnp.ndarray,
    log_prob_fn,
    grad_log_prob_fn, # Unused, kept for API compatibility
    init_position: jnp.ndarray,
    num_steps: int,
    schedule_type: str = 'constant',
    target_accept: float = 0.65,
    max_iter: int = 1000,
    inv_mass_matrix: jnp.ndarray = None,
    current_step_size: float = None, # Optional starting point
    fixed_steepness: float = 10.0,   # Fixed steepness value
) -> Tuple[float, float, float, Dict]:
    """Jointly tune step_size and gamma for GRAHMC.
    
    Replaces coordinate-wise tuning. Steepness is fixed (not tuned).
    """
    friction_schedule = get_friction_schedule(schedule_type)
    n_dim = init_position.shape[-1]
    
    # 1. Initialization
    # Use current_step_size if provided (from Phase 2), else heuristic
    if current_step_size is None:
        init_step = 0.5 / jnp.sqrt(n_dim)
    else:
        init_step = current_step_size
        
    init_gamma = 1.0  # Per RAHMC paper Section 3.3: gamma_0 = 1.0
    
    # Pack into vector: [step_size, gamma]
    init_params = jnp.array([init_step, init_gamma])
    state = joint_da_init(init_params)
    
    # Track history
    history = {
        "step_size": [],
        "gamma": [],
        "accept_rate": []
    }
    
    current_position = init_position
    n_samples_per_tune = 50 # Speed vs noise trade-off
    
    print(f"    Joint Tuning (Step+Gamma) for {max_iter} iterations...")
    
    for m in range(1, max_iter + 1):
        key, subkey = random.split(key)
        
        # Unpack current smoothed parameters
        # We use the smoothed bar values for the actual sampling to be stable
        curr_params = jnp.exp(state.log_params) # Use noisy for exploration? 
        # Standard DA uses noisy 'log_params' for exploration, 'log_params_bar' for final result.
        # Let's use noisy for the run to ensure we explore the space.
        
        curr_step = float(curr_params[0])
        curr_gamma = float(curr_params[1])
        
        # Clip gamma to sane bounds (prevent 0 or explosion) during tuning
        curr_gamma = max(0.001, min(curr_gamma, 50.0))
        
        # Run Sampler
        _, _, accept_rate, final_state = rahmc_run(
            subkey, log_prob_fn, current_position,
            step_size=curr_step,
            num_steps=num_steps,
            gamma=curr_gamma,
            steepness=fixed_steepness, # FIXED
            num_samples=n_samples_per_tune, 
            burn_in=0,
            friction_schedule=friction_schedule,
            inv_mass_matrix=inv_mass_matrix
        )
        
        current_position = final_state.position
        alpha = float(jnp.mean(accept_rate))
        
        # Update State
        state = joint_da_update(state, alpha, target_accept)
        
        # Logging
        smooth_params = jnp.exp(state.log_params_bar)
        history["step_size"].append(float(smooth_params[0]))
        history["gamma"].append(float(smooth_params[1]))
        history["accept_rate"].append(alpha)
        
        if m % 100 == 0:
            print(f"      Iter {m}: step={smooth_params[0]:.4f}, gamma={smooth_params[1]:.4f}, accept={alpha:.3f}")

    # Final Extraction
    final_params = jnp.exp(state.log_params_bar)
    final_step = float(final_params[0])
    final_gamma = float(final_params[1])
    
    print(f"    Converged: step={final_step:.4f}, gamma={final_gamma:.4f}")
    
    return final_step, final_gamma, fixed_steepness, history

class DualAveragingState(NamedTuple):
    """State for step-wise dual averaging."""
    log_step: float       # Current log(step_size)
    log_step_bar: float   # Smoothed log(step_size)
    H_bar: float          # Running average of (target - accept)
    mu: float             # Reference point
    count: int            # Iteration counter (m)
    gamma: float = 0.05
    t0: float = 10.0
    kappa: float = 0.75


def da_init(initial_step_size: float) -> DualAveragingState:
    """Initialize dual averaging state."""
    log_step = jnp.log(initial_step_size)
    return DualAveragingState(
        log_step=float(log_step),
        log_step_bar=0.0,  # Will be initialized on first update
        H_bar=0.0,
        mu=float(log_step),
        count=0
    )


def da_update(state: DualAveragingState, accept_stat: float, target_accept: float) -> DualAveragingState:
    """Perform one step of dual averaging parameter update.
    
    Returns updated state. To get the current step size, use exp(state.log_step).
    """
    m = state.count + 1
    
    # Update H_bar
    eta_m = 1.0 / (m + state.t0)
    H_bar = (1 - eta_m) * state.H_bar + eta_m * (target_accept - accept_stat)
    
    # Compute log_step
    log_step = state.mu - (jnp.sqrt(m) / state.gamma) * H_bar
    
    # Compute log_step_bar (smoothed)
    m_kappa = m ** (-state.kappa)
    
    # Special handling for first iteration to initialize log_step_bar correctly
    if m == 1:
        log_step_bar = log_step
    else:
        log_step_bar = m_kappa * log_step + (1 - m_kappa) * state.log_step_bar
        
    return DualAveragingState(
        log_step=float(log_step),
        log_step_bar=float(log_step_bar),
        H_bar=float(H_bar),
        mu=state.mu,
        count=m,
        gamma=state.gamma,
        t0=state.t0,
        kappa=state.kappa
    )

def da_reset(state: DualAveragingState) -> DualAveragingState:
    """Reset the counter and stats, but keep the current step size as the new target (mu).

    Used at the start of a new adaptation window.

    IMPORTANT: Uses the smoothed estimate (log_step_bar) if available, not the noisy
    estimate (log_step). This prevents drift when resetting after mass matrix updates.
    """
    # Use smoothed estimate if we have enough samples, otherwise fall back to noisy
    if state.count > 0:
        current_step = state.log_step_bar
    else:
        current_step = state.log_step

    return DualAveragingState(
        log_step=current_step,
        log_step_bar=current_step,  # Initialize to current best, not 0
        H_bar=0.0,
        mu=current_step,  # Reset target to smoothed best guess
        count=0,
        gamma=state.gamma,
        t0=state.t0,
        kappa=state.kappa
    )