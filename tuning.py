
"""
Automated Parameter Tuning for MCMC Samplers

This module implements gradient-based optimization of MCMC sampler parameters
using Expected Squared Jump Distance (ESJD) as the objective function.

Supported Samplers:
- RWMH: Random Walk Metropolis-Hastings (tunes proposal_scale)
- HMC: Hamiltonian Monte Carlo (tunes step_size, total_time)
- GRAHMC: Generalized RAHMC with schedules (tunes step_size, total_time, gamma, steepness)
- NUTS: No-U-Turn Sampler (tunes step_size, trajectory length is automatic)

Key Features:
- Gradient-based optimization (more efficient than dual averaging)
- Convergence detection (min_iter, tolerance, patience)
- Dynamic masking for continuous trajectory length optimization (fully differentiable)
- Maximizes proposal-level ESJD weighted by acceptance probability
- Variance reduction via multiple independent runs

Usage:
    # Command line
    python tuning.py --sampler rwmh --dim 10
    python tuning.py --sampler grahmc --schedule tanh --dim 10

    # As module
    from tuning import tune_sampler
    params = tune_sampler('rwmh', key, log_prob_fn, init_position)
"""

from __future__ import annotations
import argparse
import sys
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax
import optax
from typing import NamedTuple, Callable, Dict, List, Tuple, Optional, Any
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

import os
os.environ['JAX_ENABLE_X64'] = 'True' # Enable float64 for better numerical stability

# Import all samplers
from samplers.RWMH import rwMH_run
from samplers.HMC import hmc_run, hmc_run_dynamic
from samplers.GRAHMC import (
    Array,
    LogProbFn,
    FrictionScheduleFn,
    rahmc_init,
    rahmc_run,
    constant_schedule,
    tanh_schedule,
    sigmoid_schedule,
    linear_schedule,
    sine_schedule,
    standard_normal_log_prob,
    RAHMCState,
    _trajectory_with_schedule,
)
from samplers.NUTS import nuts_run

# %%
def compute_autocorr(x: jnp.ndarray, max_lag: int | None = None) -> jnp.ndarray:
    """Compute 1D autocorrelation using FFT."""
    n = x.shape[0]
    if max_lag is None:
        max_lag = n - 1
    x_centered = x - jnp.mean(x)
    fft_size = 2 ** int(jnp.ceil(jnp.log2(jnp.maximum(1, 2 * n - 1))))
    fft_x = jnp.fft.fft(x_centered, n=fft_size)
    autocorr = jnp.fft.ifft(fft_x * jnp.conj(fft_x)).real[:max_lag]
    return autocorr / autocorr[0]

def estimate_ess_geyer(samples: jnp.ndarray) -> float:
    """Geyer initial positive sequence ESS estimator (JAX compatible)."""
    n = samples.shape[0]
    max_lag = int(jnp.minimum(n // 2, 1000))
    rho = compute_autocorr(samples, max_lag)
    rho_even = rho[1::2]
    rho_odd = rho[2::2]
    n_pairs = jnp.minimum(rho_even.shape[0], rho_odd.shape[0])
    rho_pairs = rho_even[:n_pairs] + rho_odd[:n_pairs]
    positive = rho_pairs > 0
    cutoff = jnp.argmax(~positive)
    cutoff = jnp.where(jnp.any(~positive), cutoff, rho_pairs.shape[0])
    indices = jnp.arange(rho_pairs.shape[0])
    masked_pairs = jnp.where(indices < cutoff, rho_pairs, 0.0)
    rho_sum = rho[0] + 2.0 * jnp.sum(masked_pairs)
    return n / jnp.maximum(rho_sum, 1.0)

def estimate_ess_bulk(samples: jnp.ndarray) -> jnp.ndarray:
    """Bulk ESS across chains and dimensions."""
    n_chains, n_samples, n_dim = samples.shape
    reshaped = samples.reshape(n_chains * n_samples, n_dim)
    dim_indices = jnp.arange(n_dim)
    return vmap(lambda i: estimate_ess_geyer(reshaped[:, i]))(dim_indices)

# ============================================================================
# Parameter Structures for All Samplers
# ============================================================================

# RWMH Parameters
class RWMHParams(NamedTuple):
    """RWMH tuning parameters in log space."""
    log_proposal_scale: float  # log(σ) - proposal standard deviation

# HMC Parameters
class HMCParams(NamedTuple):
    """HMC tuning parameters in log space."""
    log_step_size: float   # log(ε) - leapfrog step size
    log_total_time: float  # log(T) - total trajectory time

# GRAHMC Parameters (all friction schedules)
class GRAHMCParams(NamedTuple):
    """GRAHMC tuning parameters in log space."""
    log_step_size: float   # log(ε) - leapfrog step size
    log_total_time: float  # log(T) - total trajectory time
    log_gamma: float       # log(γ) - friction magnitude
    log_steepness: float   # log(s) - schedule transition steepness

# NUTS Parameters
class NUTSParams(NamedTuple):
    """NUTS tuning parameters in log space."""
    log_step_size: float   # log(ε) - leapfrog step size
    # Note: trajectory length is automatic in NUTS

# Legacy aliases for backward compatibility
TuningParams = GRAHMCParams
DynamicParams = GRAHMCParams


def params_to_dict(params: Any) -> Dict[str, float]:
    """Convert any parameter NamedTuple to human-readable dictionary."""
    result = {}

    if isinstance(params, RWMHParams):
        result['proposal_scale'] = float(jnp.exp(params.log_proposal_scale))

    elif isinstance(params, (HMCParams, GRAHMCParams)):
        result['step_size'] = float(jnp.exp(params.log_step_size))
        result['total_time'] = float(jnp.exp(params.log_total_time))
        result['expected_num_steps'] = result['total_time'] / result['step_size']

        if isinstance(params, GRAHMCParams):
            result['gamma'] = float(jnp.exp(params.log_gamma))
            result['steepness'] = float(jnp.exp(params.log_steepness))

    elif isinstance(params, NUTSParams):
        result['step_size'] = float(jnp.exp(params.log_step_size))

    return result

FRICTION_SCHEDULES = {
    'constant': constant_schedule,
    'tanh': tanh_schedule,
    'sigmoid': sigmoid_schedule,
    'linear': linear_schedule,
    'sine': sine_schedule,
}

def get_friction_schedule(schedule_type: str):
    """Get friction schedule function by name."""
    return FRICTION_SCHEDULES[schedule_type]


# ============================================================================
# RAHMC Step and Run Functions
# ============================================================================
@partial(jit, static_argnames=("log_prob_fn", "return_proposal", "friction_schedule", "num_steps"))
def rahmc_step(
    state: RAHMCState,
    step_size: float,
    num_steps: int,
    gamma_max: float,
    steepness: float,
    key: Array,
    log_prob_fn: LogProbFn,
    friction_schedule: FrictionScheduleFn = None,
    return_proposal: bool = False, # return proposal positions and log probs (key, new_state, proposal_position, proposal_log_prob)
) -> Tuple[Array, RAHMCState] | Tuple[Array, RAHMCState, Array, Array]:
    if friction_schedule is None:
        friction_schedule = constant_friction(gamma_max)

    n_chains, n_dim = state.position.shape
    pos_dtype = state.position.dtype
    logprob_dtype = state.log_prob.dtype

    key, step_key = random.split(key)
    k_mom, k_acc = random.split(step_key, 2)

    p0 = random.normal(k_mom, shape=(n_chains, n_dim), dtype=pos_dtype)

    kin0 = 0.5 * jnp.sum(p0**2, axis=-1)
    H0 = -state.log_prob + kin0.astype(logprob_dtype)

    total_time = step_size * num_steps

    q, p, lp, glp = _trajectory_with_schedule(
        state.position, p0, step_size, gamma_max, steepness,
        state.log_prob, state.grad_log_prob,
        num_steps,
        log_prob_fn=log_prob_fn, friction_schedule=friction_schedule,
    )

    # flip momentum
    p = -p

    # compute final energies
    kin1 = 0.5 * jnp.sum(p**2, axis=-1)
    H1 = -lp + kin1.astype(logprob_dtype)

    # add overflow protection
    H1 = jnp.where(jnp.isfinite(H1), H1, jnp.array(1e10, dtype=logprob_dtype))

    # MH test
    log_alpha = H0 - H1
    delta_H = H1 - H0

    u = random.uniform(k_acc, shape=(n_chains,), dtype=logprob_dtype)
    accept = jnp.log(u) < jnp.minimum(0.0, log_alpha)

    new_pos = jnp.where(accept[:, None], q, state.position)
    new_lp = jnp.where(accept, lp, state.log_prob)
    new_glp = jnp.where(accept[:, None], glp, state.grad_log_prob)
    new_acc = state.accept_count + accept.astype(jnp.int32)

    new_state = RAHMCState(new_pos, new_lp, new_glp, new_acc)
    
    if return_proposal:
        # Return proposal info for ESJD computation
        return key, new_state, q, lp, delta_H
    else:
        return key, new_state
    
    
@partial(jit, static_argnames=("log_prob_fn", "num_samples", "burn_in", "track_proposals", "friction_schedule", "num_steps"))
def rahmc_run(
    key: Array,
    log_prob_fn: LogProbFn,
    init_position: Array,
    step_size: float,
    num_steps: int,
    gamma: float,
    steepness: float,
    num_samples: int,
    burn_in: int = 0,
    friction_schedule: FrictionScheduleFn = None,
    track_proposals: bool = False,
) -> Tuple[Array, Array, Array, RAHMCState] | Tuple[Array, Array, Array, RAHMCState, Array, Array]:
    
    if friction_schedule is None:
        friction_schedule = constant_friction(gamma)
    state = rahmc_init(init_position, log_prob_fn)
    n_chains, n_dim = state.position.shape

    pos_type = state.position.dtype
    lp_type = state.log_prob.dtype

    eps = jnp.asarray(step_size, dtype=pos_type)
    gam = jnp.asarray(gamma, dtype=pos_type)
    steep = jnp.asarray(steepness, dtype=pos_type)

    # burn-in
    if burn_in > 0:
        def burn_body(carry, _):
            k, s = carry
            k, s = rahmc_step(s, eps, num_steps, gam, steep, k, log_prob_fn, friction_schedule, return_proposal=False)
            return (k, s), None
        (key, state), _ = lax.scan(burn_body, (key, state), length=burn_in)
        state = state._replace(accept_count=jnp.zeros(n_chains, dtype=jnp.int32))

    # sampling
    if track_proposals:
        def body_with_proposals(carry, _):
            k, s = carry
            pre_pos, pre_lp = s.position, s.log_prob
            k, s, prop_pos, prop_lp, delta_H = rahmc_step(
                s, eps, num_steps, gam, steep, k, log_prob_fn, friction_schedule, return_proposal=True
            )
            return (k, s), (pre_pos, pre_lp, prop_pos, prop_lp, delta_H, s.position, s.log_prob)
        
        # (key, state), (samples, lps, prop_positions, prop_lps, delta_H) = lax.scan(
        #     body_with_proposals, (key, state), length=num_samples
        # )
        (key, state), (pre_positions, pre_lps, prop_positions, prop_lps, deltas_H, post_positions, post_lps) = lax.scan(
            body_with_proposals, (key, state), length=num_samples
        )
        
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples

        return (
            post_positions, post_lps,
            accept_rate, state,
            pre_positions, pre_lps,
            prop_positions, prop_lps,
            deltas_H
        )
    else:
        def body(carry, _):
            k, s = carry
            k, s = rahmc_step(s, eps, num_steps, gam, steep, k, log_prob_fn, friction_schedule, return_proposal=False)
            return (k, s), (s.position, s.log_prob)
        
        (key, state), (samples, lps) = lax.scan(body, (key, state), length=num_samples)
        
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return samples, lps, accept_rate, state

# ============================================================================
# ESJD Computation
# ============================================================================

def compute_proposal_esjd_soft(
    initial_positions: jnp.ndarray,
    proposals: jnp.ndarray,
    delta_H: jnp.ndarray, # (n_samples, n_chains)
    # log_prob_initial: jnp.ndarray,
    # log_prob_proposal: jnp.ndarray,
) -> float:
    """Compute proposal-level ESJD with soft (probability-weighted) acceptance."""
    # Compute squared jumping distances
    squared_jumps = jnp.sum((proposals - initial_positions)**2, axis=-1)  # (n_samples, n_chains)
    
    # Compute acceptance probabilities (smooth and differentiable!)
    alpha = jnp.minimum(1.0, jnp.exp(-delta_H))  # (n_samples, n_chains)

    # smoother alternative:
    # alpha = jax.nn.sigmoid(-delta_H / 1.0)
    
    # Weight jumps by acceptance probability
    weighted_jumps = squared_jumps * alpha
    
    # Average over all proposals
    esjd = jnp.mean(weighted_jumps)
    
    return esjd



# ============================================================================
# Optimization Objectives
# ============================================================================

def objective_proposal_esjd(
    dyn_params: DynamicParams,
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    num_samples: int,
    burn_in: int,
    schedule_type: str,
) -> float:
    """
    Maximize proposal-level acceptance-weighted ESJD.
    Returns the negative value for minimization.

    Uses stochastic rounding for num_steps:
    - T / epsilon = n + r, where n = floor(T/epsilon) and 0 <= r < 1
    - num_steps = n with probability (1-r), n+1 with probability r
    - Unbiased in expectation: E[num_steps] = n + r = T / epsilon
    """
    # Extract parameters
    step_size = jnp.exp(dyn_params.log_step_size)
    total_time = jnp.exp(dyn_params.log_total_time)
    gamma = jnp.exp(dyn_params.log_gamma)
    steepness = jnp.exp(dyn_params.log_steepness)

    # Stochastic rounding: num_steps ~ floor(T/eps) + Bernoulli(remainder)
    key, round_key = random.split(key)
    num_steps_float = total_time / step_size
    n = jnp.floor(num_steps_float)
    r = num_steps_float - n  # remainder in [0, 1)

    # Round up to n+1 with probability r, else stay at n
    round_up = random.uniform(round_key) < r
    num_steps_jax = n + round_up.astype(jnp.int32)
    # Convert to Python int for use as static argument in JIT
    num_steps = int(jnp.maximum(num_steps_jax, 1))

    friction_schedule = get_friction_schedule(schedule_type)

    (samples, lps, accept_rate, final_state,
     pre_positions, pre_lps, prop_positions, prop_lps, deltas_H) = rahmc_run(
        key=key,
        log_prob_fn=log_prob_fn,
        init_position=init_position,
        step_size=step_size,
        num_steps=num_steps,
        gamma=gamma,
        steepness=steepness,
        num_samples=num_samples,
        burn_in=burn_in,
        friction_schedule=friction_schedule,
        track_proposals=True,
    )

    # ESJD over proposals
    squared_jumps = jnp.sum((prop_positions - pre_positions) ** 2, axis=-1) # (S, C)
    alpha = jnp.minimum(1.0, jnp.exp(-deltas_H)) # exact MH weight
    esjd = jnp.mean(squared_jumps * alpha)

    log_esjd = jnp.log(esjd + 1e-10)

    # Acceptance rate penalties (target 0.65 for HMC-like samplers)
    mean_alpha = jnp.mean(alpha)
    target_accept = 0.65
    accept_penalty = (mean_alpha - target_accept) ** 2
    low_accept_guard = jnp.maximum(0.0, 0.15 - mean_alpha) ** 2
    high_accept_guard = jnp.maximum(0.0, mean_alpha - 0.90) ** 2

    objective_value = log_esjd \
        - 50.0 * accept_penalty \
        - 25.0 * (low_accept_guard + high_accept_guard)

    # Parameter bound penalties
    penalty = 0.0
    # Step size: 0.01 <= epsilon <= 1.0
    penalty += 1.0 * jnp.maximum(0.0, dyn_params.log_step_size - jnp.log(1.0)) ** 2
    penalty += 1.0 * jnp.maximum(0.0, -dyn_params.log_step_size + jnp.log(0.01)) ** 2

    # Total time: 1.0 <= T <= 100.0
    penalty += 1.0 * jnp.maximum(0.0, dyn_params.log_total_time - jnp.log(100.0)) ** 2
    penalty += 5.0 * jnp.maximum(0.0, jnp.log(1.0) - dyn_params.log_total_time) ** 2

    # Gamma: 0.01 <= gamma <= 1.5
    penalty += 100.0 * jnp.maximum(0.0, dyn_params.log_gamma - jnp.log(1.5)) ** 2
    penalty += 1.0 * jnp.maximum(0.0, jnp.log(0.01) - dyn_params.log_gamma) ** 2

    # Steepness: 0.5 <= steepness <= 50.0
    penalty += 0.5 * jnp.maximum(0.0, dyn_params.log_steepness - jnp.log(50.0)) ** 2
    penalty += 0.5 * jnp.maximum(0.0, jnp.log(0.5) - dyn_params.log_steepness) ** 2

    return -(objective_value - penalty)


def objective_proposal_esjd_variance_reduced(
    dyn_params: DynamicParams,
    keys: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    num_samples: int,
    burn_in: int,
    schedule_type: str,
) -> float:
    """
    Proposal-level ESJD objective with multiple runs for variance reduction.

    Averages the objective over multiple independent runs to reduce gradient
    variance from stochastic rounding and MCMC randomness.

    Note: We use a Python loop instead of vmap because stochastic rounding
    requires concrete integer values that can't be traced.
    """
    neg_esjd_values = []
    for key in keys:
        neg_esjd = objective_proposal_esjd(
            dyn_params,
            key, log_prob_fn, init_position,
            num_samples, burn_in, schedule_type
        )
        neg_esjd_values.append(neg_esjd)

    return jnp.mean(jnp.array(neg_esjd_values))


# ============================================================================
# Objective Functions for Other Samplers
# ============================================================================

def objective_rwmh_esjd(
    params: RWMHParams,
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    num_samples: int,
    burn_in: int,
) -> float:
    """RWMH objective: maximize ESJD by tuning proposal_scale."""
    proposal_scale = jnp.exp(params.log_proposal_scale)

    # Run RWMH (note: rwMH_run doesn't track proposals, so we compute ESJD from samples)
    samples, lps, accept_rate, _ = rwMH_run(
        key, log_prob_fn, init_position,
        num_samples=num_samples + burn_in,
        scale=proposal_scale,
        burn_in=0
    )

    # Discard burn-in
    samples = samples[burn_in:]

    # Compute ESJD from consecutive samples
    diffs = samples[1:] - samples[:-1]  # (S-1, C, D)
    squared_jumps = jnp.sum(diffs ** 2, axis=-1)  # (S-1, C)
    esjd = jnp.mean(squared_jumps)

    log_esjd = jnp.log(esjd + 1e-10)

    # Acceptance rate penalty (target 0.234 for RWMH)
    mean_accept = jnp.mean(accept_rate)  # Keep in JAX
    target_accept = 0.234
    accept_penalty = (mean_accept - target_accept) ** 2

    objective_value = log_esjd - 50.0 * accept_penalty

    # Parameter bounds: 0.01 <= scale <= 10.0
    penalty = 0.0
    penalty += 1.0 * jnp.maximum(0.0, params.log_proposal_scale - jnp.log(10.0)) ** 2
    penalty += 1.0 * jnp.maximum(0.0, -params.log_proposal_scale + jnp.log(0.01)) ** 2

    return -(objective_value - penalty)


def objective_hmc_esjd(
    params: HMCParams,
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    num_samples: int,
    burn_in: int,
    max_steps: int = 200,
) -> float:
    """HMC objective: maximize ESJD by tuning step_size and total_time.

    Uses dynamic trajectory with straight-through estimator:
    - Forward: Hard threshold (i < actual_steps) maintains MCMC correctness
    - Backward: Soft sigmoid allows gradients to flow through actual_steps

    Note: Must use dynamic trajectory (not int(num_steps)) because converting
    to Python int breaks gradient flow, even with STE.
    """
    step_size = jnp.exp(params.log_step_size)
    total_time = jnp.exp(params.log_total_time)

    # Compute actual steps as continuous variable (gradients flow!)
    actual_steps = total_time / step_size

    # Run HMC with dynamic trajectory length (STE masking maintains detailed balance)
    samples, lps, accept_rate, _ = hmc_run_dynamic(
        key, log_prob_fn, init_position,
        step_size=step_size,
        actual_steps=actual_steps,  # Continuous, differentiable
        max_steps=max_steps,  # Static maximum (for JIT)
        num_samples=num_samples + burn_in,
        burn_in=0,
        track_proposals=False,
    )

    samples = samples[burn_in:]

    # Compute ESJD
    diffs = samples[1:] - samples[:-1]
    squared_jumps = jnp.sum(diffs ** 2, axis=-1)
    esjd = jnp.mean(squared_jumps)

    log_esjd = jnp.log(esjd + 1e-10)

    # Acceptance rate penalty (target 0.65 for HMC)
    mean_accept = jnp.mean(accept_rate)  # Keep in JAX
    target_accept = 0.65
    accept_penalty = (mean_accept - target_accept) ** 2
    low_accept_guard = jnp.maximum(0.0, 0.15 - mean_accept) ** 2
    high_accept_guard = jnp.maximum(0.0, mean_accept - 0.90) ** 2

    objective_value = log_esjd - 50.0 * accept_penalty - 25.0 * (low_accept_guard + high_accept_guard)

    # Parameter bounds
    penalty = 0.0
    penalty += 1.0 * jnp.maximum(0.0, params.log_step_size - jnp.log(1.0)) ** 2
    penalty += 1.0 * jnp.maximum(0.0, -params.log_step_size + jnp.log(0.01)) ** 2
    penalty += 1.0 * jnp.maximum(0.0, params.log_total_time - jnp.log(100.0)) ** 2
    penalty += 5.0 * jnp.maximum(0.0, jnp.log(1.0) - params.log_total_time) ** 2

    return -(objective_value - penalty)


def objective_nuts_esjd(
    params: NUTSParams,
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    num_samples: int,
    burn_in: int,
) -> float:
    """
    NUTS objective: maximize ESJD by tuning step_size (trajectory length is automatic).

    Note: Uses stop_gradient because NUTS uses lax.while_loop internally, which isn't
    differentiable. Gradients come from variation in ESJD across parameters (zeroth-order).
    """
    step_size = jnp.exp(params.log_step_size)

    # Run NUTS - keep step_size in JAX
    samples, lps, accept_rate, _, _, _ = nuts_run(
        key, log_prob_fn, init_position,
        step_size=step_size,  # Keep as JAX array
        num_samples=num_samples + burn_in,
        burn_in=0
    )

    # Stop gradient through sampler (NUTS while_loop isn't differentiable)
    samples = jax.lax.stop_gradient(samples)
    accept_rate = jax.lax.stop_gradient(accept_rate)

    samples = samples[burn_in:]

    # Compute ESJD
    diffs = samples[1:] - samples[:-1]
    squared_jumps = jnp.sum(diffs ** 2, axis=-1)
    esjd = jnp.mean(squared_jumps)

    log_esjd = jnp.log(esjd + 1e-10)

    # Acceptance rate penalty (target 0.65 for NUTS)
    mean_accept = jnp.mean(accept_rate)  # Keep in JAX
    target_accept = 0.65
    accept_penalty = (mean_accept - target_accept) ** 2
    low_accept_guard = jnp.maximum(0.0, 0.15 - mean_accept) ** 2
    high_accept_guard = jnp.maximum(0.0, mean_accept - 0.90) ** 2

    objective_value = log_esjd - 50.0 * accept_penalty - 25.0 * (low_accept_guard + high_accept_guard)

    # Parameter bounds: 0.01 <= step_size <= 1.0
    penalty = 0.0
    penalty += 1.0 * jnp.maximum(0.0, params.log_step_size - jnp.log(1.0)) ** 2
    penalty += 1.0 * jnp.maximum(0.0, -params.log_step_size + jnp.log(0.01)) ** 2

    return -(objective_value - penalty)


# Variance-reduced wrappers (for consistency, though simpler samplers may not need variance reduction)
def objective_rwmh_variance_reduced(params: RWMHParams, keys: jnp.ndarray, log_prob_fn, init_position, num_samples, burn_in):
    return jnp.mean(jnp.array([objective_rwmh_esjd(params, k, log_prob_fn, init_position, num_samples, burn_in) for k in keys]))

def objective_hmc_variance_reduced(params: HMCParams, keys: jnp.ndarray, log_prob_fn, init_position, num_samples, burn_in, max_steps: int = 200):
    # Use vmap for proper gradient flow instead of list comprehension
    obj_fn = lambda key: objective_hmc_esjd(params, key, log_prob_fn, init_position, num_samples, burn_in, max_steps)
    values = vmap(obj_fn)(keys)
    return jnp.mean(values)

def objective_nuts_variance_reduced(params: NUTSParams, keys: jnp.ndarray, log_prob_fn, init_position, num_samples, burn_in):
    return jnp.mean(jnp.array([objective_nuts_esjd(params, k, log_prob_fn, init_position, num_samples, burn_in) for k in keys]))


# ============================================================================
# Parameter Optimization
# ============================================================================
def optimize_parameters(
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    schedule_type: str,
    initial_params: DynamicParams,
    num_samples: int = 1000,
    burn_in: int = 500,
    max_iter: int = 500,
    min_iter: int = 50,
    tolerance: float = 0.01,
    patience: int = 10,
    n_runs_per_eval: int = 3,
    learning_rate: float = 0.02,
    verbose: bool = False,
) -> Tuple[DynamicParams, List[Dict], float]:
    """
    Optimize all RAHMC parameters (step_size, total_time, gamma, steepness)
    for a given friction schedule using gradient descent with convergence detection.

    Parameters are optimized in log space for positivity constraints.
    Uses variance reduction via multiple independent runs per evaluation.

    Convergence Detection:
        - Waits for min_iter iterations before checking convergence
        - Checks relative change in best metric < tolerance
        - Requires patience consecutive converged iterations
        - Stops early if converged, or continues until max_iter

    Returns:
        best_params: Optimized parameters
        history: List of dicts with iteration stats
        best_neg_metric: Best negative ESJD achieved
    """
    dyn_params = initial_params

    objective_fn = objective_proposal_esjd_variance_reduced
    metric_name = "log(ESJD)"

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(dyn_params)

    grad_fn = jax.grad(objective_fn, argnums=0)

    # Track history and convergence
    history = []
    best_neg_metric = jnp.inf
    best_dyn_params = dyn_params
    prev_best = jnp.inf
    converged_count = 0

    if verbose:
        print("\n" + "="*80)
        print(f"OPTIMIZING: {schedule_type.upper()}")
        print("="*80)

    for iteration in range(1, max_iter + 1):
        key, *eval_keys = random.split(key, n_runs_per_eval + 1)
        eval_keys = jnp.array(eval_keys)

        # Evaluate objective
        neg_metric = objective_fn(
            dyn_params,
            eval_keys, log_prob_fn, init_position,
            num_samples, burn_in, schedule_type
        )

        # Compute gradient
        grads = grad_fn(
            dyn_params,
            eval_keys, log_prob_fn, init_position,
            num_samples, burn_in, schedule_type
        )

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        dyn_params = optax.apply_updates(dyn_params, updates)

        # Track best
        if neg_metric < best_neg_metric:
            best_neg_metric = neg_metric
            best_dyn_params = dyn_params

        # Convert to full params for logging
        full_params = TuningParams(
            dyn_params.log_step_size,
            dyn_params.log_total_time,
            dyn_params.log_gamma,
            dyn_params.log_steepness
        )
        params_dict = params_to_dict(full_params)
        history.append({
            'iteration': iteration,
            f'neg_{metric_name.lower()}': float(neg_metric),
            metric_name.lower(): float(-neg_metric),
            **params_dict,
        })

        # Check convergence after minimum iterations
        if iteration >= min_iter:
            relative_change = abs(best_neg_metric - prev_best) / (abs(prev_best) + 1e-10)
            if relative_change < tolerance:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= patience:
                if verbose:
                    print(f"  Converged after {iteration} iterations!")
                    print(f"  Final {metric_name}={-float(best_neg_metric):7.3f}")
                break

        prev_best = best_neg_metric

        if verbose and (iteration % 25 == 0 or iteration == max_iter):
            print(f"  Iter {iteration:3d}: {metric_name}={-float(neg_metric):7.3f} (best={-float(best_neg_metric):7.3f}) | "
                  f"step_size={params_dict['step_size']:.4f}, "
                  f"T={params_dict['total_time']:.2f}, "
                  f"gamma={params_dict['gamma']:.4f}, "
                  f"steep={params_dict['steepness']:.2f}")

    if verbose and iteration == max_iter:
        print(f"  Reached max iterations ({max_iter})")

    return best_dyn_params, history, best_neg_metric

# ============================================================================
# Tuning Functions for Each Sampler
# ============================================================================

def tune_rwmh(
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    max_iter: int = 500,
    min_iter: int = 50,
    tolerance: float = 0.01,
    patience: int = 10,
    num_samples: int = 1000,
    burn_in: int = 500,
    n_runs_per_eval: int = 2,
    learning_rate: float = 0.02,
    verbose: bool = True,
) -> RWMHParams:
    """Tune RWMH proposal_scale using gradient-based ESJD optimization."""
    # Initialize
    d = init_position.shape[-1]
    initial_scale = 2.38 / jnp.sqrt(d)  # Roberts & Rosenthal optimal starting point
    params = RWMHParams(log_proposal_scale=jnp.log(initial_scale))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    grad_fn = jax.grad(objective_rwmh_variance_reduced, argnums=0)

    best_neg_metric = jnp.inf
    best_params = params
    prev_best = jnp.inf
    converged_count = 0

    if verbose:
        print(f"\n{'='*80}\nTUNING RWMH\n{'='*80}")

    for iteration in range(1, max_iter + 1):
        key, *eval_keys = random.split(key, n_runs_per_eval + 1)
        eval_keys = jnp.array(eval_keys)

        neg_metric = objective_rwmh_variance_reduced(params, eval_keys, log_prob_fn, init_position, num_samples, burn_in)
        grads = grad_fn(params, eval_keys, log_prob_fn, init_position, num_samples, burn_in)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if neg_metric < best_neg_metric:
            best_neg_metric = neg_metric
            best_params = params

        # Convergence check
        if iteration >= min_iter:
            relative_change = abs(best_neg_metric - prev_best) / (abs(prev_best) + 1e-10)
            converged_count = converged_count + 1 if relative_change < tolerance else 0
            if converged_count >= patience:
                if verbose:
                    print(f"  Converged after {iteration} iterations!")
                break

        prev_best = best_neg_metric

        if verbose and (iteration % 25 == 0 or iteration == max_iter):
            scale = float(jnp.exp(best_params.log_proposal_scale))
            print(f"  Iter {iteration:3d}: ESJD={-float(best_neg_metric):7.3f} | scale={scale:.4f}")

    if verbose:
        final_dict = params_to_dict(best_params)
        print(f"  Final proposal_scale = {final_dict['proposal_scale']:.4f}")

    return best_params


def tune_hmc(
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    max_iter: int = 500,
    min_iter: int = 50,
    tolerance: float = 0.01,
    patience: int = 10,
    num_samples: int = 1000,
    burn_in: int = 500,
    n_runs_per_eval: int = 3,
    learning_rate: float = 0.02,
    verbose: bool = True,
) -> HMCParams:
    """Tune HMC step_size and total_time using gradient-based ESJD optimization."""
    params = HMCParams(log_step_size=jnp.log(0.2), log_total_time=jnp.log(10.0))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    grad_fn = jax.grad(objective_hmc_variance_reduced, argnums=0)

    best_neg_metric = jnp.inf
    best_params = params
    prev_best = jnp.inf
    converged_count = 0

    if verbose:
        print(f"\n{'='*80}\nTUNING HMC\n{'='*80}")

    for iteration in range(1, max_iter + 1):
        key, *eval_keys = random.split(key, n_runs_per_eval + 1)
        eval_keys = jnp.array(eval_keys)

        neg_metric = objective_hmc_variance_reduced(params, eval_keys, log_prob_fn, init_position, num_samples, burn_in)
        grads = grad_fn(params, eval_keys, log_prob_fn, init_position, num_samples, burn_in)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if neg_metric < best_neg_metric:
            best_neg_metric = neg_metric
            best_params = params

        if iteration >= min_iter:
            relative_change = abs(best_neg_metric - prev_best) / (abs(prev_best) + 1e-10)
            converged_count = converged_count + 1 if relative_change < tolerance else 0
            if converged_count >= patience:
                if verbose:
                    print(f"  Converged after {iteration} iterations!")
                break

        prev_best = best_neg_metric

        if verbose:
            pdict_current = params_to_dict(params)
            grad_norms = f"grads: step={float(grads.log_step_size):7.2f}, T={float(grads.log_total_time):7.2f}"
            rel_change = abs(best_neg_metric - prev_best) / (abs(prev_best) + 1e-10) if iteration > 1 else float('inf')
            # Print every iteration for monitoring
            print(f"  {iteration:3d}: ESJD={-float(neg_metric):7.3f} (best={-float(best_neg_metric):7.3f}) | "
                  f"step={pdict_current['step_size']:.4f}, T={pdict_current['total_time']:.2f} | "
                  f"{grad_norms} | change={rel_change:.5f}")

    if verbose:
        final_dict = params_to_dict(best_params)
        print(f"  Final: step_size={final_dict['step_size']:.4f}, T={final_dict['total_time']:.2f}, E[L]={final_dict['expected_num_steps']:.1f}")

    return best_params


def tune_nuts(
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    max_iter: int = 500,
    min_iter: int = 50,
    tolerance: float = 0.01,
    patience: int = 10,
    num_samples: int = 1000,
    burn_in: int = 500,
    n_runs_per_eval: int = 2,
    learning_rate: float = 0.02,
    verbose: bool = True,
) -> NUTSParams:
    """Tune NUTS step_size using gradient-based ESJD optimization."""
    params = NUTSParams(log_step_size=jnp.log(0.2))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    grad_fn = jax.grad(objective_nuts_variance_reduced, argnums=0)

    best_neg_metric = jnp.inf
    best_params = params
    prev_best = jnp.inf
    converged_count = 0

    if verbose:
        print(f"\n{'='*80}\nTUNING NUTS\n{'='*80}")

    for iteration in range(1, max_iter + 1):
        key, *eval_keys = random.split(key, n_runs_per_eval + 1)
        eval_keys = jnp.array(eval_keys)

        neg_metric = objective_nuts_variance_reduced(params, eval_keys, log_prob_fn, init_position, num_samples, burn_in)
        grads = grad_fn(params, eval_keys, log_prob_fn, init_position, num_samples, burn_in)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if neg_metric < best_neg_metric:
            best_neg_metric = neg_metric
            best_params = params

        if iteration >= min_iter:
            relative_change = abs(best_neg_metric - prev_best) / (abs(prev_best) + 1e-10)
            converged_count = converged_count + 1 if relative_change < tolerance else 0
            if converged_count >= patience:
                if verbose:
                    print(f"  Converged after {iteration} iterations!")
                break

        prev_best = best_neg_metric

        if verbose and (iteration % 25 == 0 or iteration == max_iter):
            step_size = float(jnp.exp(best_params.log_step_size))
            print(f"  Iter {iteration:3d}: ESJD={-float(best_neg_metric):7.3f} | step_size={step_size:.4f}")

    if verbose:
        final_dict = params_to_dict(best_params)
        print(f"  Final step_size = {final_dict['step_size']:.4f}")

    return best_params


def tune_grahmc(
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    schedule_types: List[str] = None,
    initial_params: DynamicParams = None,
    max_iter: int = 500,
    min_iter: int = 50,
    tolerance: float = 0.01,
    patience: int = 10,
    num_samples: int = 2000,
    burn_in: int = 500,
    n_runs_per_eval: int = 3,
    learning_rate: float = 0.015,
    verbose: bool = True,
) -> Dict[str, Tuple[TuningParams, List[Dict]]]:
    """
    Optimize RAHMC parameters for all friction schedules.

    Uses gradient descent to jointly optimize (step_size, total_time, gamma, steepness).
    No grid search needed - total_time is optimized continuously via stochastic rounding.
    Convergence detection automatically stops when parameters stabilize.

    Returns:
        Dictionary mapping schedule_type -> (best_params, optimization_history)
    """

    if schedule_types is None:
        schedule_types = ['constant', 'tanh', 'sigmoid', 'linear', 'sine']

    if initial_params is None:
        # Default initialization
        initial_params = DynamicParams(
            log_step_size=jnp.log(0.2),     # epsilon ~ 0.2
            log_total_time=jnp.log(10.0),   # T ~ 10 (expect ~50 steps initially)
            log_gamma=jnp.log(0.2),         # gamma ~ 0.2
            log_steepness=jnp.log(2.0),     # steepness ~ 2.0
        )

    results = {}

    for schedule_type in schedule_types:
        if verbose:
            print("\n" + "="*80)
            print(f"OPTIMIZING: {schedule_type.upper()}")
            print("="*80)

        key, subkey = random.split(key)

        tuned_params, history, final_neg_metric = optimize_parameters(
            key=subkey,
            log_prob_fn=log_prob_fn,
            init_position=init_position,
            schedule_type=schedule_type,
            initial_params=initial_params,
            num_samples=num_samples,
            burn_in=burn_in,
            max_iter=max_iter,
            min_iter=min_iter,
            tolerance=tolerance,
            patience=patience,
            n_runs_per_eval=n_runs_per_eval,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        # Convert to full TuningParams for consistency
        best_overall_params = TuningParams(
            log_step_size=tuned_params.log_step_size,
            log_total_time=tuned_params.log_total_time,
            log_gamma=tuned_params.log_gamma,
            log_steepness=tuned_params.log_steepness
        )

        results[schedule_type] = (best_overall_params, history)

        # Print summary
        if verbose:
            print("\n" + "-"*80)
            print(f"OPTIMIZATION COMPLETE FOR: {schedule_type.upper()}")
            print("-"*80)
            best_dict = params_to_dict(best_overall_params)
            print(f"Best Metric: {-float(final_neg_metric):.3f}")
            print(f"Optimal parameters found:")
            print(f"  step_size        = {best_dict['step_size']:.4f}")
            print(f"  T (total_time)   = {best_dict['total_time']:.2f}")
            print(f"  E[L] (num_steps) = {best_dict['expected_num_steps']:.1f}")
            print(f"  gamma            = {best_dict['gamma']:.4f}")
            print(f"  steepness        = {best_dict['steepness']:.2f}")
            print("="*80 + "\n")

    return results


# ============================================================================
# Dispatcher Function
# ============================================================================

def tune_sampler(
    sampler: str,
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    schedule: str = 'constant',
    max_iter: int = 500,
    min_iter: int = 50,
    tolerance: float = 0.01,
    patience: int = 10,
    num_samples: int = 1000,
    burn_in: int = 500,
    n_runs_per_eval: int = 3,
    learning_rate: float = 0.02,
    verbose: bool = True,
) -> Any:
    """
    Dispatcher function to tune any supported MCMC sampler.

    Parameters:
        sampler: Sampler name ('rwmh', 'hmc', 'grahmc', 'nuts')
        key: JAX random key
        log_prob_fn: Target log probability function
        init_position: Initial positions (n_chains, n_dim)
        schedule: Friction schedule for GRAHMC ('constant', 'tanh', 'sigmoid', 'linear', 'sine')
        max_iter: Maximum optimization iterations
        min_iter: Minimum iterations before convergence check
        tolerance: Convergence tolerance for relative change
        patience: Number of converged iterations required
        num_samples: Number of MCMC samples per evaluation
        burn_in: Burn-in samples per evaluation
        n_runs_per_eval: Number of independent runs for variance reduction
        learning_rate: Adam learning rate
        verbose: Print optimization progress

    Returns:
        Tuned parameters (RWMHParams, HMCParams, GRAHMCParams, or NUTSParams)

    Examples:
        # Tune RWMH
        params = tune_sampler('rwmh', key, log_prob_fn, init_position)

        # Tune GRAHMC with tanh schedule
        params = tune_sampler('grahmc', key, log_prob_fn, init_position, schedule='tanh')

        # Tune NUTS
        params = tune_sampler('nuts', key, log_prob_fn, init_position)
    """
    sampler_lower = sampler.lower()

    if sampler_lower == 'rwmh':
        return tune_rwmh(
            key, log_prob_fn, init_position,
            max_iter=max_iter,
            min_iter=min_iter,
            tolerance=tolerance,
            patience=patience,
            num_samples=num_samples,
            burn_in=burn_in,
            n_runs_per_eval=n_runs_per_eval,
            learning_rate=learning_rate,
            verbose=verbose,
        )

    elif sampler_lower == 'hmc':
        return tune_hmc(
            key, log_prob_fn, init_position,
            max_iter=max_iter,
            min_iter=min_iter,
            tolerance=tolerance,
            patience=patience,
            num_samples=num_samples,
            burn_in=burn_in,
            n_runs_per_eval=n_runs_per_eval,
            learning_rate=learning_rate,
            verbose=verbose,
        )

    elif sampler_lower == 'grahmc':
        # For GRAHMC, tune single schedule or all schedules
        if schedule.lower() == 'all':
            # Tune all schedules
            return tune_grahmc(
                key, log_prob_fn, init_position,
                schedule_types=['constant', 'tanh', 'sigmoid', 'linear', 'sine'],
                max_iter=max_iter,
                min_iter=min_iter,
                tolerance=tolerance,
                patience=patience,
                num_samples=num_samples,
                burn_in=burn_in,
                n_runs_per_eval=n_runs_per_eval,
                learning_rate=learning_rate,
                verbose=verbose,
            )
        else:
            # Tune single schedule
            results = tune_grahmc(
                key, log_prob_fn, init_position,
                schedule_types=[schedule.lower()],
                max_iter=max_iter,
                min_iter=min_iter,
                tolerance=tolerance,
                patience=patience,
                num_samples=num_samples,
                burn_in=burn_in,
                n_runs_per_eval=n_runs_per_eval,
                learning_rate=learning_rate,
                verbose=verbose,
            )
            # Extract the single result
            return results[schedule.lower()][0]  # Return just the params

    elif sampler_lower == 'nuts':
        return tune_nuts(
            key, log_prob_fn, init_position,
            max_iter=max_iter,
            min_iter=min_iter,
            tolerance=tolerance,
            patience=patience,
            num_samples=num_samples,
            burn_in=burn_in,
            n_runs_per_eval=n_runs_per_eval,
            learning_rate=learning_rate,
            verbose=verbose,
        )

    else:
        raise ValueError(
            f"Unknown sampler '{sampler}'. "
            f"Supported samplers: 'rwmh', 'hmc', 'grahmc', 'nuts'"
        )


# ============================================================================
# Sampling Performance Evaluation
# ============================================================================
def evaluate_sampling_performance(
    key: jnp.ndarray,
    log_prob_fn: Callable,
    init_position: jnp.ndarray,
    results: Dict[str, Tuple[TuningParams, List[Dict]]],
    num_samples: int = 5000,
    burn_in: int = 1000,
) -> Dict[str, Dict]:
    """
    Evaluate and compare sampling performance across all schedules.

    Uses the optimized parameters to run extended sampling and compute
    ESS, ESJD, and other performance metrics.

    For evaluation, we use the expected number of steps (rounded) to ensure
    deterministic trajectory length.
    """

    performance = {}

    print("\n" + "=" * 80)
    print("EVALUATING SAMPLING PERFORMANCE")
    print("=" * 80)

    for schedule_type, (params, _) in results.items():
        key, subkey = random.split(key)

        params_dict = params_to_dict(params)
        step_size = params_dict["step_size"]
        total_time = params_dict["total_time"]
        # Use expected num_steps (rounded) for deterministic evaluation
        num_steps = int(jnp.round(total_time / step_size))
        num_steps = max(num_steps, 1)
        gamma = params_dict["gamma"]
        steepness = params_dict["steepness"]

        friction_schedule = get_friction_schedule(schedule_type)

        import time
        start = time.time()
        results_tuple = rahmc_run(
            key=subkey,
            log_prob_fn=log_prob_fn,
            init_position=init_position,
            step_size=step_size,
            num_steps=num_steps,
            gamma=gamma,
            steepness=steepness,
            num_samples=num_samples,
            burn_in=burn_in,
            friction_schedule=friction_schedule,
            track_proposals=True,
        )
        elapsed = time.time() - start

        (samples, lps, accept_rate, final_state,
         pre_positions, pre_lps, prop_positions, prop_lps, deltas_H) = results_tuple

        # ESS diagnostics 
        samples_transposed = jnp.transpose(samples, (1, 0, 2))
        ess_per_dim = estimate_ess_bulk(samples_transposed)
        mean_ess = float(jnp.mean(ess_per_dim))
        min_ess = float(jnp.min(ess_per_dim))

        proposal_esjd = float(compute_proposal_esjd_soft(pre_positions, prop_positions, deltas_H))

        # Accepted ESJD (realized)
        diffs = samples[1:] - samples[:-1] # (S-1, C, D)
        accepted_esjd = float(jnp.mean(jnp.sum(diffs ** 2, axis=-1)))

        ess_per_second = mean_ess / max(elapsed, 1e-9)
        mean_accept = float(jnp.mean(accept_rate))

        performance[schedule_type] = {
            "mean_ess": mean_ess,
            "min_ess": min_ess,
            "proposal_esjd": proposal_esjd,
            "accepted_esjd": accepted_esjd,
            "ess_per_second": ess_per_second,
            "accept_rate": mean_accept,
            "time": elapsed,
            "samples": samples,
            "log_probs": lps,
            "params": params_dict,
        }

        print(f"\n{schedule_type.upper():12s} | "
              f"ESS: {mean_ess:7.1f} | "
              f"Prop-ESJD: {proposal_esjd:8.3f} | "
              f"Acc-ESJD: {accepted_esjd:8.3f} | "
              f"Accept: {mean_accept:.3f} | "
              f"Time: {elapsed:6.2f}s")

    print("=" * 80 + "\n")
    return performance


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_performance_summary(performance: Dict[str, Dict]):
    """Bar plots comparing key performance metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    schedules = list(performance.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # ESS comparison
    mean_ess = [performance[s].get('mean_ess', float('nan')) for s in schedules]
    axes[0].bar(schedules, mean_ess, color=colors[:len(schedules)], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Mean ESS', fontsize=12)
    axes[0].set_title('Effective Sample Size', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Proposal ESJD
    prop_esjd = [performance[s].get('proposal_esjd', float('nan')) for s in schedules]
    acc_esjd = [performance[s].get('accepted_esjd', float('nan')) for s in schedules]
    x = np.arange(len(schedules))
    width = 0.35
    axes[1].bar(x - width/2, prop_esjd, width, label='Proposal',
                   color=colors[:len(schedules)], alpha=0.7, edgecolor='black')
    axes[1].bar(x + width/2, acc_esjd, width, label='Trajectory',
                   color='#95a5a6', alpha=0.6, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(schedules)
    axes[1].set_ylabel('ESJD', fontsize=12)
    axes[1].set_title('ESJD (proposal vs accepted)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Acceptance rate
    accept_rates = [performance[s].get('accept_rate', float('nan')) for s in schedules]
    axes[2].bar(schedules, accept_rates, color=colors[:len(schedules)], alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Acceptance Rate', fontsize=12)
    axes[2].set_title('Metropolis Acceptance', fontsize=13, fontweight='bold')
    axes[2].axhline(0.651, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (HMC)')
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    # ESS per second
    ess_per_sec = [performance[s].get('ess_per_second', float('nan')) for s in schedules]
    axes[3].bar(schedules, ess_per_sec, color=colors[:len(schedules)], alpha=0.7, edgecolor='black')
    axes[3].set_ylabel('ESS per Second', fontsize=12)
    axes[3].set_title('Sampling Efficiency', fontsize=13, fontweight='bold')
    axes[3].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved performance summary to 'performance_summary.png'")
    plt.show()


def plot_friction_schedules(results: Dict[str, Tuple[TuningParams, List[Dict]]]):
    """Visualize the optimized friction schedules."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'constant': '#e74c3c',
        'tanh': '#3498db',
        'sigmoid': '#2ecc71',
        'linear': '#f39c12',
        'sine': '#9b59b6',
    }
    
    t_vals = jnp.linspace(0, 1, 200)
    
    for schedule_type, (params, _) in results.items():
        params_dict = params_to_dict(params)
        gamma = params_dict['gamma']
        steepness = params_dict['steepness']
        
        friction_fn = get_friction_schedule(schedule_type)
        gamma_vals = vmap(lambda t: friction_fn(t, 1.0, gamma, steepness))(t_vals)
        
        ax.plot(t_vals, gamma_vals, 
               label=f"{schedule_type} (γ={gamma:.3f})",
               color=colors.get(schedule_type, 'gray'),
               linewidth=2.5, alpha=0.8)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Normalized Time (t/T)', fontsize=12)
    ax.set_ylabel('Friction γ(t)', fontsize=12)
    ax.set_title('Optimized Friction Schedules', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('friction_schedules.png', dpi=150, bbox_inches='tight')
    print("✓ Saved friction schedules to 'friction_schedules.png'")
    plt.show()



def plot_sampling_comparison(performance: Dict[str, Dict], dim: int = 0):
    """Plot trace plots and marginals for comparison."""
    
    n_schedules = len(performance)
    fig, axes = plt.subplots(n_schedules, 2, figsize=(14, 3*n_schedules))
    
    if n_schedules == 1:
        axes = axes[None, :]
    
    colors = {
        'constant': '#e74c3c',
        'tanh': '#3498db',
        'sigmoid': '#2ecc71',
        'linear': '#f39c12',
        'sine': '#9b59b6',
    }
    
    for idx, (schedule_type, perf) in enumerate(performance.items()):
        prop_esjd = perf.get('proposal_esjd', float('nan'))
        samples = perf['samples']
        color = colors.get(schedule_type, 'gray')
        
        # Trace plot (first chain, specified dimension)
        ax_trace = axes[idx, 0]
        trace = samples[:, 0, dim]  # (n_samples,)
        ax_trace.plot(trace, color=color, alpha=0.7, linewidth=0.5)
        ax_trace.set_ylabel(f'{schedule_type}\nDim {dim}', fontsize=11)
        ax_trace.set_xlabel('Sample', fontsize=10)
        ax_trace.grid(True, alpha=0.3)
        ax_trace.set_title(f"Trace Plot | ESS={perf['mean_ess']:.0f}, ESJD={prop_esjd:.3f}", fontsize=11)
        
        # Marginal histogram
        ax_hist = axes[idx, 1]
        all_samples = samples[:, :, dim].flatten()  # all chains
        ax_hist.hist(all_samples, bins=50, color=color, alpha=0.6, 
                     density=True, edgecolor='black', linewidth=0.5)
        
        # Overlay true density (for Gaussian)
        x_range = jnp.linspace(-4, 4, 200)
        true_density = jnp.exp(-0.5 * x_range**2) / jnp.sqrt(2 * jnp.pi)
        ax_hist.plot(x_range, true_density, 'k--', linewidth=2, label='True')
        
        ax_hist.set_ylabel('Density', fontsize=10)
        ax_hist.set_xlabel(f'Dimension {dim}', fontsize=10)
        ax_hist.set_title(f"Marginal | Accept={perf['accept_rate']:.3f}", fontsize=11)
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved sampling comparison to 'sampling_comparison.png'")
    plt.show()


def plot_optimization_history(results: Dict[str, Tuple[TuningParams, List[Dict]]]):
    """Plot optimization history for all schedules."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Colors for each schedule
    colors = {
        'constant': '#e74c3c',
        'tanh': '#3498db',
        'sigmoid': '#2ecc71',
        'linear': '#f39c12',
        'sine': '#9b59b6',
    }
    
    metric_key = "ess"
    metric_title = "ESS"
    for _, history in results.items():
        if history:
            entry = history[-1]
            if "esjd" in entry:
                metric_key, metric_title = "esjd", "ESJD"
            elif "mixing" in entry:
                metric_key, metric_title = "mixing", "Mixing"
            elif "ess" in entry:
                metric_key, metric_title = "ess", "ESS"
            break
    metrics = [metric_key, "step_size", "num_steps", "gamma", "steepness"]
    titles = [metric_title, "Step Size (ε)", "Num Steps (L)", "Gamma (γ)", "Steepness"]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for schedule_type, (_, history) in results.items():
            iterations = [h['iteration'] for h in history]
            values = [h.get(metric, float("nan")) for h in history]
            
            ax.plot(iterations, values, 
                   label=schedule_type, 
                   color=colors.get(schedule_type, 'gray'),
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved optimization history to 'optimization_history.png'")
    plt.show()


# ============================================================================
# Complete Analysis Pipeline
# ============================================================================

def run_complete_analysis(
    dim: int = 10,
    n_optimization_steps: int = 50,
    n_eval_samples: int = 5000,
    seed: int = 42,
):
    """
    Run complete RAHMC tuning and evaluation pipeline.

    Performs parameter optimization for all friction schedules, then evaluates
    sampling performance and generates comprehensive visualizations.

    Args:
        dim: Dimensionality of target distribution (standard normal)
        n_optimization_steps: Number of gradient descent iterations
        n_eval_samples: Number of samples for final evaluation
        seed: Random seed for reproducibility

    Returns:
        results: Optimized parameters for each schedule
        performance: Performance metrics for each schedule
    """
    print("\n" + "="*80)
    print(f"COMPLETE RAHMC ANALYSIS: {dim}-DIMENSIONAL GAUSSIAN")
    print("="*80)
    
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    
    # Target
    log_prob_fn = standard_normal_log_prob
    init_position = random.normal(init_key, shape=(4, dim))  # 4 chains
    
    print("\n" + "─"*80)
    print("STEP 1: OPTIMIZING PARAMETERS")
    print("─"*80)
    
    key, opt_key = random.split(key)
    
    results = optimize_all_schedules(
        key=opt_key,
        log_prob_fn=log_prob_fn,
        init_position=init_position,
        schedule_types=['constant', 'tanh', 'sigmoid', 'linear', 'sine'],
        num_samples=1000,
        burn_in=500,
        n_optimization_steps=n_optimization_steps,
        n_runs_per_eval=3,
        learning_rate=0.015,
        verbose=True,
    )

    print("\n" + "─"*80)
    print("STEP 2: PLOTTING OPTIMIZATION HISTORY")
    print("─"*80)
    # only show the history for the winning L
    plot_optimization_history(results)
    
    plot_friction_schedules(results)
    
    print("\n" + "─"*80)
    print("STEP 4: EVALUATING SAMPLING PERFORMANCE")
    print("─"*80)
    
    key, eval_key = random.split(key)
    performance = evaluate_sampling_performance(
        key=eval_key,
        log_prob_fn=log_prob_fn,
        init_position=init_position,
        results=results,
        num_samples=n_eval_samples,
        burn_in=1000,
    )
    
    print("\n" + "─"*80)
    print("STEP 5: PLOTTING SAMPLING COMPARISONS")
    print("─"*80)
    plot_sampling_comparison(performance, dim=0)
    plot_performance_summary(performance)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return results, performance


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Command-line interface for tuning MCMC samplers."""
    parser = argparse.ArgumentParser(
        description='Tune MCMC sampler parameters using gradient-based ESJD optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune RWMH on 10D standard normal
  python tuning.py --sampler rwmh --dim 10

  # Tune HMC with 200 optimization steps
  python tuning.py --sampler hmc --dim 20 --max-iter 200

  # Tune GRAHMC with tanh schedule
  python tuning.py --sampler grahmc --schedule tanh --dim 10

  # Tune all GRAHMC schedules
  python tuning.py --sampler grahmc --schedule all --dim 10

  # Tune NUTS
  python tuning.py --sampler nuts --dim 10

  # Run full GRAHMC analysis (legacy mode)
  python tuning.py --full-analysis
        """
    )

    parser.add_argument(
        '--sampler',
        type=str,
        choices=['rwmh', 'hmc', 'grahmc', 'nuts'],
        help='MCMC sampler to tune'
    )
    parser.add_argument(
        '--schedule',
        type=str,
        default='constant',
        choices=['constant', 'tanh', 'sigmoid', 'linear', 'sine', 'all'],
        help='Friction schedule for GRAHMC (default: constant)'
    )
    parser.add_argument(
        '--dim',
        type=int,
        default=10,
        help='Dimensionality of target distribution (default: 10)'
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=4,
        help='Number of parallel chains (default: 4)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=500,
        help='Maximum optimization iterations (default: 500)'
    )
    parser.add_argument(
        '--min-iter',
        type=int,
        default=50,
        help='Minimum iterations before convergence check (default: 50)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.01,
        help='Convergence tolerance (default: 0.01)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Converged iterations required to stop (default: 10)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='MCMC samples per evaluation (default: 1000)'
    )
    parser.add_argument(
        '--burn-in',
        type=int,
        default=500,
        help='Burn-in samples per evaluation (default: 500)'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=3,
        help='Independent runs for variance reduction (default: 3)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.02,
        help='Adam learning rate (default: 0.02)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--full-analysis',
        action='store_true',
        help='Run full GRAHMC analysis with all schedules (legacy mode)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress optimization progress output'
    )

    args = parser.parse_args()

    # Legacy mode: full GRAHMC analysis
    if args.full_analysis:
        print("Running complete GRAHMC tuning analysis...")
        results, performance = run_complete_analysis(
            dim=args.dim,
            n_optimization_steps=args.max_iter,
            n_eval_samples=5000,
            seed=args.seed,
        )

        # Print summary table
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"{'Schedule':<12} | {'ESS':>8} | {'Prop-ESJD':>10} | {'Acc-ESJD':>10} | "
              f"{'Accept':>8} | {'step_size':>10} | {'T':>7} | {'E[L]':>6} | {'gamma':>8}")
        print("-"*80)

        for schedule_type in results.keys():
            perf = performance[schedule_type]
            params_dict = perf['params']
            prop_esjd = perf.get('proposal_esjd', float('nan'))
            acc_esjd = perf.get('accepted_esjd', float('nan'))
            mean_ess = perf.get('mean_ess', float('nan'))
            accept = perf.get('accept_rate', float('nan'))

            print(f"{schedule_type:<12} | "
                  f"{mean_ess:8.1f} | "
                  f"{prop_esjd:10.4f} | "
                  f"{acc_esjd:10.4f} | "
                  f"{accept:8.3f} | "
                  f"{params_dict['step_size']:10.4f} | "
                  f"{params_dict['total_time']:7.2f} | "
                  f"{params_dict['expected_num_steps']:6.1f} | "
                  f"{params_dict['gamma']:8.4f}")

        print("="*80 + "\n")
        return

    # Standard mode: tune specific sampler
    if args.sampler is None:
        parser.error("--sampler is required (or use --full-analysis)")

    # Setup
    key = random.PRNGKey(args.seed)
    key, init_key = random.split(key)

    # Target: standard normal
    log_prob_fn = standard_normal_log_prob
    init_position = random.normal(init_key, shape=(args.chains, args.dim))

    print("\n" + "="*80)
    print(f"TUNING {args.sampler.upper()}" +
          (f" ({args.schedule})" if args.sampler == 'grahmc' else ""))
    print("="*80)
    print(f"Target: {args.dim}-dimensional standard normal")
    print(f"Chains: {args.chains}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Convergence: tolerance={args.tolerance}, patience={args.patience}")
    print("="*80)

    # Tune
    key, tune_key = random.split(key)
    tuned_params = tune_sampler(
        sampler=args.sampler,
        key=tune_key,
        log_prob_fn=log_prob_fn,
        init_position=init_position,
        schedule=args.schedule,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        tolerance=args.tolerance,
        patience=args.patience,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        n_runs_per_eval=args.n_runs,
        learning_rate=args.learning_rate,
        verbose=not args.quiet,
    )

    # Print final parameters
    print("\n" + "="*80)
    print("TUNING COMPLETE")
    print("="*80)

    if args.sampler == 'grahmc' and args.schedule == 'all':
        # Multiple schedules returned
        print("Optimized parameters for all schedules:")
        for schedule_name, (params, _) in tuned_params.items():
            params_dict = params_to_dict(params)
            print(f"\n{schedule_name.upper()}:")
            for key, value in params_dict.items():
                print(f"  {key:20s} = {value:.4f}")
    else:
        # Single parameter set returned
        params_dict = params_to_dict(tuned_params)
        print("Optimized parameters:")
        for key, value in params_dict.items():
            print(f"  {key:20s} = {value:.4f}")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
