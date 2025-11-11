"""Hamiltonian Monte Carlo (HMC) sampler implementation.

This module implements the Hamiltonian Monte Carlo algorithm with:
- Leapfrog integration for Hamiltonian dynamics
- Parallel chain execution via JAX vmap
- Configurable step size and trajectory length
- Burn-in support with acceptance counter reset
- Optional proposal tracking for diagnostics
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, random, vmap, lax

# Type aliases
Array = jnp.ndarray
LogProbFn = Callable[[Array], Array]  # Maps x -> log p(x)


class HMCState(NamedTuple):
    """State for Hamiltonian Monte Carlo sampler.

    Attributes:
        position: Current positions, shape (n_chains, n_dim)
        log_prob: Log probabilities at current positions, shape (n_chains,) [float64]
        grad_log_prob: Gradients of log probability, shape (n_chains, n_dim)
        accept_count: Number of accepted proposals per chain, shape (n_chains,) [int32]
    """
    position: Array
    log_prob: Array
    grad_log_prob: Array
    accept_count: Array


def _ensure_batched(x: Array) -> Tuple[Array, bool]:
    """Ensure input has batched shape (n_chains, n_dim).

    Args:
        x: Input array with shape (n_dim,) or (n_chains, n_dim)

    Returns:
        Tuple of (batched_array, was_unbatched) where batched_array has shape
        (n_chains, n_dim) and was_unbatched is True if a batch dimension was added.

    Raises:
        ValueError: If input has invalid shape (not 1D or 2D)
    """
    x = jnp.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    elif x.ndim == 2:
        return x, False
    else:
        raise ValueError("Input must have shape (n_dim,) or (n_chains, n_dim).")



def hmc_init(init_position: Array, log_prob_fn: LogProbFn) -> HMCState:
    """Initialize state for Hamiltonian Monte Carlo sampler.

    Args:
        init_position: Initial positions with shape (n_dim,) or (n_chains, n_dim)
        log_prob_fn: Function mapping positions to log probabilities

    Returns:
        Initial HMCState with positions, log probs, gradients, and zero accept counts
    """
    pos, _ = _ensure_batched(init_position)
    n_chains = pos.shape[0]
    log_prob, grad_log_prob = vmap(jax.value_and_grad(log_prob_fn))(pos)
    log_prob = log_prob.astype(jnp.float64)  # Use float64 for numerical stability
    grad_log_prob = grad_log_prob.astype(pos.dtype)
    accept_count = jnp.zeros(n_chains, dtype=jnp.int32)
    return HMCState(
        position=pos,
        log_prob=log_prob,
        grad_log_prob=grad_log_prob,
        accept_count=accept_count
    )


@partial(jit, static_argnames=("log_prob_fn", "num_steps"))
def leapfrog(
    position: Array,
    momentum: Array,
    step_size: float,
    log_prob: Array,
    grad_log_prob: Array,
    log_prob_fn: LogProbFn,
    num_steps: int,
) -> Tuple[Array, Array, Array, Array]:
    """Perform leapfrog integration for Hamiltonian dynamics.

    Args:
        position: Current positions, shape (n_chains, n_dim)
        momentum: Current momenta, shape (n_chains, n_dim)
        step_size: Integration step size
        log_prob: Current log probabilities, shape (n_chains,)
        grad_log_prob: Current gradients of log prob, shape (n_chains, n_dim)
        log_prob_fn: Function to compute log probability and gradient
        num_steps: Number of leapfrog steps

    Returns:
        Tuple of (final_position, final_momentum, final_grad_log_prob, final_log_prob)
    """
    pos_dtype = position.dtype
    lp_dtype = log_prob.dtype
    step_sz = jnp.asarray(step_size, dtype=pos_dtype)
    half = jnp.array(0.5, dtype=pos_dtype)

    def lf_step(carry, _):
        pos, mom, lp, grad_lp = carry
        # Half step for momentum
        mom = mom + half * step_sz * grad_lp
        # Full step for position
        pos = pos + step_sz * mom
        # Update gradient at new position
        new_lp, new_grad_lp = vmap(jax.value_and_grad(log_prob_fn))(pos)
        new_lp = new_lp.astype(lp_dtype)
        new_grad_lp = new_grad_lp.astype(pos_dtype)
        # Half step for momentum
        mom = mom + half * step_sz * new_grad_lp
        return (pos, mom, new_lp, new_grad_lp), None

    (final_pos, final_mom, final_lp, final_grad_lp), _ = lax.scan(
        lf_step, (position, momentum, log_prob, grad_log_prob), length=num_steps
    )

    return final_pos, final_mom, final_grad_lp, final_lp


@partial(jit, static_argnames=("log_prob_fn", "max_steps"))
def leapfrog_dynamic(
    position: Array,
    momentum: Array,
    step_size: float,
    log_prob: Array,
    grad_log_prob: Array,
    log_prob_fn: LogProbFn,
    actual_steps: Array,  # Continuous, differentiable
    max_steps: int,  # Static maximum
) -> Tuple[Array, Array, Array, Array]:
    """Perform leapfrog integration with dynamic trajectory length.

    Uses conditional masking to only apply leapfrog updates for i < actual_steps,
    allowing gradients to flow through the continuous actual_steps parameter.

    Args:
        position: Current positions, shape (n_chains, n_dim)
        momentum: Current momenta, shape (n_chains, n_dim)
        step_size: Integration step size
        log_prob: Current log probabilities, shape (n_chains,)
        grad_log_prob: Current gradients of log prob, shape (n_chains, n_dim)
        log_prob_fn: Function to compute log probability and gradient
        actual_steps: Continuous number of steps (JAX array, differentiable)
        max_steps: Maximum number of steps (static, for JIT compilation)

    Returns:
        Tuple of (final_position, final_momentum, final_grad_log_prob, final_log_prob)
    """
    pos_dtype = position.dtype
    lp_dtype = log_prob.dtype
    step_sz = jnp.asarray(step_size, dtype=pos_dtype)
    half = jnp.array(0.5, dtype=pos_dtype)

    def lf_step(carry, i):
        pos, mom, lp, grad_lp = carry

        # Straight-through estimator for gradient flow while maintaining MCMC correctness
        # Forward pass: Hard threshold (i < actual_steps) - maintains detailed balance
        # Backward pass: Soft sigmoid - allows gradients to flow through actual_steps
        i_float = jnp.array(i, dtype=pos_dtype)

        # Soft threshold for gradients (backward pass)
        sharpness = 10.0
        weight_soft = jax.nn.sigmoid((actual_steps - i_float) * sharpness)

        # Hard threshold for correctness (forward pass)
        weight_hard = (i_float < actual_steps).astype(pos_dtype)

        # Straight-through: hard forward, soft backward
        weight = weight_soft + jax.lax.stop_gradient(weight_hard - weight_soft)

        # Half step for momentum
        new_mom = mom + half * step_sz * grad_lp
        # Full step for position
        new_pos = pos + step_sz * new_mom
        # Update gradient at new position
        new_lp, new_grad_lp = vmap(jax.value_and_grad(log_prob_fn))(new_pos)
        new_lp = new_lp.astype(lp_dtype)
        new_grad_lp = new_grad_lp.astype(pos_dtype)
        # Half step for momentum
        new_mom = new_mom + half * step_sz * new_grad_lp

        # Apply update using straight-through weights
        # In forward pass: weight is 0 or 1 (discrete, maintains MCMC)
        # In backward pass: weight has smooth gradient (allows optimization)
        pos = weight * new_pos + (1 - weight) * pos
        mom = weight * new_mom + (1 - weight) * mom
        lp = weight * new_lp + (1 - weight) * lp
        grad_lp = weight * new_grad_lp + (1 - weight) * grad_lp

        return (pos, mom, lp, grad_lp), None

    (final_pos, final_mom, final_lp, final_grad_lp), _ = lax.scan(
        lf_step, (position, momentum, log_prob, grad_log_prob), jnp.arange(max_steps)
    )

    return final_pos, final_mom, final_grad_lp, final_lp


@partial(jit, static_argnames=("log_prob_fn", "num_steps", "return_proposal"))
def hmc_step(
    state: HMCState,
    log_prob_fn: LogProbFn,
    step_size: float,
    num_steps: int,
    key: Array,
    return_proposal: bool = False,
) -> Tuple[Array, HMCState] | Tuple[Array, HMCState, Array, Array, Array]:
    """Perform one HMC step with Metropolis-Hastings acceptance.

    Args:
        state: Current HMC state
        log_prob_fn: Function to compute log probability and gradient
        step_size: Leapfrog integration step size
        num_steps: Number of leapfrog steps
        key: JAX random key
        return_proposal: If True, return proposal info for diagnostics

    Returns:
        If return_proposal=False: (next_key, new_state)
        If return_proposal=True: (next_key, new_state, proposal_pos, proposal_log_prob, delta_H)
    """
    n_chains, n_dim = state.position.shape
    pos_dtype = state.position.dtype
    logprob_dtype = state.log_prob.dtype

    # next_key, k_momentum, k_accept = random.split(key, 3)
    key, step_key = random.split(key)
    k_momentum, k_accept = random.split(step_key, 2)

    momentum = random.normal(k_momentum, shape=(n_chains, n_dim), dtype=pos_dtype)
    step_size_arr = jnp.asarray(step_size, dtype=pos_dtype)

    kinetic_initial = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_initial = -state.log_prob + kinetic_initial.astype(logprob_dtype)

    current_position = state.position
    grad_lp = state.grad_log_prob
    log_prob = state.log_prob

    current_position, momentum, grad_lp, log_prob = leapfrog(
        current_position,
        momentum,
        step_size_arr,
        log_prob,
        grad_lp,
        log_prob_fn=log_prob_fn,
        num_steps=num_steps,
    )

    momentum = -momentum 

    kinetic_final = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_final = -log_prob + kinetic_final.astype(logprob_dtype)
    # overflow protection
    hamiltonian_final = jnp.where(jnp.isfinite(hamiltonian_final), hamiltonian_final, jnp.array(1e10, dtype=logprob_dtype))


    log_alpha = hamiltonian_initial - hamiltonian_final
    delta_H = hamiltonian_final - hamiltonian_initial

    u = random.uniform(k_accept, shape=(n_chains,), dtype=logprob_dtype)
    zero = jnp.array(0.0, dtype=logprob_dtype)
    accept = jnp.log(u) < jnp.minimum(zero, log_alpha)

    new_position = jnp.where(accept[:, None], current_position, state.position)
    new_log_prob = jnp.where(accept, log_prob, state.log_prob)
    new_grad_log_prob = jnp.where(accept[:, None], grad_lp, state.grad_log_prob)
    new_accept_count = state.accept_count + accept.astype(jnp.int32)

    new_state = HMCState(new_position, new_log_prob, new_grad_log_prob, new_accept_count)

    if return_proposal:
        return key, new_state, current_position, log_prob, delta_H
    else:
        return key, new_state


@partial(jit, static_argnames=("log_prob_fn", "max_steps", "return_proposal"))
def hmc_step_dynamic(
    state: HMCState,
    log_prob_fn: LogProbFn,
    step_size: float,
    actual_steps: Array,  # Continuous, differentiable
    max_steps: int,  # Static maximum
    key: Array,
    return_proposal: bool = False,
) -> Tuple[Array, HMCState] | Tuple[Array, HMCState, Array, Array, Array]:
    """Perform one HMC step with dynamic trajectory length.

    Args:
        state: Current HMC state
        log_prob_fn: Function to compute log probability and gradient
        step_size: Leapfrog integration step size
        actual_steps: Continuous number of steps (JAX array, differentiable)
        max_steps: Maximum number of steps (static, for JIT)
        key: JAX random key
        return_proposal: If True, return proposal info for diagnostics

    Returns:
        If return_proposal=False: (next_key, new_state)
        If return_proposal=True: (next_key, new_state, proposal_pos, proposal_log_prob, delta_H)
    """
    n_chains, n_dim = state.position.shape
    pos_dtype = state.position.dtype
    logprob_dtype = state.log_prob.dtype

    key, step_key = random.split(key)
    k_momentum, k_accept = random.split(step_key, 2)

    momentum = random.normal(k_momentum, shape=(n_chains, n_dim), dtype=pos_dtype)
    step_size_arr = jnp.asarray(step_size, dtype=pos_dtype)

    kinetic_initial = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_initial = -state.log_prob + kinetic_initial.astype(logprob_dtype)

    current_position = state.position
    grad_lp = state.grad_log_prob
    log_prob = state.log_prob

    current_position, momentum, grad_lp, log_prob = leapfrog_dynamic(
        current_position,
        momentum,
        step_size_arr,
        log_prob,
        grad_lp,
        log_prob_fn=log_prob_fn,
        actual_steps=actual_steps,
        max_steps=max_steps,
    )

    momentum = -momentum

    kinetic_final = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_final = -log_prob + kinetic_final.astype(logprob_dtype)
    # overflow protection
    hamiltonian_final = jnp.where(jnp.isfinite(hamiltonian_final), hamiltonian_final, jnp.array(1e10, dtype=logprob_dtype))

    log_alpha = hamiltonian_initial - hamiltonian_final
    delta_H = hamiltonian_final - hamiltonian_initial

    u = random.uniform(k_accept, shape=(n_chains,), dtype=logprob_dtype)
    zero = jnp.array(0.0, dtype=logprob_dtype)
    accept = jnp.log(u) < jnp.minimum(zero, log_alpha)

    new_position = jnp.where(accept[:, None], current_position, state.position)
    new_log_prob = jnp.where(accept, log_prob, state.log_prob)
    new_grad_log_prob = jnp.where(accept[:, None], grad_lp, state.grad_log_prob)
    new_accept_count = state.accept_count + accept.astype(jnp.int32)

    new_state = HMCState(new_position, new_log_prob, new_grad_log_prob, new_accept_count)

    if return_proposal:
        return key, new_state, current_position, log_prob, delta_H
    else:
        return key, new_state


@partial(jit, static_argnames=("log_prob_fn", "max_steps", "num_samples", "burn_in", "track_proposals"))
def hmc_run_dynamic(
    key: Array,
    log_prob_fn: LogProbFn,
    init_position: Array,
    step_size: float,
    actual_steps: Array,  # Continuous, differentiable
    max_steps: int,  # Static maximum
    num_samples: int,
    burn_in: int = 0,
    track_proposals: bool = False,
) -> Tuple:
    """Run HMC sampler with dynamic trajectory length.

    Args:
        key: JAX random key
        log_prob_fn: Function to compute log probability and gradient
        init_position: Initial positions with shape (n_dim,) or (n_chains, n_dim)
        step_size: Leapfrog integration step size
        actual_steps: Continuous number of steps (JAX array, differentiable)
        max_steps: Maximum number of steps (static, for JIT)
        num_samples: Number of samples to collect (after burn-in)
        burn_in: Number of burn-in iterations (default: 0)
        track_proposals: If True, return proposal tracking info for diagnostics (default: False)

    Returns:
        If track_proposals=False:
            Tuple of (samples, log_probs, accept_rate, final_state)
        If track_proposals=True:
            Tuple of (samples, log_probs, accept_rate, final_state,
                      pre_positions, pre_lps, prop_positions, prop_lps, deltas_H)
    """
    init_state = hmc_init(init_position, log_prob_fn)
    n_chains, n_dim = init_state.position.shape
    state = init_state
    step_size_arr = jnp.asarray(step_size, dtype=init_state.position.dtype)

    # Burn-in phase
    if burn_in > 0:
        def burn_body(carry, _):
            k, s = carry
            k, s = hmc_step_dynamic(s, log_prob_fn, step_size_arr, actual_steps, max_steps, k, return_proposal=False)
            return (k, s), None

        (key, init_state), _ = lax.scan(burn_body, (key, init_state), length=burn_in)

        # Reset accept counter after burn-in
        state = HMCState(
            position=init_state.position,
            log_prob=init_state.log_prob,
            grad_log_prob=init_state.grad_log_prob,
            accept_count=jnp.zeros(n_chains, dtype=jnp.int32),
        )
    else:
        state = init_state

    # Sampling phase
    if track_proposals:
        def sample_with_proposals(carry, _):
            k, s = carry
            pre_pos, pre_lp = s.position, s.log_prob
            k, s, prop_pos, prop_lp, delta_H = hmc_step_dynamic(
                s, log_prob_fn, step_size_arr, actual_steps, max_steps, k, return_proposal=True
            )
            return (k, s), (pre_pos, pre_lp, prop_pos, prop_lp, delta_H, s.position, s.log_prob)

        (key, state), (pre_positions, pre_lps, prop_positions, prop_lps, deltas_H, post_positions, post_lps) = lax.scan(
            sample_with_proposals, (key, state), length=num_samples
        )
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return (
            post_positions, post_lps,
            accept_rate, state,
            pre_positions, pre_lps,
            prop_positions, prop_lps, deltas_H
        )
    else:
        def sample_body(carry, _):
            k, s = carry
            k, s = hmc_step_dynamic(s, log_prob_fn, step_size_arr, actual_steps, max_steps, k, return_proposal=False)
            return (k, s), (s.position, s.log_prob)

        (key, state), (samples, lps) = lax.scan(sample_body, (key, state), length=num_samples)
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return samples, lps, accept_rate, state


@partial(jit, static_argnames=("log_prob_fn", "num_steps", "num_samples", "burn_in", "track_proposals"))
def hmc_run(
    key: Array,
    log_prob_fn: LogProbFn,
    init_position: Array,
    step_size: float,
    num_steps: int,
    num_samples: int,
    burn_in: int = 0,
    track_proposals: bool = False,
) -> Tuple:
    """Run Hamiltonian Monte Carlo sampler with parallel chains.

    Args:
        key: JAX random key
        log_prob_fn: Function to compute log probability and gradient
        init_position: Initial positions with shape (n_dim,) or (n_chains, n_dim)
        step_size: Leapfrog integration step size
        num_steps: Number of leapfrog steps per HMC iteration
        num_samples: Number of samples to collect (after burn-in)
        burn_in: Number of burn-in iterations (default: 0)
        track_proposals: If True, return proposal tracking info for diagnostics (default: False)

    Returns:
        If track_proposals=False:
            Tuple of (samples, log_probs, accept_rate, final_state) where:
            - samples: Array of shape (num_samples, n_chains, n_dim)
            - log_probs: Array of shape (num_samples, n_chains)
            - accept_rate: Acceptance rate per chain, shape (n_chains,)
            - final_state: Final HMCState after sampling
        If track_proposals=True:
            Tuple of (samples, log_probs, accept_rate, final_state,
                      pre_positions, pre_lps, prop_positions, prop_lps, deltas_H)
            with additional proposal tracking arrays
    """
    init_state = hmc_init(init_position, log_prob_fn)
    n_chains, n_dim = init_state.position.shape
    state = init_state
    step_size_arr = jnp.asarray(step_size, dtype=init_state.position.dtype)
    
    # Burn-in phase
    if burn_in > 0:
        def burn_body(carry, _):
            k, s = carry
            k, s = hmc_step(s, log_prob_fn, step_size_arr, num_steps, k, return_proposal=False)
            return (k, s), None

        (key, init_state), _ = lax.scan(burn_body, (key, init_state), length=burn_in)

        # Reset accept counter after burn-in
        state = HMCState(
            position=init_state.position,
            log_prob=init_state.log_prob,
            grad_log_prob=init_state.grad_log_prob,
            accept_count=jnp.zeros(n_chains, dtype=jnp.int32),
        )
    else:
        state = init_state

    # Sampling phase
    if track_proposals:
        def sample_with_proposals(carry, _):
            k, s = carry
            pre_pos, pre_lp = s.position, s.log_prob
            k, s, prop_pos, prop_lp, delta_H = hmc_step(
                s, log_prob_fn, step_size_arr, num_steps, k, return_proposal=True
            )
            return (k, s), (pre_pos, pre_lp, prop_pos, prop_lp, delta_H, s.position, s.log_prob)

        (key, state), (pre_positions, pre_lps, prop_positions, prop_lps, deltas_H, post_positions, post_lps) = lax.scan(
            sample_with_proposals, (key, state), length=num_samples
        )
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return (
            post_positions, post_lps,
            accept_rate, state,
            pre_positions, pre_lps,
            prop_positions, prop_lps, deltas_H
        )
    else:
        def sample_body(carry, _):
            k, s = carry
            k, s = hmc_step(s, log_prob_fn, step_size_arr, num_steps, k, return_proposal=False)
            return (k, s), (s.position, s.log_prob)

        (key, state), (samples, lps) = lax.scan(sample_body, (key, state), length=num_samples)
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return samples, lps, accept_rate, state

