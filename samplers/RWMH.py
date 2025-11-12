"""Random Walk Metropolis-Hastings (RWMH) sampler implementation.

This module implements the Random Walk Metropolis-Hastings algorithm with:
- Parallel chain execution via JAX vmap
- Gaussian proposal distribution
- Configurable proposal scale parameter
- Burn-in support with acceptance counter reset
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Tuple, NamedTuple

import jax.numpy as jnp
from jax import jit, random, vmap, lax

# Type aliases
Array = jnp.ndarray
LogProbFn = Callable[[Array], Array]  # Maps x -> log p(x)

class RWMState(NamedTuple):
    """State for Random Walk Metropolis-Hastings sampler.

    Attributes:
        position: Current positions, shape (n_chains, n_dim)
        log_prob: Log probabilities at current positions, shape (n_chains,) [float64]
        accept_count: Number of accepted proposals per chain, shape (n_chains,) [int32]
    """
    position: Array
    log_prob: Array
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


def rwMH_init(init_position: Array, log_prob_fn: LogProbFn) -> RWMState:
    """Initialize state for Random Walk Metropolis-Hastings sampler.

    Args:
        init_position: Initial positions with shape (n_dim,) or (n_chains, n_dim)
        log_prob_fn: Function mapping positions to log probabilities

    Returns:
        Initial RWMState with positions, log probabilities, and zero accept counts
    """
    pos, _ = _ensure_batched(init_position)
    n_chains = pos.shape[0]
    log_prob = vmap(log_prob_fn)(pos).astype(jnp.float64)  # Use float64 for stability
    accept_count = jnp.zeros(n_chains, dtype=jnp.int32)
    return RWMState(position=pos, log_prob=log_prob, accept_count=accept_count)


@partial(jit, static_argnames=("log_prob_fn",))
def rwMH_step(
    state: RWMState,
    log_prob_fn: LogProbFn,
    scale: float,
    key: Array,
) -> Tuple[Array, RWMState]:
    """Perform one step of Random Walk Metropolis-Hastings.

    Args:
        state: Current sampler state
        log_prob_fn: Function mapping positions to log probabilities
        scale: Proposal standard deviation (scale parameter for Gaussian proposal)
        key: JAX random key

    Returns:
        Tuple of (next_key, new_state) where next_key is the updated random key
        and new_state contains updated positions, log probs, and accept counts
    """
    n_chains, n_dim = state.position.shape
    key, key_noise, key_accept = random.split(key, 3)

    # Generate Gaussian proposals: x' = x + scale * eps, eps ~ N(0, I)
    pos_dtype = state.position.dtype
    scale_arr = jnp.asarray(scale, dtype=pos_dtype)
    noise = random.normal(key_noise, shape=(n_chains, n_dim), dtype=pos_dtype)
    proposal = state.position + scale_arr * noise

    # Evaluate log probability at proposals
    log_prob_dtype = state.log_prob.dtype
    proposal_log_prob = vmap(log_prob_fn)(proposal).astype(log_prob_dtype)

    # Metropolis-Hastings acceptance test: accept if log(u) < log(p'/p)
    log_ratio = proposal_log_prob - state.log_prob
    log_uniform = jnp.log(random.uniform(key_accept, shape=(n_chains,), dtype=log_prob_dtype))
    accept = log_uniform < jnp.minimum(0.0, log_ratio)

    # Update state (functionally - no mutation)
    new_position = jnp.where(accept[:, None], proposal, state.position)
    new_log_prob = jnp.where(accept, proposal_log_prob, state.log_prob)
    new_accept_count = state.accept_count + accept.astype(jnp.int32)

    new_state = RWMState(
        position=new_position,
        log_prob=new_log_prob,
        accept_count=new_accept_count,
    )
    return key, new_state


@partial(jit, static_argnames=("log_prob_fn", "num_samples", "burn_in"))
def rwMH_run(
    key: Array,
    log_prob_fn: LogProbFn,
    init_position: Array,
    num_samples: int,
    scale: float,
    burn_in: int = 0,
) -> Tuple[Array, Array, Array, RWMState]:
    """Run Random Walk Metropolis-Hastings sampler with parallel chains.

    Args:
        key: JAX random key
        log_prob_fn: Function mapping positions to log probabilities
        init_position: Initial positions with shape (n_dim,) or (n_chains, n_dim)
        num_samples: Number of samples to collect (after burn-in)
        scale: Proposal standard deviation
        burn_in: Number of burn-in iterations (default: 0)

    Returns:
        Tuple of (samples, log_probs, accept_rate, final_state) where:
        - samples: Array of shape (num_samples, n_chains, n_dim)
        - log_probs: Array of shape (num_samples, n_chains)
        - accept_rate: Acceptance rate per chain, shape (n_chains,)
        - final_state: Final RWMState after sampling
    """
    state = rwMH_init(init_position, log_prob_fn)
    n_chains, n_dim = state.position.shape

    # Burn-in phase
    if burn_in > 0:
        def burn_body(carry, _):
            k, st = carry
            k, st = rwMH_step(st, log_prob_fn, scale, k)
            return (k, st), None

        (key, state), _ = lax.scan(burn_body, (key, state), length=burn_in)

        # Reset accept counter after burn-in
        state = RWMState(
            position=state.position,
            log_prob=state.log_prob,
            accept_count=jnp.zeros(n_chains, dtype=jnp.int32),
        )

    # Sampling phase
    def sample_body(carry, _):
        k, st = carry
        k, st = rwMH_step(st, log_prob_fn, scale, k)
        return (k, st), (st.position, st.log_prob)

    (key, final_state), (samples, log_probs) = lax.scan(
        sample_body, (key, state), length=num_samples
    )

    accept_rate = final_state.accept_count.astype(jnp.float32) / num_samples
    return samples, log_probs, accept_rate, final_state
