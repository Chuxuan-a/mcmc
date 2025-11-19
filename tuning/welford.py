"""Welford's online algorithm for estimating mean and diagonal variance.

This module provides JAX-compatible functions to compute running statistics
(mean and variance) of a stream of samples without storing them all in memory.
"""
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import jit, lax

class WelfordState(NamedTuple):
    """State for Welford's online estimator.
    
    Attributes:
        count: Number of samples seen so far (scalar).
        mean: Running mean of samples (n_dim,).
        m2: Sum of squared differences from the mean (n_dim,).
    """
    count: jnp.ndarray  # float64 scalar
    mean: jnp.ndarray   # (n_dim,)
    m2: jnp.ndarray     # (n_dim,)


def welford_init(n_dim: int, dtype=jnp.float64) -> WelfordState:
    """Initialize Welford estimator state.

    Args:
        n_dim: Dimensionality of the data.
        dtype: Data type for statistics (default: float64 for precision).
    
    Returns:
        Initial WelfordState with zeros.
    """
    return WelfordState(
        count=jnp.array(0.0, dtype=dtype),
        mean=jnp.zeros(n_dim, dtype=dtype),
        m2=jnp.zeros(n_dim, dtype=dtype),
    )


@jit
def welford_update(state: WelfordState, x: jnp.ndarray) -> WelfordState:
    """Update Welford state with a single new sample.

    Args:
        state: Current WelfordState.
        x: New sample with shape (n_dim,).

    Returns:
        Updated WelfordState.
    """
    x = x.astype(state.mean.dtype)
    count = state.count + 1.0
    delta = x - state.mean
    mean = state.mean + delta / count
    delta2 = x - mean
    m2 = state.m2 + delta * delta2
    
    return WelfordState(count, mean, m2)


@jit
def welford_update_batch(state: WelfordState, batch: jnp.ndarray) -> WelfordState:
    """Update Welford state with a batch of samples.
    
    Useful for processing output from multiple chains simultaneously.

    Args:
        state: Current WelfordState.
        batch: Batch of samples with shape (batch_size, n_dim).

    Returns:
        Updated WelfordState.
    """
    # Use scan to process the batch sequentially (statistically correct)
    # but compiled into a single efficient kernel.
    def body_fn(carry_state, sample):
        new_state = welford_update(carry_state, sample)
        return new_state, None

    final_state, _ = lax.scan(body_fn, state, batch)
    return final_state


@jit
def welford_covariance(state: WelfordState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Finalize and return the estimated mean and diagonal covariance.

    Args:
        state: Current WelfordState.

    Returns:
        Tuple of (mean, variance). 
        Variance is the diagonal of the covariance matrix.
    """
    # Standard sample variance: m2 / (n - 1)
    # Add epsilon to avoid division by zero if count <= 1
    n = jnp.maximum(state.count, 2.0)
    variance = state.m2 / (n - 1.0)
    return state.mean, variance