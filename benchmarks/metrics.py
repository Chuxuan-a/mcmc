"""
Sliced Wasserstein distance for MCMC sample quality evaluation.
"""

import jax.numpy as jnp
from jax import random
from typing import Optional

from benchmarks.targets import get_reference_sampler


def sliced_wasserstein_distance(
    samples1: jnp.ndarray,
    samples2: jnp.ndarray,
    n_projections: int = 500,
    key: Optional[jnp.ndarray] = None,
) -> float:
    """
    Compute the Sliced Wasserstein-2 distance between two sample sets.

    Projects both distributions onto random 1D directions and computes
    the average 1D Wasserstein distance across projections.

    Args:
        samples1: First sample set, shape (n1, dim)
        samples2: Second sample set, shape (n2, dim)
        n_projections: Number of random 1D projections
        key: JAX random key (uses fixed seed if None)

    Returns:
        Sliced W2 distance (scalar)
    """
    if key is None:
        key = random.PRNGKey(42)

    n1, dim = samples1.shape
    n2, _ = samples2.shape

    # Generate random unit vectors for projection
    directions = random.normal(key, (n_projections, dim))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    # Project samples onto each direction
    proj1 = samples1 @ directions.T  # (n1, n_projections)
    proj2 = samples2 @ directions.T  # (n2, n_projections)

    # Sort projections for 1D Wasserstein computation
    proj1_sorted = jnp.sort(proj1, axis=0)
    proj2_sorted = jnp.sort(proj2, axis=0)

    # Handle different sample sizes via quantile matching
    if n1 != n2:
        n_quantiles = min(n1, n2)
        quantile_indices = jnp.linspace(0, 1, n_quantiles)
        proj1_quantiles = jnp.quantile(proj1, quantile_indices, axis=0)
        proj2_quantiles = jnp.quantile(proj2, quantile_indices, axis=0)
    else:
        proj1_quantiles = proj1_sorted
        proj2_quantiles = proj2_sorted

    # W2 for each 1D projection
    w2_per_proj = jnp.sqrt(jnp.mean((proj1_quantiles - proj2_quantiles) ** 2, axis=0))

    return float(jnp.mean(w2_per_proj))


def compute_sliced_w2(
    samples: jnp.ndarray,
    target_name: str,
    dim: int,
    n_reference: int = 50000,
    n_projections: int = 500,
    key: Optional[jnp.ndarray] = None,
    **target_kwargs,
) -> Optional[float]:
    """
    Compute Sliced W2 between MCMC samples and ground truth.

    Args:
        samples: MCMC samples, shape (n_samples, n_chains, dim) or (n_samples, dim)
        target_name: Name of the target distribution
        dim: Dimensionality
        n_reference: Number of reference samples to generate
        n_projections: Number of projections for Sliced W2
        key: JAX random key
        **target_kwargs: Additional arguments for the target (e.g., correlation, df)

    Returns:
        Sliced W2 distance, or None if no reference sampler available (Rosenbrock)
    """
    if key is None:
        key = random.PRNGKey(123)

    # Get reference sampler
    ref_sampler = get_reference_sampler(target_name, dim, **target_kwargs)
    if ref_sampler is None:
        return None  # Rosenbrock has no direct sampler

    # Flatten samples: (n_samples, n_chains, dim) -> (n_total, dim)
    if samples.ndim == 3:
        flat_samples = samples.reshape(-1, dim)
    else:
        flat_samples = samples

    n_samples = flat_samples.shape[0]

    # Generate reference samples
    key, subkey = random.split(key)
    reference_samples = ref_sampler(subkey, n_reference)

    # Subsample MCMC samples if larger than reference
    if n_samples > n_reference:
        key, subkey = random.split(key)
        idx = random.choice(subkey, n_samples, (n_reference,), replace=False)
        flat_samples = flat_samples[idx]

    # Compute Sliced W2
    key, subkey = random.split(key)
    return sliced_wasserstein_distance(
        flat_samples,
        reference_samples,
        n_projections=n_projections,
        key=subkey,
    )
