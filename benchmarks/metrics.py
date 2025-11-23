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
    projection_batch_size: int = 100,
) -> float:
    """
    Compute the Sliced Wasserstein-2 distance between two sample sets.
    Projects both distributions onto random 1D directions and computes
    the average 1D Wasserstein distance across projections.
    
    Uses batched computation for memory efficiency with large sample sets.
    
    Args:
        samples1: First sample set, shape (n1, dim)
        samples2: Second sample set, shape (n2, dim)
        n_projections: Number of random 1D projections
        key: JAX random key (uses fixed seed if None)
        projection_batch_size: Number of projections to process simultaneously.
            Reduce if encountering OOM errors. Default: 100
    
    Returns:
        Sliced W2 distance (scalar)
    
    Note:
        For very large sample sets (>50k samples), consider reducing 
        projection_batch_size (e.g., 50 or 25) if memory issues occur.
    """
    if key is None:
        key = random.PRNGKey(30)
    
    n1, dim = samples1.shape
    n2, _ = samples2.shape
    
    # Process projections in batches
    n_batches = (n_projections + projection_batch_size - 1) // projection_batch_size
    w2_distances = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * projection_batch_size
        end_idx = min(start_idx + projection_batch_size, n_projections)
        current_batch_size = end_idx - start_idx
        
        # Generate random unit vectors for this batch
        batch_key = random.fold_in(key, batch_idx)
        directions = random.normal(batch_key, (current_batch_size, dim))
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
        
        # Project samples onto each direction
        proj1 = samples1 @ directions.T
        proj2 = samples2 @ directions.T
        
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
        
        # W2 for each 1D projection in this batch
        w2_batch = jnp.sqrt(jnp.mean((proj1_quantiles - proj2_quantiles) ** 2, axis=0))
        w2_distances.append(w2_batch)
    
    all_w2 = jnp.concatenate(w2_distances)
    return float(jnp.mean(all_w2))


def compute_sliced_w2(
    samples: jnp.ndarray,
    target_name: str,
    dim: int,
    n_reference: int = 50000,
    n_projections: int = 500,
    projection_batch_size: int = 100,
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
        projection_batch_size: Number of projections per batch (adjust for memory)
        key: JAX random key
        **target_kwargs: Additional arguments for the target (e.g., correlation, df)
    
    Returns:
        Sliced W2 distance, or None if no reference sampler available
    """
    if key is None:
        key = random.PRNGKey(123)
    
    ref_sampler = get_reference_sampler(target_name, dim, **target_kwargs)
    if ref_sampler is None:
        return None
    
    # Flatten samples
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
        projection_batch_size=projection_batch_size,
        key=subkey,
    )