"""
Target distributions for MCMC benchmarking.

This module provides a collection of target distributions with varying challenges
for testing MCMC samplers. Each target is designed to stress different aspects
of sampling performance.

Each target distribution provides:
- log_prob_fn: Function computing log p(x)
- dim: Dimensionality
- true_mean: Known true mean (for validation)
- true_cov: Known true covariance (for validation, None if not tractable)
- name: Descriptive name
- description: What the target tests
- init_sampler: Optional custom initialization function
"""

from typing import Callable, NamedTuple, Optional
import jax.numpy as jnp
import jax.random as random


class TargetDistribution(NamedTuple):
    """Container for target distribution specification."""
    log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray]
    dim: int
    true_mean: jnp.ndarray
    true_cov: Optional[jnp.ndarray]
    name: str
    description: str
    init_sampler: Optional[Callable] = None  # (key, n_chains) -> positions


# ============================================================================
# Target Distribution Factories
# ============================================================================

def standard_normal(dim: int = 10) -> TargetDistribution:
    """
    Standard normal distribution N(0, I) in dim dimensions.

    Tests: Basic sampler correctness, well-conditioned target.

    Args:
        dim: Dimensionality of the distribution.

    Returns:
        TargetDistribution with standard normal log probability.
    """
    def log_prob_fn(x):
        D = x.shape[-1]
        return -0.5 * (jnp.sum(x**2, axis=-1) + D * jnp.log(2.0 * jnp.pi))

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=jnp.zeros(dim),
        true_cov=jnp.eye(dim),
        name=f"StandardNormal{dim}D",
        description=f"{dim}D standard normal N(0, I) - tests basic correctness"
    )


def correlated_gaussian(dim: int = 10, correlation: float = 0.9) -> TargetDistribution:
    """
    Gaussian with compound symmetry covariance: Σ_ij = ρ if i≠j, 1 if i=j.

    Tests: High correlation between variables, requires good momentum in HMC.

    Args:
        dim: Dimensionality of the distribution.
        correlation: Off-diagonal correlation coefficient (0 < ρ < 1).

    Returns:
        TargetDistribution with correlated Gaussian log probability.
    """
    # Build covariance matrix: Σ = (1-ρ)I + ρ J where J is all-ones matrix
    # This is a valid covariance if -1/(dim-1) < ρ < 1
    cov = (1.0 - correlation) * jnp.eye(dim) + correlation * jnp.ones((dim, dim))

    # For compound symmetry, inverse has closed form:
    # Σ^{-1} = (1/(1-ρ)) * I - (ρ/((1-ρ)(1+(dim-1)ρ))) * J
    a = 1.0 / (1.0 - correlation)
    b = -correlation / ((1.0 - correlation) * (1.0 + (dim - 1) * correlation))
    cov_inv = a * jnp.eye(dim) + b * jnp.ones((dim, dim))

    # Log determinant: log|Σ| = (dim-1)*log(1-ρ) + log(1+(dim-1)ρ)
    log_det_cov = (dim - 1) * jnp.log(1.0 - correlation) + jnp.log(1.0 + (dim - 1) * correlation)

    def log_prob_fn(x):
        D = x.shape[-1]
        # -0.5 * (x^T Σ^{-1} x + log|Σ| + D log(2π))
        if x.ndim == 1:
            mahalanobis = x @ cov_inv @ x
        else:
            # Batched: (n_chains, dim) @ (dim, dim) @ (dim, n_chains)
            mahalanobis = jnp.sum((x @ cov_inv) * x, axis=-1)
        return -0.5 * (mahalanobis + log_det_cov + D * jnp.log(2.0 * jnp.pi))

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=jnp.zeros(dim),
        true_cov=cov,
        name=f"CorrelatedGaussian{dim}D_rho{correlation}",
        description=f"{dim}D Gaussian with correlation rho={correlation} - tests handling of correlation"
    )


def ill_conditioned_gaussian(dim: int = 10, condition_number: float = 100.0) -> TargetDistribution:
    """
    Gaussian with diagonal covariance having specified condition number.

    Eigenvalues are linearly spaced from 1 to condition_number.
    Tests: Step size tuning, sensitivity to ill-conditioning.

    Args:
        dim: Dimensionality of the distribution.
        condition_number: Ratio of largest to smallest eigenvalue (κ > 1).

    Returns:
        TargetDistribution with ill-conditioned Gaussian log probability.
    """
    # Create covariance with eigenvalues from 1 to condition_number
    eigenvalues = jnp.linspace(1.0, condition_number, dim)
    cov = jnp.diag(eigenvalues)
    cov_inv = jnp.diag(1.0 / eigenvalues)
    log_det_cov = jnp.sum(jnp.log(eigenvalues))

    def log_prob_fn(x):
        D = x.shape[-1]
        # -0.5 * (x^T Σ^{-1} x + log|Σ| + D log(2π))
        if x.ndim == 1:
            mahalanobis = x @ cov_inv @ x
        else:
            mahalanobis = jnp.sum((x @ cov_inv) * x, axis=-1)
        return -0.5 * (mahalanobis + log_det_cov + D * jnp.log(2.0 * jnp.pi))

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=jnp.zeros(dim),
        true_cov=cov,
        name=f"IllConditioned{dim}D_kappa{int(condition_number)}",
        description=f"{dim}D Gaussian with kappa={condition_number} - tests ill-conditioning"
    )


def neals_funnel(dim: int = 10) -> TargetDistribution:
    """
    Neal's funnel distribution: challenging hierarchical model.

    Structure:
        x[0] ~ N(0, 3)           # "neck" variable
        x[i] ~ N(0, exp(x[0]))   for i > 0

    Tests: Varying curvature, requires dynamic step size adaptation.
    The funnel has exponentially varying scale across the space.

    Args:
        dim: Dimensionality (recommended: 10-20).

    Returns:
        TargetDistribution with Neal's funnel log probability.
    """
    def log_prob_fn(x):
        # x can be (dim,) or (n_chains, dim)
        if x.ndim == 1:
            x0 = x[0]
            x_rest = x[1:]
            D_rest = dim - 1
        else:
            x0 = x[:, 0]
            x_rest = x[:, 1:]
            D_rest = dim - 1

        # log p(x0) = -0.5 * (x0^2 / 9 + log(2π * 9))
        log_p_x0 = -0.5 * (x0**2 / 9.0 + jnp.log(2.0 * jnp.pi * 9.0))

        # log p(x_rest | x0) = -0.5 * (sum(x_rest^2) / exp(x0) + D_rest * x0 + D_rest * log(2π))
        variance = jnp.exp(x0)
        sum_sq = jnp.sum(x_rest**2, axis=-1) if x.ndim > 1 else jnp.sum(x_rest**2)
        log_p_rest = -0.5 * (sum_sq / variance + D_rest * x0 + D_rest * jnp.log(2.0 * jnp.pi))

        return log_p_x0 + log_p_rest

    def init_sampler(key, n_chains):
        """Custom initialization: sample from prior."""
        key1, key2 = random.split(key)
        x0 = random.normal(key1, (n_chains, 1)) * 3.0  # N(0, 9)
        # Initialize x_rest with moderate variance (exp(0) = 1) to avoid extreme values
        x_rest = random.normal(key2, (n_chains, dim - 1))
        return jnp.concatenate([x0, x_rest], axis=1)

    # True moments are tractable:
    # E[x0] = 0, Var[x0] = 9
    # E[x_i | x0] = 0, E[x_i] = 0  for i > 0
    # Var[x_i] = E[exp(x0)] = exp(E[x0] + Var[x0]/2) = exp(4.5) (log-normal moment)
    var_rest = jnp.exp(4.5)

    true_mean = jnp.zeros(dim)
    true_cov_diag = jnp.concatenate([jnp.array([9.0]), jnp.ones(dim - 1) * var_rest])
    true_cov = jnp.diag(true_cov_diag)

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=true_mean,
        true_cov=true_cov,
        name=f"NealsFunnel{dim}D",
        description=f"{dim}D Neal's funnel - tests varying curvature and scale",
        init_sampler=init_sampler
    )


def rosenbrock(dim: int = 10, scale: float = 0.1) -> TargetDistribution:
    """
    Rosenbrock "banana" distribution: strongly curved, banana-shaped density.

    Structure (following Haario et al. 2001):
        log p(x) = -sum_{i=1}^{dim-1} [(x[i+1] - x[i]^2)^2 / (2*scale^2) + (x[i] - 1)^2 / 2]

    Tests: Curved geometry, strong nonlinear correlations.

    Args:
        dim: Dimensionality (recommended: 10-20).
        scale: Scale parameter controlling curvature (smaller = more curved).

    Returns:
        TargetDistribution with Rosenbrock log probability.
    """
    def log_prob_fn(x):
        # x can be (dim,) or (n_chains, dim)
        if x.ndim == 1:
            x_curr = x[:-1]
            x_next = x[1:]
        else:
            x_curr = x[:, :-1]
            x_next = x[:, 1:]

        # Banana term: (x_{i+1} - x_i^2)^2 / (2 * scale^2)
        banana_term = jnp.sum((x_next - x_curr**2)**2, axis=-1) / (2.0 * scale**2)

        # Centering term: (x_i - 1)^2 / 2
        centering_term = jnp.sum((x_curr - 1.0)**2, axis=-1) / 2.0

        return -(banana_term + centering_term)

    def init_sampler(key, n_chains):
        """Initialize near mode at (1, 1, ..., 1)."""
        return random.normal(key, (n_chains, dim)) * 0.5 + 1.0

    # True moments not analytically tractable - set to None
    # Empirically, mode is near (1, 1, ..., 1)
    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=None,  # Not analytically tractable
        true_cov=None,   # Not analytically tractable
        name=f"Rosenbrock{dim}D_scale{scale}",
        description=f"{dim}D Rosenbrock (banana) - tests curved geometry",
        init_sampler=init_sampler
    )


# ============================================================================
# Convenience Functions
# ============================================================================

def get_target(name: str, dim: int = 10, **kwargs) -> TargetDistribution:
    """
    Get a target distribution by name.

    Args:
        name: One of ['standard_normal', 'correlated_gaussian',
                      'ill_conditioned_gaussian', 'neals_funnel', 'rosenbrock']
        dim: Dimensionality for the target.
        **kwargs: Additional arguments passed to target factory.

    Returns:
        TargetDistribution object.
    """
    targets = {
        'standard_normal': standard_normal,
        'correlated_gaussian': correlated_gaussian,
        'ill_conditioned_gaussian': ill_conditioned_gaussian,
        'neals_funnel': neals_funnel,
        'rosenbrock': rosenbrock,
    }

    if name not in targets:
        raise ValueError(f"Unknown target '{name}'. Available: {list(targets.keys())}")

    return targets[name](dim=dim, **kwargs)


def list_targets():
    """Print available target distributions with descriptions."""
    targets = [
        standard_normal(10),
        correlated_gaussian(10),
        ill_conditioned_gaussian(10),
        neals_funnel(10),
        rosenbrock(10),
    ]

    print("Available Target Distributions:")
    print("=" * 80)
    for target in targets:
        print(f"\n{target.name}")
        print(f"  {target.description}")
        print(f"  Dimension: {target.dim}")
        print(f"  True mean: {'Available' if target.true_mean is not None else 'Not tractable'}")
        print(f"  True cov: {'Available' if target.true_cov is not None else 'Not tractable'}")
        print(f"  Custom init: {'Yes' if target.init_sampler is not None else 'No'}")
