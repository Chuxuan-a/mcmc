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


def log_gamma(dim: int = 10, shape: float = 2.0, rate: float = 1.0) -> TargetDistribution:
    """
    Independent log-Gamma distribution (each dimension is Gamma distributed).

    Structure:
        x_i ~ Gamma(shape, rate) for all i
        log p(x) = sum_i [(shape-1)*log(x_i) - rate*x_i - log(Gamma(shape)) - shape*log(rate)]

    Tests: Heavy tails, asymmetry, positivity constraints.

    Args:
        dim: Dimensionality
        shape: Shape parameter (alpha > 0)
        rate: Rate parameter (beta > 0)

    Returns:
        TargetDistribution with log-Gamma log probability.
    """
    from jax.scipy.special import gammaln

    def log_prob_fn(x):
        # Check positivity constraint
        if x.ndim == 1:
            valid = jnp.all(x > 0)
            sum_axis = None
        else:
            valid = jnp.all(x > 0, axis=-1)
            sum_axis = -1

        # Gamma log pdf: (shape-1)*log(x) - rate*x - log(Gamma(shape)) - shape*log(rate)
        log_normalizer = gammaln(shape) + shape * jnp.log(rate)
        log_pdf = (shape - 1.0) * jnp.log(jnp.maximum(x, 1e-10)) - rate * x - log_normalizer
        result = jnp.sum(log_pdf, axis=sum_axis) if sum_axis is not None else jnp.sum(log_pdf)

        # Return -inf for invalid (negative) values
        return jnp.where(valid, result, -jnp.inf)

    def init_sampler(key, n_chains):
        """Initialize from prior (Gamma distribution)."""
        return random.gamma(key, shape, (n_chains, dim)) / rate

    # True moments for Gamma distribution
    true_mean = jnp.ones(dim) * (shape / rate)
    true_var = shape / (rate ** 2)
    true_cov = jnp.eye(dim) * true_var

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=true_mean,
        true_cov=true_cov,
        name=f"LogGamma{dim}D_shape{shape}_rate{rate}",
        description=f"{dim}D independent Gamma - tests heavy tails and asymmetry",
        init_sampler=init_sampler
    )


def student_t(dim: int = 10, df: float = 3.0) -> TargetDistribution:
    """
    Independent Student-t distribution with specified degrees of freedom.

    Structure:
        x_i ~ Student-t(df) for all i
        log p(x) = sum_i [log(Gamma((df+1)/2)) - log(Gamma(df/2)) - 0.5*log(df*pi)
                         - ((df+1)/2)*log(1 + x_i^2/df)]

    Tests: Heavy tails, non-Gaussian geometry, robustness to outliers.
    The heavy tails make this challenging for Gaussian-based proposals.

    Why df=3:
    - df=3 has finite variance (Var = df/(df-2) = 3) but infinite 4th moment
    - Creates significant heavy tails while remaining tractable
    - HMC/NUTS struggle with heavy tails (overshooting into low-density regions)
    - GRAHMC's friction can help control momentum in the tails

    Args:
        dim: Dimensionality
        df: Degrees of freedom (nu > 2 for finite variance, nu=3 recommended)

    Returns:
        TargetDistribution with Student-t log probability.
    """
    from jax.scipy.special import gammaln

    def log_prob_fn(x):
        # Student-t log pdf: log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(νπ) - ((ν+1)/2)*log(1 + x²/ν)
        log_normalizer = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * jnp.log(df * jnp.pi)

        if x.ndim == 1:
            log_kernel = -((df + 1.0) / 2.0) * jnp.log(1.0 + x**2 / df)
            result = jnp.sum(log_normalizer + log_kernel)
        else:
            log_kernel = -((df + 1.0) / 2.0) * jnp.log(1.0 + x**2 / df)
            result = jnp.sum(log_normalizer + log_kernel, axis=-1)

        return result

    def init_sampler(key, n_chains):
        """Initialize with overdispersed samples to cover heavy tails."""
        # Use wider initialization (std=2) to better cover the heavy-tailed distribution
        return random.normal(key, (n_chains, dim)) * 2.0

    # True moments for Student-t with df > 2
    true_mean = jnp.zeros(dim)
    if df > 2:
        true_var = df / (df - 2.0)  # For df=3: var = 3
        true_cov = jnp.eye(dim) * true_var
    else:
        true_cov = None  # Variance doesn't exist for df <= 2

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=true_mean,
        true_cov=true_cov,
        name=f"StudentT{dim}D_df{df}",
        description=f"{dim}D independent Student-t(df={df}) - tests heavy tails and non-Gaussian geometry",
        init_sampler=init_sampler
    )


def rosenbrock(dim: int = 10, scale: float = 0.1) -> TargetDistribution:
    """
    Rosenbrock density: challenging non-Gaussian with curved valley.

    Structure (negative Rosenbrock as potential):
        U(x) = sum_{i=1}^{D-1} [(1-x_i)^2 + (1/scale^2)*(x_{i+1} - x_i^2)^2]
        log p(x) = -U(x)

    Tests: Curved ridges, non-linear correlations, momentum direction adaptation.
    The narrow curved valley requires precise momentum control.

    Why scale=0.1:
    - Creates a narrow curved valley (smaller scale = narrower valley)
    - Curvature is challenging for fixed mass matrix adaptation
    - The optimal path is x_i = x_{i-1}^2, a parabola
    - HMC/NUTS with mass matrix can learn the width but not the curvature
    - GRAHMC's time-varying friction can better navigate the curved geometry

    Technical notes:
    - Minimum at x* = (1, 1, ..., 1) with log p(x*) = 0
    - For scale=0.1, typical width ~ 0.1 perpendicular to valley
    - Valley curvature creates position-dependent geometry (similar to funnel)

    Args:
        dim: Dimensionality (recommended: 5-20)
        scale: Valley width parameter (smaller = narrower, harder)

    Returns:
        TargetDistribution with Rosenbrock log probability.
    """
    def log_prob_fn(x):
        # Rosenbrock function as negative log probability
        # U(x) = sum [(1-x_i)^2 + a*(x_{i+1} - x_i^2)^2] where a = 1/scale^2
        a = 1.0 / (scale ** 2)

        if x.ndim == 1:
            x_current = x[:-1]
            x_next = x[1:]
            term1 = (1.0 - x_current) ** 2
            term2 = a * (x_next - x_current ** 2) ** 2
            U = jnp.sum(term1 + term2)
        else:
            # Batched: (n_chains, dim)
            x_current = x[:, :-1]
            x_next = x[:, 1:]
            term1 = (1.0 - x_current) ** 2
            term2 = a * (x_next - x_current ** 2) ** 2
            U = jnp.sum(term1 + term2, axis=-1)

        # Return negative potential as log probability
        return -U

    def init_sampler(key, n_chains):
        """Initialize near the mode (1, 1, ..., 1) with small perturbations."""
        # Start near the global minimum x* = (1, 1, ..., 1)
        # Add small noise to explore the valley
        return jnp.ones((n_chains, dim)) + random.normal(key, (n_chains, dim)) * 0.5

    # True moments are not analytically tractable (unnormalized density)
    # But we know the mode is at (1, 1, ..., 1)
    true_mean = jnp.ones(dim)  # Approximate (mode as proxy)
    true_cov = None  # Not analytically available

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=true_mean,
        true_cov=true_cov,
        name=f"Rosenbrock{dim}D_scale{scale}",
        description=f"{dim}D Rosenbrock(scale={scale}) - tests curved valleys and non-linear geometry",
        init_sampler=init_sampler
    )


def gaussian_mixture(dim: int = 10, n_modes: int = 2, separation: float = 5.0) -> TargetDistribution:
    """
    Mixture of Gaussians in 1D with remaining dimensions independent.

    Structure (simplified for tractability):
        x[0] ~ 0.5 * N(-separation/2, 1) + 0.5 * N(+separation/2, 1)
        x[i] ~ N(0, 1) for i > 0

    Tests: Multimodality, mode-switching ability.

    Args:
        dim: Dimensionality (only first dimension is mixture)
        n_modes: Number of modes (currently only 2 supported)
        separation: Distance between modes

    Returns:
        TargetDistribution with Gaussian mixture log probability.
    """
    if n_modes != 2:
        raise NotImplementedError("Only 2-mode mixture currently supported")

    def log_prob_fn(x):
        # x can be (dim,) or (n_chains, dim)
        if x.ndim == 1:
            x0 = x[0]
            x_rest = x[1:]
        else:
            x0 = x[:, 0]
            x_rest = x[:, 1:]

        # Mixture component in first dimension: 0.5*N(-sep/2, 1) + 0.5*N(+sep/2, 1)
        mode1 = -0.5 * (x0 + separation / 2.0) ** 2
        mode2 = -0.5 * (x0 - separation / 2.0) ** 2
        # log-sum-exp trick for stability
        max_val = jnp.maximum(mode1, mode2)
        log_p_x0 = jnp.log(0.5) + max_val + jnp.log(jnp.exp(mode1 - max_val) + jnp.exp(mode2 - max_val)) - 0.5 * jnp.log(2.0 * jnp.pi)

        # Remaining dimensions are standard normal
        if x.ndim == 1:
            log_p_rest = -0.5 * (jnp.sum(x_rest ** 2) + (dim - 1) * jnp.log(2.0 * jnp.pi))
        else:
            log_p_rest = -0.5 * (jnp.sum(x_rest ** 2, axis=-1) + (dim - 1) * jnp.log(2.0 * jnp.pi))

        return log_p_x0 + log_p_rest

    def init_sampler(key, n_chains):
        """Initialize with chains split between the two modes."""
        key1, key2 = random.split(key)
        # Half chains near mode 1, half near mode 2
        n_half = n_chains // 2
        x0_mode1 = random.normal(key1, (n_half,)) - separation / 2.0
        x0_mode2 = random.normal(key1, (n_chains - n_half,)) + separation / 2.0
        x0 = jnp.concatenate([x0_mode1, x0_mode2])[:, None]
        x_rest = random.normal(key2, (n_chains, dim - 1))
        return jnp.concatenate([x0, x_rest], axis=1)

    # True mean is 0 for all dimensions (symmetric mixture)
    # True variance: x0 has var = 1 + (separation/2)^2
    var_x0 = 1.0 + (separation / 2.0) ** 2
    true_mean = jnp.zeros(dim)
    true_cov_diag = jnp.concatenate([jnp.array([var_x0]), jnp.ones(dim - 1)])
    true_cov = jnp.diag(true_cov_diag)

    return TargetDistribution(
        log_prob_fn=log_prob_fn,
        dim=dim,
        true_mean=true_mean,
        true_cov=true_cov,
        name=f"GaussianMixture{dim}D_modes{n_modes}_sep{separation}",
        description=f"{dim}D Gaussian mixture (x[0] bimodal) - tests mode-switching",
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
                      'ill_conditioned_gaussian', 'student_t', 'log_gamma',
                      'rosenbrock', 'neals_funnel', 'gaussian_mixture']
        dim: Dimensionality for the target.
        **kwargs: Additional arguments passed to target factory.

    Returns:
        TargetDistribution object.
    """
    targets = {
        'standard_normal': standard_normal,
        'correlated_gaussian': correlated_gaussian,
        'ill_conditioned_gaussian': ill_conditioned_gaussian,
        'student_t': student_t,
        'log_gamma': log_gamma,
        'rosenbrock': rosenbrock,
        'neals_funnel': neals_funnel,
        'gaussian_mixture': gaussian_mixture,
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
        student_t(10),
        log_gamma(10),
        rosenbrock(10),
        neals_funnel(10),
        gaussian_mixture(10),
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
