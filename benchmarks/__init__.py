"""Benchmarking infrastructure for MCMC samplers."""

from .targets import (
    TargetDistribution,
    standard_normal,
    correlated_gaussian,
    ill_conditioned_gaussian,
    neals_funnel,
    log_gamma,
    student_t,
    rosenbrock,
    gaussian_mixture,
    get_target,
    list_targets,
    get_reference_sampler,
    has_reference_sampler,
)

from .metrics import (
    sliced_wasserstein_distance,
    compute_sliced_w2,
)

__all__ = [
    'TargetDistribution',
    'standard_normal',
    'correlated_gaussian',
    'ill_conditioned_gaussian',
    'neals_funnel',
    'log_gamma',
    'student_t',
    'rosenbrock',
    'gaussian_mixture',
    'get_target',
    'list_targets',
    'get_reference_sampler',
    'has_reference_sampler',
    'sliced_wasserstein_distance',
    'compute_sliced_w2',
]
