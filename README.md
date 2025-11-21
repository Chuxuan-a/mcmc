# MCMC Samplers with Adaptive Friction

A JAX-based research codebase for developing and benchmarking MCMC samplers, with emphasis on Generalized Repelling-Attracting Hamiltonian Monte Carlo (GRAHMC) with various friction schedules.

## Features

- **Multiple MCMC Samplers**: Random Walk Metropolis-Hastings (RWMH), Hamiltonian Monte Carlo (HMC), No-U-Turn Sampler (NUTS), and Generalized RAHMC (GRAHMC)
- **Friction Schedules**: Five different friction schedule functions (constant, tanh, sigmoid, linear, sine) for time-varying friction dynamics
- **Automated Parameter Tuning**: Dual averaging (Hoffman & Gelman 2014) with coordinate-wise optimization for GRAHMC, Stan-style windowed adaptation with mass matrix learning
- **Diverse Target Distributions**: Standard normal, correlated Gaussian, ill-conditioned Gaussian, Neal's funnel, Student-t, Rosenbrock, and Gaussian mixtures
- **Parallel Chain Execution**: Efficient batched sampling via JAX's `vmap()`
- **Comprehensive Diagnostics**: ESS (bulk/tail), R-hat, acceptance rate tracking, z-score validation against known truth

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: JAX, Optax, ArviZ, NumPy, Matplotlib, Seaborn, Pandas

**Important**: Enable float64 precision for numerical stability:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Quick Start

### Basic Sampling

```python
import jax
import jax.numpy as jnp
from jax import random
from samplers.GRAHMC import rahmc_run, get_friction_schedule

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

# Define target distribution (10D standard normal)
def log_prob_fn(x):
    return -0.5 * jnp.sum(x**2, axis=-1)

# Run GRAHMC with tanh schedule
key = random.PRNGKey(0)
init_position = random.normal(key, shape=(4, 10)) * 2.0  # 4 chains, 10 dimensions

samples, log_probs, accept_rate, final_state = rahmc_run(
    key, log_prob_fn, init_position,
    num_samples=1000,
    burn_in=500,
    step_size=0.05,
    num_steps=16,
    gamma=1.0,
    steepness=5.0,
    friction_schedule=get_friction_schedule('tanh')
)

print(f"Acceptance rate: {jnp.mean(accept_rate):.3f}")
print(f"Samples shape: {samples.shape}")  # (1000, 4, 10)
```

### Testing Samplers

```bash
# Test samplers on various target distributions
python test_samplers.py --sampler hmc --target standard_normal --dim 10 --chains 4 --target-ess 1000
python test_samplers.py --sampler nuts --target ill_conditioned_gaussian --dim 10
python test_samplers.py --sampler grahmc --schedule tanh --target neals_funnel --dim 10
```

### Running Benchmarks

```bash
# Run comprehensive benchmarks across all sampler-target combinations
python run_benchmarks.py --output-dir results --dim 10

# Quick test mode
python run_benchmarks.py --quick

# Specific samplers and targets
python run_benchmarks.py --samplers hmc nuts grahmc --targets standard_normal ill_conditioned_gaussian
```

## Project Structure

```
.
├── samplers/                    # Core MCMC implementations
│   ├── RWMH.py                 # Random Walk Metropolis-Hastings
│   ├── HMC.py                  # Hamiltonian Monte Carlo
│   ├── NUTS.py                 # No-U-Turn Sampler
│   └── GRAHMC.py               # Generalized RAHMC with friction schedules
├── tuning/                      # Parameter tuning framework
│   ├── core.py                 # Main tuning orchestration
│   ├── dual_averaging.py       # Dual averaging algorithms
│   ├── adaptation.py           # Windowed adaptation with mass matrix
│   ├── welford.py              # Online mean/variance estimation
│   └── plots.py                # Diagnostic visualization
├── benchmarks/                  # Benchmarking infrastructure
│   └── targets.py              # Target distribution definitions
├── test_samplers.py            # Sampler testing with diagnostics
├── run_benchmarks.py           # Comprehensive benchmark suite
└── requirements.txt            # Python dependencies
```

## Samplers

### Random Walk Metropolis-Hastings (RWMH)
Basic random walk with Gaussian proposals. Target acceptance: **0.234** (Roberts & Rosenthal optimal).

### Hamiltonian Monte Carlo (HMC)
Leapfrog integration with configurable step size, trajectory length, and diagonal mass matrix. Target acceptance: **0.65**.

### No-U-Turn Sampler (NUTS)
Automatic trajectory length selection via iterative tree doubling with U-turn detection. JIT-compatible implementation using `lax.while_loop`.

### GRAHMC
Generalized Repelling-Attracting HMC with time-varying friction. The friction coefficient γ(t) transitions from negative (repelling/accelerating) to positive (attracting/damping) during each trajectory.

**Friction Schedules:**
| Schedule | Formula | Uses Steepness |
|----------|---------|----------------|
| `constant` | γ(t) = -γ if t<T/2 else +γ | No |
| `tanh` | γ(t) = γ_max × tanh(s×(2t/T - 1)) | Yes |
| `sigmoid` | γ(t) = γ_max × (2/(1+exp(-s×(t/T-0.5))) - 1) | Yes |
| `linear` | γ(t) = -γ + 2γ×t/T | No |
| `sine` | γ(t) = γ × sin(π×(t/T - 0.5)) | No |

## Target Distributions

| Target | Challenge | Key Properties |
|--------|-----------|----------------|
| `standard_normal` | Baseline | N(0, I), well-conditioned |
| `correlated_gaussian` | High correlation | Compound symmetry, ρ=0.9 |
| `ill_conditioned_gaussian` | Step size tuning | Condition number κ=100 |
| `neals_funnel` | Varying curvature | Hierarchical, exp-varying scale |
| `student_t` | Heavy tails | df=3, infinite 4th moment |
| `rosenbrock` | Curved valleys | Non-linear correlations |
| `gaussian_mixture` | Multimodality | Bimodal in first dimension |

## Tuning Framework

### Dual Averaging
Based on Hoffman & Gelman (2014), automatically adjusts parameters to achieve target acceptance rates:
- **RWMH**: Tunes proposal scale → 0.234 acceptance
- **HMC/NUTS**: Tunes step size → 0.65 acceptance
- **GRAHMC**: Coordinate-wise cycling through step_size, gamma, steepness

### Windowed Adaptation
Stan-style three-phase warmup:
1. **Fast init** (75 steps): Initial step size only
2. **Slow windows** (doubling: 25, 50, 100...): Learn diagonal mass matrix via Welford's algorithm
3. **Fast final** (50 steps): Final step size refinement

For GRAHMC, friction parameters are tuned **after** mass matrix learning to calibrate on the sphered geometry.

## Key Design Principles

- **Position arrays**: Always batched as `(n_chains, n_dim)` for parallel execution
- **Numerical precision**: Float64 for log probabilities and energy calculations
- **JIT compilation**: All sampling loops compiled for performance
- **Adaptive sampling**: Collect samples in batches until target ESS reached
- **Chain continuity**: Maintains state across batches for proper mixing

## Citation

If you use this code in your research, please cite the relevant papers on Randomized Adaptive HMC.
