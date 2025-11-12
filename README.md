# MCMC Samplers with Adaptive Friction

A JAX-based research codebase for developing and benchmarking MCMC samplers, with emphasis on Generalized Randomized Adaptive Hamiltonian Monte Carlo (GRAHMC) with various friction schedules.

## Features

- **Multiple MCMC Samplers**: Random Walk Metropolis-Hastings (RWMH), Hamiltonian Monte Carlo (HMC), No-U-Turn Sampler (NUTS), and Generalized RAHMC (GRAHMC)
- **Friction Schedules**: Five different friction schedule functions (constant, tanh, sigmoid, linear, sine) for time-varying friction
- **Automated Parameter Tuning**: Dual averaging (Hoffman & Gelman 2014) with coordinate-wise optimization for GRAHMC
- **Parallel Chain Execution**: Efficient batched sampling via JAX's `vmap()`
- **Comprehensive Diagnostics**: ESS, R-hat, acceptance rate tracking, and convergence detection

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: JAX, Optax, ArviZ, NumPy, Matplotlib, Seaborn

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

### Automated Parameter Tuning

```bash
# Tune HMC with grid search over trajectory lengths
python tuning.py --sampler hmc --dim 10 --plot

# Tune GRAHMC with coordinate-wise dual averaging
python tuning.py --sampler grahmc --schedule tanh --dim 10 --plot

# Results saved to ./tuning_output/ with diagnostic plots
```

## Project Structure

```
samplers/
├── GRAHMC.py       # Generalized RAHMC with friction schedules
├── RAHMC.py        # Basic RAHMC (constant friction)
├── HMC.py          # Standard Hamiltonian Monte Carlo
└── RWMH.py         # Random Walk Metropolis-Hastings

tuning.py           # Automated parameter optimization
run.ipynb           # Benchmarking and comparison notebook
```

## Key Design Principles

- **Position arrays**: Always batched as `(n_chains, n_dim)` for parallel execution
- **Numerical precision**: Float64 for log probabilities, float32 for positions
- **JIT compilation**: All sampling loops are JIT-compiled for performance
- **Proposal tracking**: Optional tracking of pre/post-proposal states for ESJD computation

## Friction Schedules

GRAHMC supports five friction schedules that control the transition from underdamped to overdamped dynamics:

- `constant`: Step function at trajectory midpoint
- `tanh`: Smooth hyperbolic tangent transition
- `sigmoid`: Sigmoid-based transition
- `linear`: Linear ramp from -γ to +γ
- `sine`: Sinusoidal variation

## Citation

If you use this code in your research, please cite the relevant papers on Randomized Adaptive HMC.
