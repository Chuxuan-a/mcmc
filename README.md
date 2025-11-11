# MCMC Samplers with Adaptive Friction

A JAX-based research codebase for developing and benchmarking MCMC samplers, with emphasis on Randomized Adaptive Hamiltonian Monte Carlo (RAHMC) with various friction schedules.

## Features

- **Multiple MCMC Samplers**: Random Walk Metropolis-Hastings (RWMH), Hamiltonian Monte Carlo (HMC), RAHMC, and Generalized RAHMC (GRAHMC)
- **Friction Schedules**: Five different friction schedule functions (constant, tanh, sigmoid, linear, sine) for adaptive dynamics
- **Automated Parameter Tuning**: Gradient-based optimization of sampler hyperparameters using JAX autodiff
- **Parallel Chain Execution**: Efficient batched sampling via JAX's `vmap()`
- **Comprehensive Diagnostics**: ESS, R-hat, and acceptance rate tracking

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
import jax.numpy as jnp
from jax import random
from samplers.GRAHMC import grahmc_init, grahmc_run

# Define target distribution
def log_prob_fn(x):
    return -0.5 * jnp.sum(x**2)

# Initialize and run sampler
key = random.PRNGKey(0)
init_pos = jnp.zeros(10)  # 10-dimensional problem
state = grahmc_init(key, init_pos, log_prob_fn)

samples, log_probs, accept_rate, final_state = grahmc_run(
    key, state, log_prob_fn,
    num_samples=1000,
    num_burnin=500,
    step_size=0.1,
    num_steps=30,
    gamma_max=1.0,
    schedule_type='tanh'
)
```

### Automated Parameter Tuning

```python
from tuning import optimize_all_schedules

results = optimize_all_schedules(
    key, log_prob_fn, init_pos,
    schedule_types=['constant', 'tanh', 'sigmoid', 'linear', 'sine'],
    L_grid=[30, 40, 50, 60],  # trajectory lengths
    n_optimization_steps=50
)
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
