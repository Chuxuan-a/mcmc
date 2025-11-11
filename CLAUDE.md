# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for developing and benchmarking MCMC (Markov Chain Monte Carlo) samplers, with emphasis on Randomized Adaptive Hamiltonian Monte Carlo (RAHMC) with various friction schedules. The project implements multiple samplers and provides automated parameter tuning using gradient-based optimization.

## Key Architecture

### Sampler Implementations (samplers/)

The repository contains four MCMC sampler implementations, all built with JAX:

- **RWMH.py**: Random Walk Metropolis-Hastings baseline sampler
- **HMC.py**: Standard Hamiltonian Monte Carlo
- **RAHMC.py**: Randomized Adaptive HMC with constant friction
- **GRAHMC.py**: Generalized RAHMC with multiple friction schedule functions

All samplers follow a consistent pattern:
1. State is stored in NamedTuples (e.g., `RAHMCState`, `HMCState`)
2. Each has an `*_init()` function that accepts initial positions and returns initialized state
3. Each has a `*_run()` function that performs sampling with burn-in
4. Position arrays are always batched as `(n_chains, n_dim)` for parallel chain execution
5. Log probabilities use float64 for numerical stability in energy calculations

### Friction Schedules (GRAHMC.py)

Five friction schedules are implemented for GRAHMC:
- `constant_schedule`: Step function at T/2
- `tanh_schedule`: Smooth hyperbolic tangent transition
- `sigmoid_schedule`: Sigmoid-based transition
- `linear_schedule`: Linear ramp from -γ to +γ
- `sine_schedule`: Sinusoidal variation

Each schedule takes `(t, T, gamma_max, steepness)` as parameters. Access via `FRICTION_SCHEDULES` dict or `get_friction_schedule(schedule_type)`.

### Parameter Tuning Framework (tuning.py)

The tuning system optimizes sampler hyperparameters using gradient descent on sampling efficiency metrics:

**Key components:**
- `TuningParams`: Full parameter set (step_size, num_steps, gamma, steepness) in log space
- `DynamicParams`: Subset optimized by gradient descent (excludes num_steps)
- `optimize_parameters()`: Optimizes dynamic params for a fixed trajectory length using Adam
- `optimize_all_schedules()`: Grid search over trajectory lengths + GD for each schedule type

**Objective function:**
- Maximizes proposal-level ESJD (Expected Squared Jump Distance)
- Includes acceptance rate penalties to maintain target acceptance (0.65)
- Uses variance reduction via multiple independent runs per evaluation
- All computations are JIT-compiled and differentiable through JAX

**Usage pattern:**
```python
results = optimize_all_schedules(
    key, log_prob_fn, init_position,
    schedule_types=['constant', 'tanh', 'sigmoid', 'linear', 'sine'],
    L_grid=[30, 40, 50, 60],  # trajectory lengths to try
    n_optimization_steps=50,
)
```

Returns dict mapping schedule types to `(best_params, optimization_history)`.

### JAX Integration

Critical JAX patterns used throughout:
- `jax.config.update("jax_enable_x64", True)` or `os.environ['JAX_ENABLE_X64'] = 'True'` for float64
- `@partial(jit, static_argnames=(...))` for JIT compilation with static arguments
- `vmap()` for automatic vectorization over chains
- `lax.scan()` for efficient loops in sampling
- `random.split()` for PRNG key management

## Common Commands

### Running the main analysis
```bash
# Execute the tuning script (performs full parameter optimization)
python tuning.py

# Run the Jupyter notebook for benchmarking
jupyter notebook run.ipynb
```

### Installing dependencies
```bash
pip install -r requirements.txt
```

### Key dependencies
- JAX: Core array operations and automatic differentiation
- Optax: Gradient-based optimization (Adam optimizer)
- ArviZ: MCMC diagnostics (ESS, R-hat)
- NumPy, Matplotlib, Seaborn: Data manipulation and visualization

## Important Implementation Details

### Numerical Precision
- **Positions and gradients**: float32 for memory efficiency
- **Log probabilities**: float64 for numerical stability in Metropolis-Hastings acceptance
- Energy differences (H1 - H0) are small and sensitive to precision, requiring float64

### Batching Convention
- All position arrays must be `(n_chains, n_dim)` shape
- Single chain inputs `(n_dim,)` are auto-batched by `_ensure_batched()` helper
- Enables parallel chain execution via `vmap()`

### Return Value Patterns
- Basic run: `(samples, log_probs, accept_rate, final_state)`
- With proposal tracking: `(..., pre_positions, pre_lps, prop_positions, prop_lps, deltas_H)`
- Samples shape: `(num_samples, n_chains, n_dim)`

### Trajectory Integration
The core leapfrog integration for GRAHMC is in `_trajectory_with_schedule()`:
- Integrates underdamped Langevin dynamics with time-varying friction
- Uses half-steps for momentum and full-steps for position
- Returns final `(position, momentum, log_prob, grad_log_prob)` after trajectory

### ESS Estimation
Uses Geyer's initial positive sequence estimator (`estimate_ess_geyer()`):
- FFT-based autocorrelation computation for efficiency
- Automatically finds cutoff where autocorrelation pairs become negative
- Vectorized across dimensions via `vmap()`

## Performance Considerations

- JIT compilation happens on first call; expect warmup delay
- Set `os.environ['JAX_LOG_COMPILES'] = '1'` to debug compilation
- Use `track_proposals=True` in `rahmc_run()` only when computing ESJD (adds memory overhead)
- Large trajectory lengths (L > 100) may cause memory issues with gradient tracking
- Multiple chains enable better R-hat diagnostics but increase memory linearly

## File Organization

```
.
├── samplers/           # Core MCMC implementations
│   ├── GRAHMC.py      # Main GRAHMC with friction schedules
│   ├── RAHMC.py       # Basic RAHMC
│   ├── HMC.py         # Standard HMC
│   └── RWMH.py        # Random Walk Metropolis-Hastings
├── tuning.py          # Automated parameter optimization
├── run.ipynb          # Benchmarking and comparison notebook
└── requirements.txt   # Python dependencies
```

## Git Workflow

Recent commits show focus on parameter tuning and removing old experimental code. When committing:
- Clean up old experimental files before committing
- Use descriptive messages about tuning results (e.g., "add tuning with successful settings")
