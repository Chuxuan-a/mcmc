# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for developing and benchmarking MCMC (Markov Chain Monte Carlo) samplers, with emphasis on Generalized Randomized Adaptive Hamiltonian Monte Carlo (GRAHMC) with various friction schedules. The project implements multiple samplers and provides automated parameter tuning using gradient-based optimization.

## Key Architecture

### Sampler Implementations (samplers/)

The repository contains four MCMC sampler implementations, all built with JAX:

- **RWMH.py**: Random Walk Metropolis-Hastings baseline sampler
- **HMC.py**: Standard Hamiltonian Monte Carlo
- **GRAHMC.py**: Generalized Randomized Adaptive HMC with multiple friction schedule functions (includes constant schedule which is equivalent to the original RAHMC)
- **NUTS.py**: No-U-Turn Sampler with automatic trajectory length selection

All samplers follow a consistent pattern:
1. State is stored in NamedTuples (e.g., `RAHMCState`, `HMCState`, `NUTSState`)
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

### NUTS Implementation (NUTS.py)

The No-U-Turn Sampler (NUTS) automatically selects trajectory lengths using iterative tree doubling:

**Key features:**
- **Iterative tree doubling**: Builds trajectory by doubling tree depth (2^0, 2^1, 2^2, ... steps)
- **U-turn detection**: Stops when trajectory starts doubling back on itself
- **Slice sampling**: Uses slice variable for proposal selection (always "accepts")
- **Multinomial sampling**: Samples proposals proportional to number of valid states in subtrees
- **Gradient tracking**: Caches gradients at tree endpoints to avoid recomputation
- **Divergence detection**: Flags when energy change exceeds `delta_max` threshold

**Return values:**
- `nuts_run()` returns: `(samples, log_probs, accept_rate, final_state, tree_depths, mean_accept_probs)`
- `tree_depths`: Depth reached for each sample (max trajectory = 2^depth steps)
- `mean_accept_probs`: Mean Metropolis-Hastings acceptance probability from leapfrog steps

**Important implementation details:**
- Uses `lax.while_loop` for iterative doubling (not recursion) to enable JIT compilation
- Tracks `_TrajectoryState` with left/right endpoints and their gradients
- Proposal counting: Valid subtree of depth d contributes 2^d states to multinomial sampling
- Energy and slice threshold computed once per NUTS step with consistent momentum

### Testing Framework (test_samplers.py)

Comprehensive test suite for validating all samplers on standard normal distribution:

**Dual averaging tuning:**
- Automatically tunes sampler hyperparameters using Hoffman & Gelman (2014) algorithm
- RWMH: Tunes proposal scale (target acceptance: 0.234)
- HMC: Tunes step size with fixed trajectory length (target acceptance: 0.65)
- NUTS: Tunes step size with automatic trajectory length (target acceptance: 0.65)
- GRAHMC: Coordinate-wise tuning of step_size, gamma, and steepness (target acceptance: 0.65)

**Convergence diagnostics:**
- Split R-hat (rank-normalized) for convergence detection
- Bulk and tail ESS (Effective Sample Size) using ArviZ
- Summary statistics validation (mean ≈ 0, std ≈ 1 for standard normal)

**Adaptive sampling:**
- Collects samples in batches until target ESS is reached
- Maintains chain continuity across batches
- Reports ESS progress after each batch

### Parameter Tuning Framework (tuning.py)

The tuning system optimizes sampler hyperparameters using gradient descent on sampling efficiency metrics:

**Supported samplers:**
- RWMH: Tunes `proposal_scale`
- HMC: Tunes `step_size` and `total_time` (uses stochastic rounding for trajectory length)
- GRAHMC: Tunes `step_size`, `total_time`, `gamma`, and `steepness` for each friction schedule
- NUTS: Tunes `step_size` (trajectory length is automatic)

**Key components:**
- Parameter structures: `RWMHParams`, `HMCParams`, `GRAHMCParams`, `NUTSParams` (all in log space)
- `tune_sampler()`: Dispatcher function that routes to appropriate tuning function
- `tune_rwmh()`, `tune_hmc()`, `tune_grahmc()`, `tune_nuts()`: Individual tuning functions
- `optimize_parameters()`: Core optimization loop using Adam with convergence detection

**Objective function:**
- Maximizes proposal-level ESJD (Expected Squared Jump Distance)
- Includes acceptance rate penalties to maintain target acceptance (0.234 for RWMH, 0.65 for others)
- Uses variance reduction via multiple independent runs per evaluation
- All computations are JIT-compiled and differentiable through JAX
- **Straight-through estimator for trajectory length**: Uses hard thresholds `(i < actual_steps)` in forward pass (maintains detailed balance), soft sigmoid gradients in backward pass (allows gradient flow through continuous `actual_steps = T/ε`)

**Convergence detection:**
- Waits `min_iter` iterations before checking convergence
- Checks relative change in best metric < `tolerance`
- Requires `patience` consecutive converged iterations to stop early

**Usage pattern:**
```python
# Tune RWMH
params = tune_sampler('rwmh', key, log_prob_fn, init_position)

# Tune GRAHMC with tanh schedule
params = tune_sampler('grahmc', key, log_prob_fn, init_position, schedule='tanh')

# Tune all GRAHMC schedules
results = tune_sampler('grahmc', key, log_prob_fn, init_position, schedule='all')

# Command line
python tuning.py --sampler rwmh --dim 10
python tuning.py --sampler grahmc --schedule tanh --dim 10
```

Returns tuned parameters (type depends on sampler).

### JAX Integration

Critical JAX patterns used throughout:
- `jax.config.update("jax_enable_x64", True)` or `os.environ['JAX_ENABLE_X64'] = 'True'` for float64
- `@partial(jit, static_argnames=(...))` for JIT compilation with static arguments
- `vmap()` for automatic vectorization over chains
- `lax.scan()` for efficient loops in sampling
- `random.split()` for PRNG key management

## Common Commands

### Testing samplers
```bash
# Test individual samplers on 10D standard normal (with dual averaging tuning)

# Random Walk Metropolis-Hastings
python test_samplers.py --sampler rwmh --dim 10 --chains 4 --target-ess 500

# Hamiltonian Monte Carlo
python test_samplers.py --sampler hmc --dim 10 --chains 4 --target-ess 500

# No-U-Turn Sampler (NUTS)
python test_samplers.py --sampler nuts --dim 10 --chains 4 --target-ess 500

# GRAHMC with constant friction (equivalent to RAHMC)
python test_samplers.py --sampler grahmc --schedule constant --dim 10 --chains 4 --target-ess 500

# GRAHMC with smooth schedules
python test_samplers.py --sampler grahmc --schedule tanh --dim 10 --chains 4 --target-ess 500
python test_samplers.py --sampler grahmc --schedule sigmoid --dim 10 --chains 4 --target-ess 500
python test_samplers.py --sampler grahmc --schedule linear --dim 10 --chains 4 --target-ess 500
python test_samplers.py --sampler grahmc --schedule sine --dim 10 --chains 4 --target-ess 500

# Quick debug/test of NUTS
python debug_nuts.py
```

### Tuning sampler parameters
```bash
# Tune individual samplers
python tuning.py --sampler rwmh --dim 10
python tuning.py --sampler hmc --dim 10
python tuning.py --sampler nuts --dim 10

# Tune GRAHMC with specific friction schedule
python tuning.py --sampler grahmc --schedule constant --dim 10
python tuning.py --sampler grahmc --schedule tanh --dim 10
python tuning.py --sampler grahmc --schedule sigmoid --dim 10

# Tune all GRAHMC schedules at once
python tuning.py --sampler grahmc --schedule all --dim 10

# Run full GRAHMC analysis (legacy mode with visualization)
python tuning.py --full-analysis

# Customize tuning parameters
python tuning.py --sampler hmc --dim 20 --max-iter 200 --chains 8
```

### Running notebooks
```bash
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
- NUTS run: `(samples, log_probs, accept_rate, final_state, tree_depths, mean_accept_probs)`
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
│   ├── GRAHMC.py      # Generalized RAHMC with friction schedules (constant, tanh, sigmoid, linear, sine)
│   ├── HMC.py         # Standard Hamiltonian Monte Carlo
│   ├── NUTS.py        # No-U-Turn Sampler (automatic trajectory length)
│   └── RWMH.py        # Random Walk Metropolis-Hastings
├── test_samplers.py   # Comprehensive test suite with dual averaging
├── debug_nuts.py      # Quick NUTS debugging script
├── tuning.py          # Automated parameter optimization (gradient-based ESJD)
├── run.ipynb          # Benchmarking and comparison notebook
└── requirements.txt   # Python dependencies
```

## Git Workflow

Recent commits show focus on parameter tuning and removing old experimental code. When committing:
- Clean up old experimental files before committing
- Use descriptive messages about tuning results (e.g., "add tuning with successful settings")
