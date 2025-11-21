# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for developing and benchmarking MCMC (Markov Chain Monte Carlo) samplers, with emphasis on Generalized Repelling-Attracting Hamiltonian Monte Carlo (GRAHMC) with various friction schedules. The project implements multiple samplers and provides automated parameter tuning using dual averaging optimization.

## Key Architecture

### Sampler Implementations (samplers/)

The repository contains four MCMC sampler implementations, all built with JAX:

- **RWMH.py**: Random Walk Metropolis-Hastings baseline sampler
- **HMC.py**: Standard Hamiltonian Monte Carlo with leapfrog integration
- **NUTS.py**: No-U-Turn Sampler with automatic trajectory length selection
- **GRAHMC.py**: Generalized Randomized Adaptive HMC with multiple friction schedule functions

All samplers follow a consistent pattern:
1. State is stored in NamedTuples (e.g., `RWMState`, `HMCState`, `NUTSState`, `RAHMCState`)
2. Each has an `*_init()` function that accepts initial positions and returns initialized state
3. Each has a `*_step()` function for single sampler step
4. Each has a `*_run()` function that performs sampling with burn-in
5. Position arrays are always batched as `(n_chains, n_dim)` for parallel chain execution
6. Log probabilities use float64 for numerical stability in energy calculations

### Sampler Details

#### RWMH (RWMH.py)
- **Algorithm**: Random walk with Gaussian proposals `x' = x + scale * ε`, ε ~ N(0, I)
- **Key parameter**: `scale` (proposal standard deviation)
- **Target acceptance**: 0.234 (Roberts & Rosenthal optimal)
- **State**: `RWMState(position, log_prob, accept_count)`

#### HMC (HMC.py)
- **Algorithm**: Hamiltonian Monte Carlo with leapfrog integration
- **Key parameters**: `step_size`, `num_steps` (trajectory length L), `inv_mass_matrix`
- **Target acceptance**: 0.65
- **State**: `HMCState(position, log_prob, grad_log_prob, accept_count)`
- **Leapfrog steps**: Half momentum kick → full position step → recompute gradient → half momentum kick
- **Energy**: `H(q,p) = -log p(q) + 0.5 * p^T M^{-1} p`

#### NUTS (NUTS.py)
- **Algorithm**: No-U-Turn Sampler with iterative tree doubling
- **Key parameters**: `step_size`, `max_tree_depth` (default 10 → max 1024 steps), `delta_max`
- **Target acceptance**: 0.65 (uses mean MH acceptance from trajectory)
- **State**: `NUTSState(position, log_prob, grad_log_prob, accept_count)`
- **Key features**:
  - Iterative doubling via `lax.while_loop` (JIT-compatible, not recursive)
  - U-turn detection: `(q_right - q_left) · p_left < 0 OR (q_right - q_left) · p_right < 0`
  - Slice sampling for proposals (always "accepts")
  - Multinomial sampling: valid subtree of depth d contributes 2^d states
  - Gradient caching at tree endpoints (`_TrajectoryState`)

#### GRAHMC (GRAHMC.py)
- **Algorithm**: Conformal leapfrog with time-varying friction
- **Key parameters**: `step_size`, `num_steps`, `gamma` (friction amplitude), `steepness` (for tanh/sigmoid)
- **Target acceptance**: 0.65
- **State**: `RAHMCState(position, log_prob, grad_log_prob, accept_count)`
- **Conformal leapfrog step**:
  1. Apply friction: `p = p * exp(-γε/2)`
  2. Half momentum kick: `p = p + (ε/2) * ∇log p(q)`
  3. Position drift: `q = q + ε * M^{-1} * p`
  4. Recompute gradient
  5. Half momentum kick
  6. Apply friction: `p = p * exp(-γε/2)`

### Friction Schedules (GRAHMC.py)

Five friction schedules are implemented, accessed via `FRICTION_SCHEDULES` dict or `get_friction_schedule(schedule_type)`:

| Schedule | Formula | Uses Steepness | Description |
|----------|---------|----------------|-------------|
| `constant` | γ(t) = -γ if t<T/2 else +γ | No | Step function (original RAHMC) |
| `tanh` | γ(t) = γ_max × tanh(s×(2t/T - 1)) | Yes | Smooth hyperbolic tangent |
| `sigmoid` | γ(t) = γ_max × (2/(1+exp(-s×(t/T-0.5))) - 1) | Yes | Sigmoid transition |
| `linear` | γ(t) = -γ + 2γ×t/T | No | Linear ramp |
| `sine` | γ(t) = γ × sin(π×(t/T - 0.5)) | No | Sinusoidal variation |

Each schedule takes `(t, T, gamma_max, steepness)` as parameters.

### Tuning Framework (tuning/)

#### Dual Averaging (dual_averaging.py)
Implements Hoffman & Gelman (2014) algorithm for step size adaptation:

```
η_m = 1/(m + t₀)
H̄ = (1 - η_m) * H̄ + η_m * (target_accept - α)
log(θ) = μ - (√m / γ) * H̄
log(θ̄) = m^(-κ) * log(θ) + (1 - m^(-κ)) * log(θ̄)
```

Parameters: `γ=0.05`, `t₀=10`, `κ=0.75` (from Stan)

**Sampler-specific tuning functions**:
- `dual_averaging_tune_rwmh()`: Tunes scale → 0.234 acceptance
- `dual_averaging_tune_hmc()`: Tunes step_size → 0.65 acceptance (L fixed)
- `dual_averaging_tune_nuts()`: Tunes step_size → 0.65 acceptance (uses mean_accept_probs)
- `coordinate_wise_tune_grahmc()`: Cycles through step_size → gamma → steepness

**GRAHMC coordinate-wise tuning**:
- Cycles through parameters, tuning each via DA while others fixed
- Convergence: requires <2% relative change for `patience=2` consecutive cycles
- Up to `max_cycles=110` iterations

#### Windowed Adaptation (adaptation.py)
Stan-style three-phase warmup with mass matrix learning:

**Schedule** (`build_schedule()`):
1. **fast_init** (75 steps): Initial step size adaptation only
2. **slow** (doubling windows: 25, 50, 100, ...): Learn diagonal mass matrix + step size
3. **fast_final** (50 steps): Final step size refinement

**Process** (`run_adaptive_warmup()`):
- Phase 1-2: Tune step size + learn mass matrix via Welford's algorithm
- For GRAHMC: Phase 3 tunes friction parameters on **sphered geometry** (after mass matrix learned)
- Batch processing: Updates DA every `update_freq=100` steps for efficiency

**Mass matrix regularization**:
```python
regularizer = 1e-3 * 5.0 / (n_samples + 5.0)
variance = variance * (n_samples / (n_samples + 5.0)) + regularizer
```

#### Welford's Algorithm (welford.py)
Online mean and variance estimation without storing all samples:
- `welford_init(n_dim)` → initial state
- `welford_update(state, x)` → single sample update
- `welford_update_batch(state, batch)` → batch update via `lax.scan`
- `welford_covariance(state)` → returns `(mean, variance)`

#### Core Orchestration (core.py)
High-level tuning functions combining grid search and dual averaging:
- `tune_and_sample_rwmh()`: DA tuning + adaptive sampling
- `tune_and_sample_hmc_grid()`: Grid search over L, DA for step_size per L
- `tune_and_sample_nuts()`: DA tuning + adaptive sampling
- `tune_and_sample_grahmc_grid()`: Grid search over L, coordinate-wise DA per L

### Target Distributions (benchmarks/targets.py)

All targets follow `TargetDistribution` namedtuple with: `log_prob_fn`, `dim`, `true_mean`, `true_cov`, `name`, `description`, `init_sampler`

| Target | Factory Function | Challenge | Custom Init |
|--------|------------------|-----------|-------------|
| `standard_normal` | `standard_normal(dim)` | Baseline, well-conditioned | No |
| `correlated_gaussian` | `correlated_gaussian(dim, correlation=0.9)` | High correlation | No |
| `ill_conditioned_gaussian` | `ill_conditioned_gaussian(dim, condition_number=100)` | Step size tuning | No |
| `neals_funnel` | `neals_funnel(dim)` | Varying curvature | Yes |
| `student_t` | `student_t(dim, df=3)` | Heavy tails | Yes |
| `log_gamma` | `log_gamma(dim, shape=2, rate=1)` | Asymmetry, positivity | Yes |
| `rosenbrock` | `rosenbrock(dim, scale=0.1)` | Curved valleys | Yes |
| `gaussian_mixture` | `gaussian_mixture(dim, n_modes=2, separation=5)` | Multimodality | Yes |

Access via `get_target(name, dim=10, **kwargs)`.

### Testing & Benchmarking

#### test_samplers.py
Standalone sampler testing with comprehensive diagnostics:
- `run_sampler()`: Tunes parameters → samples adaptively until target ESS
- `compute_diagnostics()`: ArviZ-based R-hat (split, rank-normalized) and ESS (bulk/tail)
- `check_summary_statistics()`: Z-score test comparing sample means to truth using MCSE

**Validation criteria**:
- R-hat < 1.01
- Bulk ESS ≥ target_ess
- Tail ESS ≥ 50% of target_ess
- Z-score < 5 sigma (using MCSE)

#### run_benchmarks.py
Comprehensive benchmark suite across all sampler-target combinations:
- `run_single_benchmark()`: Runs one sampler on one target
- `run_all_benchmarks()`: Iterates all combinations, saves CSV/JSON
- `print_summary()`: Pass/fail rates, best performer per target

**Modes**:
- Adaptive ESS targeting (default): Sample until target ESS reached
- Fixed budget: `--fixed-budget N` samples all configurations with same count

## Common Commands

### Testing samplers
```bash
# Test individual samplers on various targets
python test_samplers.py --sampler rwmh --target standard_normal --dim 10 --chains 4 --target-ess 500
python test_samplers.py --sampler hmc --target ill_conditioned_gaussian --dim 10
python test_samplers.py --sampler nuts --target neals_funnel --dim 10
python test_samplers.py --sampler grahmc --schedule tanh --target correlated_gaussian --dim 10
```

### Running benchmarks
```bash
# Full benchmark suite
python run_benchmarks.py --output-dir results --dim 10

# Quick test mode (reduced parameters)
python run_benchmarks.py --quick

# Specific samplers/targets
python run_benchmarks.py --samplers hmc nuts --targets standard_normal ill_conditioned_gaussian

# Fixed sample budget (for fair comparison)
python run_benchmarks.py --fixed-budget 20000 --output-dir fixed_budget_results
```

### Parameter tuning via tuning/core.py
```bash
# Tune samplers with grid search
python -m tuning.core --sampler hmc --target standard_normal --dim 10
python -m tuning.core --sampler grahmc --schedule tanh --target standard_normal --dim 10
```

### Installing dependencies
```bash
pip install -r requirements.txt
```

### Key dependencies
- JAX: Core array operations and automatic differentiation
- ArviZ: MCMC diagnostics (ESS, R-hat)
- NumPy, Matplotlib, Seaborn: Data manipulation and visualization
- Pandas: Results tabulation

## Important Implementation Details

### Numerical Precision
- **Positions and gradients**: float32 for memory efficiency
- **Log probabilities**: float64 for numerical stability in Metropolis-Hastings acceptance
- Energy differences (H1 - H0) are small and sensitive to precision, requiring float64
- Overflow protection: `H = where(isfinite(H), H, 1e10)` to reject divergent proposals

### Batching Convention
- All position arrays must be `(n_chains, n_dim)` shape
- Single chain inputs `(n_dim,)` are auto-batched by `_ensure_batched()` helper
- Enables parallel chain execution via `vmap()`
- Samples output shape: `(num_samples, n_chains, n_dim)`

### Return Value Patterns
- Basic run: `(samples, log_probs, accept_rate, final_state)`
- With proposal tracking: `(..., pre_positions, pre_lps, prop_positions, prop_lps, deltas_H)`
- NUTS run: `(samples, log_probs, accept_rate, final_state, tree_depths, mean_accept_probs)`

### JAX Patterns
- `jax.config.update("jax_enable_x64", True)` for float64 support
- `@partial(jit, static_argnames=(...))` for JIT compilation with static arguments
- `vmap()` for automatic vectorization over chains
- `lax.scan()` for efficient loops in sampling
- `lax.while_loop()` for dynamic termination (NUTS tree building)
- `lax.cond()` for conditional execution without Python branching
- `random.split()` for PRNG key management

### Convergence Detection
Both parameter stability AND acceptance rate stability required:
```python
if relative_change < tolerance and accept_error < accept_tolerance:
    converged_count += 1
else:
    converged_count = 0
if converged_count >= patience:
    # Converged
```

### Adaptive Sampling
Samples collected in batches until target ESS reached:
```python
while total_samples < max_samples:
    samples_batch, ..., final_state = sampler_run(...)
    current_position = final_state.position  # Chain continuity
    all_samples.append(samples_batch)
    if compute_ess(all_samples) >= target_ess:
        break
```

## Performance Considerations

- JIT compilation happens on first call; expect warmup delay
- Set `os.environ['JAX_LOG_COMPILES'] = '1'` to debug compilation
- Large trajectory lengths (L > 100) may cause memory issues with gradient tracking
- Multiple chains enable better R-hat diagnostics but increase memory linearly
- Batch DA updates (`update_freq=100`) provide 10-50x speedup vs per-step updates

## File Organization

```
.
├── samplers/               # Core MCMC implementations
│   ├── RWMH.py            # Random Walk Metropolis-Hastings
│   ├── HMC.py             # Hamiltonian Monte Carlo
│   ├── NUTS.py            # No-U-Turn Sampler
│   └── GRAHMC.py          # Generalized RAHMC with friction schedules
├── tuning/                 # Parameter tuning framework
│   ├── __init__.py
│   ├── core.py            # Main tuning orchestration
│   ├── dual_averaging.py  # Dual averaging algorithms
│   ├── adaptation.py      # Windowed adaptation with mass matrix
│   ├── welford.py         # Online mean/variance estimation
│   └── plots.py           # Diagnostic visualization
├── benchmarks/             # Benchmarking infrastructure
│   ├── __init__.py
│   └── targets.py         # Target distribution definitions
├── test_samplers.py       # Sampler testing with diagnostics
├── run_benchmarks.py      # Comprehensive benchmark suite
├── run_benchmarks_v2.py   # Alternative benchmark script
├── run_benchmarks_no_mass.py  # Benchmarks without mass matrix
└── requirements.txt       # Python dependencies
```

## Git Workflow

When committing:
- Clean up old experimental files before committing
- Use descriptive messages about tuning results
- Results files (CSV, JSON) in `tuning_results/` or `benchmark_results/` directories
