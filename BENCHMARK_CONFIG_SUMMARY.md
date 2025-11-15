# MCMC Benchmark Configuration & Output Summary

## Test Matrix Overview

**Total configurations per full run**: 24 combinations
- **Samplers**: 4 (RWMH, HMC, NUTS, GRAHMC)
- **Targets**: 5 (standard_normal, correlated_gaussian, ill_conditioned_gaussian, neals_funnel, rosenbrock)
- **GRAHMC schedules**: 5 (constant, tanh, sigmoid, linear, sine)
- **Total**: 3 simple samplers × 5 targets + GRAHMC × 5 schedules × 5 targets = 40 experiments

## Per-Target Specifications

| Target | Dim | Key Parameters | What It Tests | Ground Truth |
|--------|-----|----------------|---------------|--------------|
| **standard_normal** | Any | μ=0, Σ=I | Baseline correctness | ✓ Mean, Cov known |
| **correlated_gaussian** | Any | ρ=0.9 (compound symmetry) | High correlation handling | ✓ Mean, Cov known |
| **ill_conditioned_gaussian** | Any | κ=100 (eigenvalues 1→100) | Step size sensitivity | ✓ Mean, Cov known |
| **neals_funnel** | 10-20 | x₀~N(0,9), xᵢ~N(0,exp(x₀)) | Varying curvature/scale | ✓ Mean, Cov known |
| **rosenbrock** | 10-20 | scale=0.1 (banana) | Curved geometry | ✗ Empirical only |

## Per-Sampler Configurations

### 1. RWMH (Random Walk Metropolis-Hastings)

**Fixed hyperparameters:**
- Initial scale: 2.38/√d (Roberts & Rosenthal optimal)

**Tuned via dual averaging:**
- `scale`: Proposal standard deviation
- Target acceptance: 0.234 (optimal for RWMH)
- Max iterations: 2000
- Convergence: 10 consecutive iterations with <1% relative change

**Sampling:**
- Batch size: 2000 samples
- Max samples: 50,000
- Adaptive batching until ESS ≥ 1000

**Outputs:**
```python
{
  "scale": float,              # Tuned proposal scale
  "accept_rate": float,        # Actual acceptance rate during sampling
  "total_samples": int,        # Samples needed to reach target ESS
  "ess_bulk_min": float,       # Minimum bulk ESS across dimensions
  "ess_bulk_mean": float,      # Mean bulk ESS
  "ess_tail_min": float,       # Minimum tail ESS
  "rhat_max": float,           # Maximum R-hat (convergence diagnostic)
  "rhat_mean": float,          # Mean R-hat
  "elapsed_time": float,       # Wall-clock time (seconds)
  "overall_pass": bool         # rhat<1.01 AND ess>=1000 AND stats_pass
}
```

### 2. HMC (Hamiltonian Monte Carlo)

**Fixed hyperparameters:**
- Trajectory length: L=20 steps (not tuned - requires grid search)
- Initial step size: 0.5/√d

**Tuned via dual averaging:**
- `step_size`: Leapfrog integration step size
- Target acceptance: 0.65 (optimal for HMC)
- Max iterations: 2000
- Convergence: 10 consecutive iterations with <1% relative change

**Sampling:**
- Batch size: 2000 samples
- Max samples: 50,000
- Adaptive batching until ESS ≥ 1000

**Outputs:**
```python
{
  "step_size": float,          # Tuned integration step size
  "num_steps": int,            # Fixed at 20
  "accept_rate": float,
  "total_samples": int,
  "ess_bulk_min": float,
  "ess_bulk_mean": float,
  "ess_tail_min": float,
  "rhat_max": float,
  "rhat_mean": float,
  "elapsed_time": float,
  "overall_pass": bool
}
```

### 3. NUTS (No-U-Turn Sampler)

**Fixed hyperparameters:**
- Max tree depth: 10 (max trajectory = 2¹⁰ = 1024 steps)
- Initial step size: 0.5/√d
- Delta max: 1000.0 (divergence threshold)

**Tuned via dual averaging:**
- `step_size`: Leapfrog integration step size
- Target acceptance: 0.65
- Max iterations: 2000
- Convergence: 10 consecutive iterations with <1% relative change

**Sampling:**
- Batch size: 2000 samples
- Max samples: 50,000
- Adaptive batching until ESS ≥ 1000
- **Note**: NUTS automatically adapts trajectory length per sample

**Outputs:**
```python
{
  "step_size": float,          # Tuned integration step size
  "max_tree_depth": int,       # Fixed at 10
  "accept_rate": float,        # Actually mean MH prob (not true acceptance)
  "total_samples": int,
  "ess_bulk_min": float,
  "ess_bulk_mean": float,
  "ess_tail_min": float,
  "rhat_max": float,
  "rhat_mean": float,
  "elapsed_time": float,
  "overall_pass": bool,
  # NUTS-specific:
  "tree_depths": array,        # Tree depth per sample (not saved in CSV)
  "mean_tree_depth": float     # Could add this
}
```

### 4. GRAHMC (Generalized Repelling-Attracting HMC)

**Fixed hyperparameters:**
- Trajectory length: L=20 steps
- Initial step size: 0.5/√d
- Initial gamma: 0.5
- Initial steepness: 5.0 (tanh), 10.0 (sigmoid), N/A (others)

**Tuned via coordinate-wise dual averaging:**
- `step_size`, `gamma`, and optionally `steepness`
- Target acceptance: 0.65
- Max cycles: 10 (each cycle tunes all parameters sequentially)
- Convergence: 2 consecutive cycles with <1% change in all parameters

**Per-schedule details:**

| Schedule | Parameters Tuned | Friction Function γ(t) |
|----------|------------------|----------------------|
| constant | step_size, gamma | Step at T/2: -γ if t<T/2, +γ if t≥T/2 |
| tanh | step_size, gamma, steepness | γ·tanh(steepness·(t-T/2)/T) |
| sigmoid | step_size, gamma, steepness | γ·(2/(1+exp(-steepness·(t-T/2)/T)) - 1) |
| linear | step_size, gamma | γ·(2t/T - 1) |
| sine | step_size, gamma | γ·sin(π·t/T) |

**Sampling:**
- Batch size: 2000 samples
- Max samples: 50,000
- Adaptive batching until ESS ≥ 1000

**Outputs:**
```python
{
  "step_size": float,          # Tuned integration step size
  "num_steps": int,            # Fixed at 20
  "gamma": float,              # Tuned friction amplitude
  "steepness": float | None,   # Tuned (tanh/sigmoid) or None
  "schedule_type": str,        # 'constant', 'tanh', 'sigmoid', 'linear', 'sine'
  "accept_rate": float,
  "total_samples": int,
  "ess_bulk_min": float,
  "ess_bulk_mean": float,
  "ess_tail_min": float,
  "rhat_max": float,
  "rhat_mean": float,
  "elapsed_time": float,
  "overall_pass": bool
}
```

## Shared Diagnostic Metrics

All samplers compute identical diagnostics via ArviZ:

**Convergence:**
- **rhat_max**: Maximum split R-hat across dimensions (should be <1.01)
- **rhat_mean**: Mean split R-hat

**Efficiency:**
- **ess_bulk_min**: Minimum bulk ESS across dimensions (should be ≥ target_ess)
- **ess_bulk_mean**: Mean bulk ESS
- **ess_tail_min**: Minimum tail ESS

**Validation:**
- **stats_pass**: True if sample mean/std within 15% of true values (Gaussian targets only)

**Pass/Fail:**
- **overall_pass**: `rhat_max < 1.01 AND ess_bulk_min >= target_ess AND stats_pass`

## Initialization Strategies

| Target | Initialization |
|--------|----------------|
| standard_normal | N(0, 4I) (overdispersed) |
| correlated_gaussian | N(0, 4I) |
| ill_conditioned_gaussian | N(0, 4I) |
| neals_funnel | **Custom**: x₀~N(0,9), xᵢ~N(0,1) |
| rosenbrock | **Custom**: N(1, 0.25I) (near mode) |

## Key Tuning Parameters Summary

| Sampler | Tuning Method | Target Accept | Max Iter/Cycles | Patience | What's Tuned |
|---------|---------------|---------------|-----------------|----------|--------------|
| RWMH | Dual averaging | 0.234 | 2000 | 10 | scale |
| HMC | Dual averaging | 0.65 | 2000 | 10 | step_size |
| NUTS | Dual averaging | 0.65 | 2000 | 10 | step_size |
| GRAHMC | Coordinate-wise DA | 0.65 | 10 cycles | 2 | step_size, gamma, [steepness] |

**Note**: All dual averaging uses Hoffman & Gelman (2014) parameters:
- γ = 0.05
- t₀ = 10.0
- κ = 0.75

---

## Robustness Refinement Suggestions

### 1. **Critical: Add Gradient Call Tracking**

**Issue**: Current metrics don't measure computational efficiency properly.

**Recommendation**: Track gradient evaluations per sampler:
- RWMH: 0 gradients (proposal-based)
- HMC: `total_samples × num_steps`
- NUTS: `sum(2^tree_depths - 1)` per sample (complex)
- GRAHMC: `total_samples × num_steps`

**Add to outputs:**
```python
"n_grad_evals": int,           # Total gradient calls
"ess_per_grad": float,         # ess_bulk_min / n_grad_evals
```

**Why**: This is the gold standard for comparing MCMC efficiency. ESS alone is misleading when HMC uses 20 gradients/sample.

### 2. **Medium: Increase Chain Count for Robust R-hat**

**Issue**: 4 chains can give noisy R-hat estimates, leading to false failures.

**Recommendation**:
```python
--chains 8  # for default runs
```

**Why**: R-hat variance ∝ 1/n_chains. With 4 chains, random fluctuations can push R-hat > 1.01 even when converged.

### 3. **Medium: Add Tuning Diagnostics to Output**

**Issue**: Can't debug why tuning failed without inspecting terminal output.

**Recommendation**: Add to metadata:
```python
"tuning_iters": int,           # Iterations/cycles until convergence
"tuning_converged": bool,      # Whether tuning converged vs hit max_iter
"final_accept_rate_tuning": float,  # Acceptance during final tuning step
```

**Why**: Helps identify when tuning is unstable (e.g., GRAHMC steepness explosion).

### 4. **Low: Dimensionality Scaling Tests**

**Issue**: All tests use dim=10. Unclear how samplers scale.

**Recommendation**: Add workflow for dimension sweep:
```bash
# Add to BENCHMARKING_GUIDE.md
python run_benchmarks.py --dim 5 --output-dir results/dim5
python run_benchmarks.py --dim 10 --output-dir results/dim10
python run_benchmarks.py --dim 20 --output-dir results/dim20
python run_benchmarks.py --dim 50 --output-dir results/dim50
```

**Why**: High-dimensional performance is critical for real applications.

### 5. **Low: Add Warmup/Burn-in Phase**

**Issue**: Samples start from overdispersed initialization, chains haven't reached typical set.

**Current behavior**: `burn_in=0` during sampling (tuning acts as warmup)

**Recommendation**: Add explicit warmup:
```python
# After tuning, run warmup with tuned parameters
warmup_samples, _, _, final_position = sampler_run(
    ..., num_samples=n_tune, burn_in=0
)
# Discard warmup, start sampling from final_position
```

**Why**: Current approach works but is implicit. Explicit warmup is clearer and standard practice.

### 6. **Medium: Add Per-Dimension ESS Reporting**

**Issue**: Only report min/mean ESS. Can't identify which dimensions are problematic.

**Recommendation**:
```python
"ess_bulk_by_dim": list,       # ESS for each dimension
"worst_dim": int,              # Index of dimension with lowest ESS
```

**Why**: Helps diagnose issues (e.g., Neal's funnel should show x₀ has lower ESS than x₁...xₙ).

### 7. **Critical: Add Error Handling for Numerical Issues**

**Issue**: NaN/Inf in gradients cause cryptic failures.

**Recommendation**: Add checks in `run_single_benchmark`:
```python
if jnp.any(jnp.isnan(samples)) or jnp.any(jnp.isinf(samples)):
    return {
        ...,
        "error": "NaN/Inf detected in samples",
        "numerical_failure": True
    }
```

**Why**: Helps identify when targets are too challenging for current tuning.

### 8. **Low: Seed-Level Reproducibility**

**Issue**: Single seed per full benchmark. Hard to quantify variance.

**Current**: `--seed 42`

**Recommendation**: Add multi-seed workflow to guide:
```bash
for seed in 42 43 44 45 46; do
    python run_benchmarks.py --seed $seed --output-dir results/seed_$seed
done
```

**Why**: Publication-quality results need mean ± std across seeds.

### 9. **Medium: Add Memory Profiling**

**Issue**: Unknown which sampler/target combinations cause OOM.

**Recommendation**: Track peak memory:
```python
import tracemalloc
tracemalloc.start()
# ... run sampling ...
current, peak = tracemalloc.get_traced_memory()
metadata["peak_memory_mb"] = peak / 1024 / 1024
tracemalloc.stop()
```

**Why**: NUTS with deep trees can use excessive memory.

### 10. **Critical: Make L (Trajectory Length) a Tunable Parameter**

**Issue**: L=20 is fixed for HMC/GRAHMC. May be suboptimal for different targets.

**Current workaround**: Manual grid search in `tuning.py`

**Recommendation**: Add trajectory length to benchmark grid:
```python
--num-steps-grid 8 16 32 64
```

Run each sampler with multiple L values, report best ESS/gradient.

**Why**: L is as important as step_size. Fixing it biases results.

---

## Highest Priority Recommendations

1. **Add gradient call tracking** (essential for fair comparison)
2. **Make trajectory length tunable** (currently biases against HMC/GRAHMC)
3. **Add numerical error handling** (prevents cryptic failures)
4. **Increase default chains to 8** (more robust R-hat)
5. **Add tuning diagnostics to output** (debuggability)

These changes would make the framework publication-ready and robust to edge cases.
