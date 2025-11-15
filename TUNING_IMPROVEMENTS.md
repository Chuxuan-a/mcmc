# Tuning Robustness Improvements

## Summary of Changes

This document describes the improvements made to the MCMC tuning infrastructure to enhance robustness and ensure tuned parameters are close to optimal.

## 1. Increased Samples Per Tuning Iteration ✅

**Problem**: Only 20 samples per tuning iteration gave ±10.7% standard error in acceptance rate estimates, causing noisy dual averaging updates.

**Solution**: Increased `n_samples_per_tune` from 20 → 100

**Impact**: Standard error reduced to ~4.8%, providing much more stable convergence.

**Files Modified**:
- `test_samplers.py`: All tuning functions now use `n_samples_per_tune = 100`

```python
# Before
n_samples_per_tune = 20  # SE ≈ 10.7%

# After
n_samples_per_tune = 100  # SE ≈ 4.8%
```

## 2. Added Acceptance Rate Convergence Check ✅

**Problem**: Tuning only checked parameter stability, not whether acceptance rate reached target. Could converge to suboptimal parameters.

**Solution**: Added dual convergence criterion:
1. Parameter must be stable (< 1% relative change)
2. Acceptance rate must be near target (within ±5%)

**Files Modified**:
- `test_samplers.py`: All tuning functions

```python
# Before
if relative_change < tolerance:
    converged_count += 1

# After
accept_error = abs(alpha - target_accept)
if relative_change < tolerance and accept_error < accept_tolerance:
    converged_count += 1
```

**Warning System**: If tuning reaches max iterations with acceptance far from target:
```
WARNING: Final acceptance 0.521 differs from target 0.650 by 0.129
```

## 3. Tightened GRAHMC Tolerances & Increased Exploration ✅

**Problem**: GRAHMC coordinate-wise tuning used same tolerance (1%) as single-parameter tuning, and low patience (2 cycles) could cause premature convergence.

**Solution**:
- Tolerance: 1.0% → 0.5% (tighter control for 3D parameter space)
- Max cycles: 10 → 15 (more exploration)
- Patience: 2 → 3 cycles (more conservative)

**Files Modified**:
- `test_samplers.py`: `dual_averaging_tune_grahmc()` and sub-functions

```python
# dual_averaging_tune_grahmc
tolerance = 0.005      # Was 0.01 (0.5% vs 1%)
max_cycles = 15        # Was 10
patience = 3           # Was 2

# Sub-functions (step_size, gamma, steepness)
tolerance = 0.005      # Was 0.01
```

## 4. Return Tuning Metadata ✅

**Problem**: No visibility into tuning process - couldn't debug failures or track convergence quality.

**Solution**: All tuning functions now return `(tuned_value, metadata_dict)` tuple.

**Metadata Fields**:

**Single-parameter tuning** (RWMH, HMC, NUTS):
```python
{
    "converged": True/False,
    "iterations": int,           # Iterations used
    "final_accept": float,       # Final acceptance rate
    "target_accept": float       # Target acceptance rate
}
```

**Coordinate-wise tuning** (GRAHMC):
```python
{
    "converged": True/False,
    "cycles": int                # Cycles used
}
```

**Files Modified**:
- `test_samplers.py`: All `dual_averaging_tune_*()` functions
- `test_samplers.py`: `run_sampler()` now unpacks and stores metadata

**Usage**:
```python
step_size, tune_meta = dual_averaging_tune_hmc(...)
print(f"Converged: {tune_meta['converged']}")
print(f"Iterations: {tune_meta['iterations']}")
```

## 5. Evolving Chain Positions During Tuning ✅

**Problem**: Every tuning iteration restarted from same `init_position`, tuning for initialization region rather than typical set.

**Solution**: Chains now evolve during tuning - each iteration continues from where previous iteration ended.

**Files Modified**:
- `test_samplers.py`: All tuning functions

```python
# Before
for m in range(max_iter):
    _, _, accept_rate, _ = sampler_run(
        key, log_prob_fn, init_position,  # Always restart from init
        ...
    )

# After
current_position = init_position
for m in range(max_iter):
    _, _, accept_rate, final_state = sampler_run(
        key, log_prob_fn, current_position,  # Evolves
        ...
    )
    current_position = final_state.position  # Update for next iteration
```

**Impact**: Tuned parameters now optimize for typical set, not initialization.

## 6. Grid Search for GRAHMC Trajectory Length ✅

**Status**: Grid search already exists in `tuning.py`

**Location**: `tune_grahmc_grid_search()` function

**Default Grid**: `[8, 16, 32, 64]` steps

**Usage**:
```python
python tuning.py --sampler grahmc --schedule constant --dim 10 \
    --num-steps-grid 8 16 32 64 --plot
```

**Recommendation**: Use tuning.py for grid search, test_samplers.py for fixed L testing.

---

## Breaking Changes

⚠️ **Function Signatures Changed**:

All tuning functions now return tuples instead of single values:

```python
# Old
step_size = dual_averaging_tune_hmc(...)

# New
step_size, metadata = dual_averaging_tune_hmc(...)
```

**Migration**: Code calling tuning functions must unpack the tuple:
```python
# If you don't need metadata
step_size, _ = dual_averaging_tune_hmc(...)

# If you want metadata
step_size, tune_meta = dual_averaging_tune_hmc(...)
if not tune_meta["converged"]:
    print("WARNING: Tuning did not converge!")
```

---

## Expected Performance Improvements

### Tuning Time
- **Increased** by ~5× due to n_samples_per_tune: 20 → 100
- **Worth it**: Much more stable convergence, fewer false positives

### Tuning Quality
- **Acceptance rate accuracy**: ±10.7% → ±4.8% SE
- **GRAHMC parameter precision**: 1% → 0.5% tolerance
- **Convergence reliability**: Dual criterion (parameter + acceptance)
- **Typical set optimization**: Chains evolve vs restart

### Failure Detection
- **Warnings** when acceptance ≠ target
- **Metadata** tracks convergence status
- **Easier debugging** with tuning iteration counts

---

## Validation Checklist

Before using updated code:

- [ ] Update all calls to tuning functions to unpack tuples
- [ ] Check that test_samplers.py runs without errors
- [ ] Verify tuning converges on standard_normal target
- [ ] Confirm acceptance rates are within ±5% of target
- [ ] Check metadata is populated correctly

---

## Recommendations for Future Work

### Priority 1: Add to Benchmarking
- Track `tuning_converged` in benchmark outputs
- Flag experiments where tuning failed
- Plot tuning iterations vs target difficulty

### Priority 2: Adaptive Tuning Parameters
- Increase `max_iter` if target is challenging (e.g., Neal's funnel)
- Tighten `tolerance` for publication-quality results
- Adjust `patience` based on dimension

### Priority 3: Tuning Diagnostics
- Save full tuning trajectories (parameter and acceptance history)
- Plot convergence curves for debugging
- Detect oscillations in coordinate-wise tuning

### Priority 4: Default Chain Count
- Increase from 4 → 8 chains for more robust R-hat
- Especially important for challenging targets

---

## References

- Hoffman & Gelman (2014): "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
- Roberts & Rosenthal (2001): "Optimal scaling for various Metropolis-Hastings algorithms" (RWMH target = 0.234)
- Stan Reference Manual: Dual averaging parameters (γ=0.05, t₀=10, κ=0.75)
