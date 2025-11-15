# MCMC Benchmarking Guide

This guide explains how to benchmark MCMC samplers across diverse target distributions to evaluate performance and make informed algorithmic decisions.

---

## Quick Start

### 1. **Fast Test** (Verify Setup - ~30 seconds)
```bash
python run_benchmarks.py --quick --no-confirm --output-dir results/quick_test
```
**What it does:** Tests 2 samplers (HMC, NUTS) on 2 targets with reduced parameters
**When to use:** Before running full benchmarks, testing changes, debugging

### 2. **Standard Benchmark** (Comprehensive - ~30-60 minutes)
```bash
python run_benchmarks.py --no-confirm --output-dir results/standard_10d
```
**What it does:** All samplers × all targets with default settings (10D, 4 chains, ESS=1000)
**When to use:** Main benchmarking for research studies

### 3. **Custom Study** (Targeted Investigation)
```bash
# Focus on specific samplers and targets
python run_benchmarks.py \
    --samplers hmc nuts grahmc \
    --targets neals_funnel rosenbrock \
    --grahmc-schedules constant tanh \
    --dim 20 \
    --no-confirm \
    --output-dir results/challenging_targets_20d
```
**When to use:** Investigating specific research questions

---

## Understanding the Targets

Each target stresses different aspects of sampling:

| Target | Dimension | What It Tests | Expected Challenge |
|--------|-----------|---------------|-------------------|
| **standard_normal** | Any | Baseline correctness | None (easy) |
| **correlated_gaussian** | Any | High correlation (ρ=0.9) | Coordinate-wise methods struggle |
| **ill_conditioned_gaussian** | Any | Bad scaling (κ=100) | Step size sensitivity |
| **neals_funnel** | 10-20 | Varying curvature | Dynamic adaptation needed |
| **rosenbrock** | 10-20 | Curved geometry (banana) | Nonlinear correlations |

**Interpretation Tip:** If a sampler works on `standard_normal` but fails on `ill_conditioned_gaussian`, it has **step size tuning issues**.

---

## Command Reference

### Basic Options

```bash
--samplers SAMPLER [SAMPLER ...]
    Which samplers to test (default: all)
    Choices: rwmh, hmc, nuts, grahmc

--targets TARGET [TARGET ...]
    Which targets to test (default: all)
    Choices: standard_normal, correlated_gaussian,
             ill_conditioned_gaussian, neals_funnel, rosenbrock

--dim DIM
    Dimensionality (default: 10)
    Typical values: 5 (fast), 10 (standard), 20 (challenging), 50 (hard)

--chains N
    Number of parallel chains (default: 4)
    Recommendation: 4 for R-hat diagnostics, 8+ for better statistics

--target-ess ESS
    Target effective sample size (default: 1000)
    Recommendation: 1000 for research, 200 for quick tests
```

### GRAHMC-Specific Options

```bash
--grahmc-schedules SCHEDULE [SCHEDULE ...]
    Friction schedules for GRAHMC (default: all)
    Choices: constant, tanh, sigmoid, linear, sine

    Schedules explained:
    - constant: Step function (equivalent to RAHMC)
    - tanh/sigmoid: Smooth transitions (may have steepness instability)
    - linear/sine: Simple smooth schedules
```

### Performance Tuning

```bash
--batch-size SIZE
    Samples per batch (default: 2000)
    Larger = fewer batch overhead, more memory

--max-samples MAX
    Maximum total samples (default: 50000)
    Safety limit to prevent infinite loops

--seed SEED
    Random seed for reproducibility (default: 42)
```

### Convenience Flags

```bash
--quick
    Fast test mode: dim=5, chains=2, ess=200, subset of samplers/targets

--no-confirm
    Skip confirmation prompt (useful for scripts)

--output-dir DIR
    Where to save results (default: ./benchmark_results)
```

---

## Interpreting Results

### Output Files

After running, you'll get:

```
results/
├── benchmark_results.csv    # Spreadsheet-compatible
└── benchmark_results.json   # Machine-readable
```

### Key Metrics

| Metric | What It Measures | Good Value | Bad Value |
|--------|------------------|------------|-----------|
| **ess_bulk_min** | Effective sample size (minimum across dimensions) | ≥ target_ess | < target_ess |
| **rhat_max** | Convergence diagnostic (max across dimensions) | < 1.01 | > 1.05 |
| **accept_rate** | Acceptance probability | 0.6-0.8 (HMC/GRAHMC), 0.2-0.3 (RWMH) | < 0.1 or > 0.95 |
| **elapsed_time** | Wall-clock time (seconds) | Lower is better | - |
| **total_samples** | Samples needed to reach target ESS | Lower is better | Hit max_samples |

**Note on acceptance rates:**
- Reported acceptance may be higher than tuning target (this is normal!)
- Acceptance increases as chains move from initialization → typical set
- NUTS shows ~1.0 (uses slice sampling, not Metropolis-Hastings)

### Terminal Output Summary

```
BEST SAMPLER PER TARGET (by min ESS)
================================================================================
StandardNormal10D                    -> hmc              (ESS=1907.3)
IllConditioned10D_kappa100          -> nuts             (ESS=491.0)
```

This tells you: **HMC dominates on standard normal, but NUTS handles ill-conditioning better.**

---

## Example Workflows

### Workflow 1: Initial Exploration
**Goal:** Which sampler should I use for my problem?

```bash
# 1. Run quick test
python run_benchmarks.py --quick --no-confirm

# 2. Examine which sampler performs best on each target
cat benchmark_results/benchmark_results.csv | grep -E "target|ess_bulk"

# 3. Run full benchmark on winner
python test_samplers.py --sampler nuts --target neals_funnel --dim 10
```

### Workflow 2: Algorithm Development
**Goal:** Does my new GRAHMC schedule improve performance?

```bash
# Test new schedule vs baseline
python run_benchmarks.py \
    --samplers hmc grahmc \
    --grahmc-schedules constant tanh \
    --targets standard_normal neals_funnel \
    --no-confirm \
    --output-dir results/schedule_comparison

# Analyze: does tanh beat constant (RAHMC)?
```

### Workflow 3: Scaling Study
**Goal:** How does performance degrade with dimension?

```bash
for dim in 5 10 20 50; do
    python run_benchmarks.py \
        --dim $dim \
        --samplers hmc nuts \
        --targets standard_normal ill_conditioned_gaussian \
        --no-confirm \
        --output-dir results/scaling_dim_$dim
done

# Compare ESS across dimensions
python -c "
import pandas as pd
for d in [5, 10, 20, 50]:
    df = pd.read_csv(f'results/scaling_dim_{d}/benchmark_results.csv')
    print(f'Dim {d}: ESS = {df.ess_bulk_min.mean():.1f}')
"
```

### Workflow 4: Detailed Analysis in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/standard_10d/benchmark_results.csv')

# Compare samplers on each target
for target in df['target'].unique():
    target_df = df[df['target'] == target]

    print(f"\n{target}:")
    print(target_df[['sampler', 'schedule', 'ess_bulk_min', 'elapsed_time']].to_string())

# Plot ESS by sampler
pivot = df.pivot_table(
    values='ess_bulk_min',
    index='target',
    columns='sampler'
)
pivot.plot(kind='bar', figsize=(10, 6))
plt.ylabel('ESS (bulk, min)')
plt.title('Sampler Performance by Target')
plt.tight_layout()
plt.savefig('sampler_comparison.png')
```

---

## Troubleshooting

### Problem: All tests fail with low ESS
**Cause:** Not enough samples collected
**Solution:** Increase `--max-samples` or decrease `--target-ess`

```bash
python run_benchmarks.py --max-samples 100000 --target-ess 500
```

### Problem: Tests fail with high R-hat
**Cause:** Not enough chains or poor mixing
**Solution:** Increase `--chains` to 8 or more

```bash
python run_benchmarks.py --chains 8
```

### Problem: Benchmarks take too long
**Solution 1:** Use `--quick` mode
**Solution 2:** Reduce targets/samplers
```bash
python run_benchmarks.py --samplers hmc nuts --targets standard_normal
```

### Problem: GRAHMC steepness explodes
**Expected behavior** on Gaussian targets - smooth schedules (tanh/sigmoid) can have parameter instability.
**Not a bug!** Document in your results. Consider using `constant` schedule (RAHMC).

### Problem: "overall_pass" is False but metrics look good
**Likely cause:** R-hat slightly above 1.01 due to random variation with few chains
**Solution:** Not a real problem if ESS is good. Use more chains for better R-hat.

---

## Best Practices

### For Reproducible Research

1. **Always set seed:** `--seed 42`
2. **Document settings:** Save command in README
3. **Version control:** Commit results/ directory with git tags
4. **Save environment:** `pip freeze > requirements_benchmark.txt`

Example:
```bash
# Save exact command used
echo "python run_benchmarks.py --dim 10 --seed 42 --no-confirm" > results/COMMAND.txt

# Save system info
python --version > results/SYSTEM_INFO.txt
```

### For Fair Comparisons

- **Same seed** across runs
- **Same target ESS** (don't compare ESS=200 vs ESS=1000)
- **Same dimensionality**
- **Same number of chains** (affects R-hat)

### For Publication-Quality Results

1. Use **default parameters** (dim=10, chains=4, ess=1000)
2. Run **multiple seeds** and report mean ± std
3. Include **all targets** to show robustness
4. Report **both ESS and timing**

Example:
```bash
for seed in 42 43 44 45 46; do
    python run_benchmarks.py --seed $seed --no-confirm \
        --output-dir results/seed_$seed
done

# Aggregate results
python -c "
import pandas as pd
dfs = [pd.read_csv(f'results/seed_{s}/benchmark_results.csv')
       for s in [42,43,44,45,46]]
combined = pd.concat(dfs)
summary = combined.groupby(['sampler', 'target']).agg({
    'ess_bulk_min': ['mean', 'std'],
    'elapsed_time': ['mean', 'std']
})
print(summary)
" > results/summary_across_seeds.txt
```

---

## Research Questions You Can Answer

✅ **"Which sampler is best for my target class?"**
→ Run benchmark, check BEST SAMPLER PER TARGET table

✅ **"Does GRAHMC improve over standard HMC?"**
→ Compare ESS of `hmc` vs `grahmc-constant` (RAHMC)

✅ **"Which friction schedule works best?"**
→ Compare all `--grahmc-schedules`, check ESS on each target

✅ **"How does ill-conditioning affect performance?"**
→ Compare ESS on `standard_normal` vs `ill_conditioned_gaussian`

✅ **"Can NUTS adapt to varying curvature?"**
→ Check performance on `neals_funnel` vs other samplers

✅ **"What's the computational cost per effective sample?"**
→ Calculate `elapsed_time / ess_bulk_min` from results

✅ **"Which method scales best to high dimensions?"**
→ Run scaling study (see Workflow 3 above)

---

## Tips for Success

1. **Start small:** Always run `--quick` first
2. **Iterate:** Test one target/sampler, then expand
3. **Monitor:** Watch terminal output for errors
4. **Validate:** Check that standard_normal passes (sanity check)
5. **Document:** Save commands and parameters used
6. **Compare:** Use the CSV for easy pandas analysis
7. **Visualize:** Plot results (see Python example above)

---

## Reporting Results

### Minimal Report Template

```markdown
## Benchmark Results

**Setup:**
- Targets: standard_normal, ill_conditioned_gaussian, neals_funnel
- Samplers: HMC, NUTS, GRAHMC (constant, tanh)
- Dimension: 10
- Chains: 4
- Target ESS: 1000
- Seed: 42

**Key Findings:**
1. HMC achieved highest ESS on standard_normal (ESS=1907)
2. NUTS handled ill-conditioning better (ESS=491 vs HMC's 236)
3. GRAHMC-constant performed 10× worse than HMC on Gaussian targets
4. Neal's funnel revealed adaptive advantages of NUTS

**Recommendations:**
- Use HMC for well-conditioned Gaussian targets
- Use NUTS for unknown/challenging geometry
- GRAHMC provides no benefit for simple targets
```

---

## Next Steps

After benchmarking:
1. Identify best sampler for your use case
2. Fine-tune parameters with `tuning.py --plot`
3. Run production sampling with `test_samplers.py`
4. Report findings in papers/presentations

**Remember:** Benchmarking reveals sampler strengths/weaknesses. Use these insights to make informed decisions!
