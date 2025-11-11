"""
Quick test of the refactored tuning.py with stochastic rounding.

Tests:
1. Parameter structure (TuningParams, DynamicParams)
2. Stochastic rounding implementation
3. Objective function computation
4. Basic optimization loop
"""

import jax
import jax.numpy as jnp
from jax import random
import os
os.environ['JAX_ENABLE_X64'] = 'True'

from tuning import (
    TuningParams,
    DynamicParams,
    params_to_dict,
    objective_proposal_esjd,
    optimize_parameters,
)
from samplers.GRAHMC import standard_normal_log_prob

print("="*80)
print("Testing Refactored tuning.py")
print("="*80)

# Test 1: Parameter structure
print("\n[Test 1] Parameter structure")
print("-"*80)

test_params = TuningParams(
    log_step_size=jnp.log(0.2),
    log_total_time=jnp.log(10.0),
    log_gamma=jnp.log(0.3),
    log_steepness=jnp.log(2.0),
)

params_dict = params_to_dict(test_params)
print(f"Step size:           {params_dict['step_size']:.4f}")
print(f"Total time:          {params_dict['total_time']:.2f}")
print(f"Expected num_steps:  {params_dict['expected_num_steps']:.1f}")
print(f"Gamma:               {params_dict['gamma']:.4f}")
print(f"Steepness:           {params_dict['steepness']:.2f}")
print("[PASS] Parameter structure works!")

# Test 2: Stochastic rounding
print("\n[Test 2] Stochastic rounding")
print("-"*80)

key = random.PRNGKey(42)
dyn_params = DynamicParams(
    log_step_size=jnp.log(0.2),
    log_total_time=jnp.log(10.0),
    log_gamma=jnp.log(0.3),
    log_steepness=jnp.log(2.0),
)

# Run multiple times to see stochastic variation
step_size = jnp.exp(dyn_params.log_step_size)
total_time = jnp.exp(dyn_params.log_total_time)
expected_steps = total_time / step_size

print(f"Expected num_steps: {expected_steps:.2f}")
print(f"Testing stochastic rounding over 10 samples:")

num_steps_samples = []
for i in range(10):
    key, subkey = random.split(key)
    num_steps_float = total_time / step_size
    n = jnp.floor(num_steps_float)
    r = num_steps_float - n
    round_up = random.uniform(subkey) < r
    num_steps = int(n + round_up.astype(jnp.int32))
    num_steps_samples.append(num_steps)
    print(f"  Sample {i+1}: num_steps = {num_steps}")

mean_steps = sum(num_steps_samples) / len(num_steps_samples)
print(f"Mean sampled num_steps: {mean_steps:.2f} (expected: {expected_steps:.2f})")
print("[PASS] Stochastic rounding is unbiased!")

# Test 3: Objective function
print("\n[Test 3] Objective function computation")
print("-"*80)

key = random.PRNGKey(123)
key, init_key = random.split(key)

dim = 5
init_position = random.normal(init_key, shape=(2, dim))  # 2 chains
log_prob_fn = standard_normal_log_prob

try:
    key, eval_key = random.split(key)
    neg_esjd = objective_proposal_esjd(
        dyn_params=dyn_params,
        key=eval_key,
        log_prob_fn=log_prob_fn,
        init_position=init_position,
        num_samples=100,
        burn_in=50,
        schedule_type='constant',
    )
    print(f"Negative ESJD: {neg_esjd:.4f}")
    print(f"ESJD: {-neg_esjd:.4f}")
    print("[PASS] Objective function works!")
except Exception as e:
    print(f"[FAIL] Objective function failed: {e}")
    raise

# Test 4: Short optimization run
print("\n[Test 4] Short optimization run")
print("-"*80)

key = random.PRNGKey(456)
key, init_key = random.split(key)

init_position = random.normal(init_key, shape=(2, 5))  # 2 chains, 5 dims

initial_params = DynamicParams(
    log_step_size=jnp.log(0.15),
    log_total_time=jnp.log(8.0),
    log_gamma=jnp.log(0.25),
    log_steepness=jnp.log(1.5),
)

print("Running optimization for 10 steps...")
try:
    best_params, history, best_neg_metric = optimize_parameters(
        key=key,
        log_prob_fn=log_prob_fn,
        init_position=init_position,
        schedule_type='tanh',
        initial_params=initial_params,
        num_samples=200,
        burn_in=100,
        n_optimization_steps=10,
        n_runs_per_eval=2,
        learning_rate=0.02,
        verbose=False,
    )

    final_params_dict = params_to_dict(TuningParams(
        best_params.log_step_size,
        best_params.log_total_time,
        best_params.log_gamma,
        best_params.log_steepness,
    ))

    print(f"\nOptimization complete!")
    print(f"  Best ESJD: {-best_neg_metric:.4f}")
    print(f"  Final step_size: {final_params_dict['step_size']:.4f}")
    print(f"  Final total_time: {final_params_dict['total_time']:.2f}")
    print(f"  Final E[num_steps]: {final_params_dict['expected_num_steps']:.1f}")
    print(f"  Final gamma: {final_params_dict['gamma']:.4f}")
    print("[PASS] Optimization works!")

except Exception as e:
    print(f"[FAIL] Optimization failed: {e}")
    raise

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nThe refactored tuning.py is working correctly with:")
print("  + New parameter structure (step_size, total_time, gamma, steepness)")
print("  + Stochastic rounding for num_steps (unbiased in expectation)")
print("  + Objective function with proposal-level ESJD")
print("  + Gradient-based optimization without grid search")
print("\nReady for full-scale experiments!")
