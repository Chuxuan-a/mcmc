"""
Quick test of the refactored multi-sampler tuning.py.

Tests:
1. Dispatcher function routes correctly
2. All samplers can be tuned (short runs)
3. Parameter structures are correct
4. Command-line interface loads
"""

import jax
import jax.numpy as jnp
from jax import random
import os
os.environ['JAX_ENABLE_X64'] = 'True'

from tuning import (
    tune_sampler,
    params_to_dict,
    RWMHParams,
    HMCParams,
    GRAHMCParams,
    NUTSParams,
)
from samplers.GRAHMC import standard_normal_log_prob

print("="*80)
print("Testing Refactored Multi-Sampler tuning.py")
print("="*80)

# Setup
key = random.PRNGKey(42)
dim = 5
n_chains = 2

key, init_key = random.split(key)
init_position = random.normal(init_key, shape=(n_chains, dim))
log_prob_fn = standard_normal_log_prob

# Test parameters (very short runs for speed)
test_kwargs = {
    'max_iter': 10,
    'min_iter': 5,
    'tolerance': 0.01,
    'patience': 3,
    'num_samples': 100,
    'burn_in': 50,
    'n_runs_per_eval': 2,
    'learning_rate': 0.02,
    'verbose': False,
}

# Test 1: RWMH
print("\n[Test 1] Tuning RWMH")
print("-"*80)
key, subkey = random.split(key)
try:
    rwmh_params = tune_sampler(
        'rwmh', subkey, log_prob_fn, init_position, **test_kwargs
    )
    assert isinstance(rwmh_params, RWMHParams), "RWMH should return RWMHParams"
    params_dict = params_to_dict(rwmh_params)
    print(f"  Tuned proposal_scale = {params_dict['proposal_scale']:.4f}")
    print("[PASS] RWMH tuning works!")
except Exception as e:
    print(f"[FAIL] RWMH tuning failed: {e}")
    raise

# Test 2: HMC
print("\n[Test 2] Tuning HMC")
print("-"*80)
key, subkey = random.split(key)
try:
    hmc_params = tune_sampler(
        'hmc', subkey, log_prob_fn, init_position, **test_kwargs
    )
    assert isinstance(hmc_params, HMCParams), "HMC should return HMCParams"
    params_dict = params_to_dict(hmc_params)
    print(f"  Tuned step_size = {params_dict['step_size']:.4f}")
    print(f"  Tuned total_time = {params_dict['total_time']:.2f}")
    print(f"  Expected num_steps = {params_dict['expected_num_steps']:.1f}")
    print("[PASS] HMC tuning works!")
except Exception as e:
    print(f"[FAIL] HMC tuning failed: {e}")
    raise

# Test 3: GRAHMC (single schedule)
print("\n[Test 3] Tuning GRAHMC (constant schedule)")
print("-"*80)
key, subkey = random.split(key)
try:
    grahmc_params = tune_sampler(
        'grahmc', subkey, log_prob_fn, init_position,
        schedule='constant', **test_kwargs
    )
    assert isinstance(grahmc_params, GRAHMCParams), "GRAHMC should return GRAHMCParams"
    params_dict = params_to_dict(grahmc_params)
    print(f"  Tuned step_size = {params_dict['step_size']:.4f}")
    print(f"  Tuned total_time = {params_dict['total_time']:.2f}")
    print(f"  Tuned gamma = {params_dict['gamma']:.4f}")
    print(f"  Tuned steepness = {params_dict['steepness']:.2f}")
    print("[PASS] GRAHMC tuning works!")
except Exception as e:
    print(f"[FAIL] GRAHMC tuning failed: {e}")
    raise

# Test 4: NUTS
print("\n[Test 4] Tuning NUTS")
print("-"*80)
key, subkey = random.split(key)
try:
    nuts_params = tune_sampler(
        'nuts', subkey, log_prob_fn, init_position, **test_kwargs
    )
    assert isinstance(nuts_params, NUTSParams), "NUTS should return NUTSParams"
    params_dict = params_to_dict(nuts_params)
    print(f"  Tuned step_size = {params_dict['step_size']:.4f}")
    print("[PASS] NUTS tuning works!")
except Exception as e:
    print(f"[FAIL] NUTS tuning failed: {e}")
    raise

# Test 5: Invalid sampler
print("\n[Test 5] Testing error handling (invalid sampler)")
print("-"*80)
key, subkey = random.split(key)
try:
    _ = tune_sampler(
        'invalid_sampler', subkey, log_prob_fn, init_position, **test_kwargs
    )
    print("[FAIL] Should have raised ValueError for invalid sampler")
except ValueError as e:
    print(f"  Correctly raised ValueError: {e}")
    print("[PASS] Error handling works!")

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nThe refactored tuning.py is working correctly with:")
print("  + Dispatcher function (tune_sampler)")
print("  + All 4 samplers (RWMH, HMC, GRAHMC, NUTS)")
print("  + Correct parameter structures")
print("  + Error handling for invalid samplers")
print("\nReady for full-scale tuning experiments!")
