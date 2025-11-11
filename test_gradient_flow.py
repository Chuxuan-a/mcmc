"""
Test gradient flow through dynamic HMC trajectory length.

Verifies that gradients flow properly through the continuous actual_steps
parameter when using soft masking.
"""
import jax
import jax.numpy as jnp
from jax import grad, random
import os
os.environ['JAX_ENABLE_X64'] = 'True'

from tuning import objective_hmc_esjd, HMCParams
from samplers.GRAHMC import standard_normal_log_prob

print("="*80)
print("Testing Gradient Flow through Dynamic HMC")
print("="*80)

# Setup
key = random.PRNGKey(42)
key, init_key = random.split(key)

dim = 5
n_chains = 2
init_position = random.normal(init_key, shape=(n_chains, dim))
log_prob_fn = standard_normal_log_prob

# Test parameters
params = HMCParams(
    log_step_size=jnp.log(0.2),
    log_total_time=jnp.log(10.0)
)

print(f"\nInitial parameters:")
print(f"  log_step_size = {params.log_step_size:.4f}")
print(f"  log_total_time = {params.log_total_time:.4f}")
print(f"  step_size = {jnp.exp(params.log_step_size):.4f}")
print(f"  total_time = {jnp.exp(params.log_total_time):.4f}")

# Compute objective
key, eval_key = random.split(key)
neg_esjd = objective_hmc_esjd(params, eval_key, log_prob_fn, init_position, num_samples=100, burn_in=50, max_steps=200)
print(f"\nObjective (negative ESJD): {neg_esjd:.6f}")

# Compute gradients
print("\nComputing gradients...")
grad_fn = grad(objective_hmc_esjd, argnums=0)
grads = grad_fn(params, eval_key, log_prob_fn, init_position, 100, 50, 200)

print(f"\nGradients:")
print(f"  d(obj)/d(log_step_size) = {grads.log_step_size:.6f}")
print(f"  d(obj)/d(log_total_time) = {grads.log_total_time:.6f}")

# Check if gradients are non-zero
if abs(grads.log_step_size) > 1e-6:
    print(f"\n[OK] Gradient flows through log_step_size")
else:
    print(f"\n[WARNING] Gradient through log_step_size is zero or negligible!")

if abs(grads.log_total_time) > 1e-6:
    print(f"[OK] Gradient flows through log_total_time")
else:
    print(f"[WARNING] Gradient through log_total_time is zero or negligible!")

# Test gradient with different total_time values
print("\n" + "-"*80)
print("Testing gradient sensitivity to total_time changes:")
print("-"*80)

for T in [5.0, 10.0, 15.0, 20.0]:
    params_test = HMCParams(log_step_size=jnp.log(0.2), log_total_time=jnp.log(T))
    key, eval_key = random.split(key)
    neg_esjd = objective_hmc_esjd(params_test, eval_key, log_prob_fn, init_position, 100, 50, 200)
    grads = grad_fn(params_test, eval_key, log_prob_fn, init_position, 100, 50, 200)
    print(f"T={T:5.1f}: obj={neg_esjd:8.4f}, grad_T={grads.log_total_time:8.4f}, actual_steps={T/0.2:5.1f}")

print("\n" + "="*80)
if abs(grads.log_step_size) > 1e-6 and abs(grads.log_total_time) > 1e-6:
    print("SUCCESS: Gradients flow through both parameters!")
else:
    print("FAILURE: Gradient flow is broken!")
print("="*80)
