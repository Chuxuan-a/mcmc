"""Windowed adaptation manager for MCMC preconditioning."""
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple, Dict, Any

from tuning.welford import welford_init, welford_update_batch, welford_covariance
from tuning.dual_averaging import da_init, da_update, da_reset

from samplers.HMC import hmc_run
from samplers.NUTS import nuts_run
from samplers.GRAHMC import rahmc_run


def build_schedule(num_steps: int, initial_buffer: int = 75, final_buffer: int = 50, window_base: int = 25) -> list:
    """Build a Stan-style windowed schedule.
    
    Returns a list of tuples: (start_index, end_index, type)
    Types: 'fast_init', 'slow', 'fast_final'
    """
    schedule = []
    start = 0
    
    # 1. Initial fast interval
    if num_steps < initial_buffer + final_buffer + window_base:
        # Fallback for very short runs
        return [(0, num_steps, 'fast_init')]
        
    schedule.append((start, start + initial_buffer, 'fast_init'))
    start += initial_buffer
    
    # 2. Windowed slow intervals (doubling size)
    # Remaining space for windows
    available = num_steps - start - final_buffer
    window_size = window_base
    
    while available >= 2 * window_size: # While we can fit a window
        schedule.append((start, start + window_size, 'slow'))
        start += window_size
        available -= window_size
        window_size *= 2 # Double the window
        
    # If there is leftover space in 'slow' region, extend the last window
    if available > 0 and len(schedule) > 1 and schedule[-1][2] == 'slow':
        prev_start, _, _ = schedule.pop()
        # Add the remaining available time to this window
        schedule.append((prev_start, start + available, 'slow'))
        start += available
    elif available > 0:
        # Create one last slow window if we couldn't merge
        schedule.append((start, start + available, 'slow'))
        start += available

    # 3. Final fast interval
    schedule.append((start, num_steps, 'fast_final'))
    
    return schedule


def run_adaptive_warmup(
    key: jnp.ndarray,
    sampler_name: str,
    log_prob_fn: Any,
    init_position: jnp.ndarray,
    num_warmup: int = 1000,
    target_accept: float = 0.8, # Higher default for NUTS/HMC in warmup
    **sampler_kwargs
) -> Tuple[float, jnp.ndarray, jnp.ndarray, Dict]:
    """Run full windowed adaptation to find step_size and mass_matrix.
    
    Args:
        sampler_name: 'hmc', 'nuts', or 'grahmc'
        init_position: (n_chains, n_dim)
        num_warmup: Total warmup iterations
        sampler_kwargs: constants like num_steps (L), gamma, steepness, etc.
        
    Returns:
        (step_size, inv_mass_matrix, final_position, info_dict)
    """
    n_chains, n_dim = init_position.shape
    
    # 1. Initialize State
    # Initial step size guess (standard heuristic)
    initial_step = 1.0 # Will be quickly tuned by DA
    da_state = da_init(initial_step)
    
    # Initial Mass Matrix (Identity)
    # inv_mass_matrix represents the diagonal vector M^{-1}
    inv_mass_matrix = jnp.ones(n_dim)
    
    # Current position
    position = init_position
    
    # Build Schedule
    schedule = build_schedule(num_warmup)
    
    print(f"Adaptation Schedule ({num_warmup} steps):")
    for s, e, t in schedule:
        print(f"  [{s:4d} - {e:4d}] {t}")

    # 2. Run Schedule
    welford_state = None
    
    # Batch size for running the sampler inside the loop
    # We run 1 step at a time (or small chunks) to update DA
    # For efficiency in Python loop, we'll run chunks of 1 if possible, 
    # or rely on the sampler_run function for 'n' samples.
    # Ideally, we pass `inv_mass_matrix` to the sampler.
    
    for start_idx, end_idx, phase in schedule:
        window_len = end_idx - start_idx
        
        # Reset Welford at the start of a 'slow' window
        if phase == 'slow':
            welford_state = welford_init(n_dim)
            
        # Run the window
        # We process in small batches to update DA frequently
        # A batch size of 10-20 is a good balance between JAX dispatch overhead and DA updates
        update_freq = 1 
        
        for i in range(window_len):
            key, subkey = random.split(key)
            
            # Current parameters
            current_step_size = jnp.exp(da_state.log_step)
            
            # Run Sampler (1 step)
            # IMPORTANT: Samplers must now accept inv_mass_matrix!
            if sampler_name == "hmc":
                # Unpack kwargs
                L = sampler_kwargs.get("num_steps", 20)
                _, _, accept_rate, final_state = hmc_run(
                    subkey, log_prob_fn, position,
                    step_size=float(current_step_size), num_steps=L,
                    inv_mass_matrix=inv_mass_matrix, # <--- Component A integration
                    num_samples=1, burn_in=0
                )
            elif sampler_name == "nuts":
                depth = sampler_kwargs.get("max_tree_depth", 10)
                _, _, _, final_state, _, mean_accept_probs = nuts_run(
                    subkey, log_prob_fn, position,
                    step_size=float(current_step_size), max_tree_depth=depth,
                    inv_mass_matrix=inv_mass_matrix, # <--- Component A integration
                    num_samples=1, burn_in=0
                )
                accept_rate = mean_accept_probs # NUTS uses specific stat
            elif sampler_name in ["grahmc", "rahmc"]:
                L = sampler_kwargs.get("num_steps", 20)
                gamma = sampler_kwargs.get("gamma", 0.1)
                steepness = sampler_kwargs.get("steepness", 5.0)
                fs = sampler_kwargs.get("friction_schedule", None)
                
                _, _, accept_rate, final_state = rahmc_run(
                    subkey, log_prob_fn, position,
                    step_size=float(current_step_size), num_steps=L,
                    gamma=gamma, steepness=steepness,
                    inv_mass_matrix=inv_mass_matrix, # <--- Component A integration
                    friction_schedule=fs,
                    num_samples=1, burn_in=0
                )
            
            position = final_state.position
            avg_accept = float(jnp.mean(accept_rate))
            
            # Update Dual Averaging (Step Size)
            da_state = da_update(da_state, avg_accept, target_accept)
            
            # Update Welford (Mass Matrix) if in slow phase
            if phase == 'slow':
                # position shape is (n_chains, n_dim)
                welford_state = welford_update_batch(welford_state, position)

        # End of Window Actions
        if phase == 'slow':
            # 1. Estimate Variance
            _, variance = welford_covariance(welford_state)
            
            # 2. Regularize (Stan approach)
            # Push towards identity slightly to avoid 0 variance
            # var_new = (n / (n + 5)) * var + (5 / (n + 5)) * 1e-3
            n_samples = welford_state.count
            regularizer = 1e-3 * 5.0 / (n_samples + 5.0)
            variance = variance * (n_samples / (n_samples + 5.0)) + regularizer
            
            # 3. Update Mass Matrix
            inv_mass_matrix = variance
            print(f"  Window finished. Updated Mass Matrix. Range: [{jnp.min(variance):.4f}, {jnp.max(variance):.4f}]")
            
            # 4. Reset Dual Averaging
            # We keep the step size we found, but reset the 'memory' 
            # because the geometry just changed dramatically.
            da_state = da_reset(da_state)
    
    final_step_size = float(jnp.exp(da_state.log_step_bar))
    print(f"Warmup Complete. Final step_size: {final_step_size:.5f}")
    
    info = {
        "final_step_size": final_step_size,
        "inv_mass_matrix": inv_mass_matrix
    }
    
    return final_step_size, inv_mass_matrix, position, info