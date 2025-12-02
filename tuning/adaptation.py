"""Optimized windowed adaptation manager for MCMC preconditioning."""
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time

from tuning.welford import welford_init, welford_update, welford_update_batch, welford_covariance
from tuning.dual_averaging import da_init, da_update, da_reset

from samplers.HMC import hmc_init, hmc_run
from samplers.NUTS import nuts_init, nuts_run
from samplers.GRAHMC import rahmc_init, rahmc_run


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
    sampler: str,
    target_log_prob: Any,
    target_grad_log_prob: Any,
    initial_position: jnp.ndarray,
    key: jnp.ndarray,
    num_warmup: int = 1000,
    target_accept: float = 0.65,
    schedule_type: Optional[str] = None,
    update_freq: int = 100,  # Update DA every N steps instead of every step
    learn_mass_matrix: bool = True,  # If False, skip mass matrix learning (use identity)
    **kwargs,
) -> Tuple[float, Optional[jnp.ndarray], jnp.ndarray, Dict]:
    """Run optimized windowed adaptation to find step_size and mass_matrix.

    OPTIMIZATIONS:
    1. Batch sampling: Run sampler for `update_freq` steps between DA updates (10-50x speedup)
    2. GRAHMC tuning: Properly tune gamma and steepness AFTER mass matrix learning

    CORRECT TUNING ORDER FOR GRAHMC:
    1. Phase 1-2: Tune step size + learn mass matrix (sphere the distribution)
    2. Phase 3: Tune gamma/steepness on the SPHERED geometry using the learned mass matrix

    Args:
        sampler: 'hmc', 'nuts', or 'grahmc'/'rahmc'
        target_log_prob: Log probability function
        target_grad_log_prob: Gradient of log probability
        initial_position: (n_chains, n_dim)
        key: JAX random key
        num_warmup: Total warmup iterations
        target_accept: Target acceptance rate
        schedule_type: Friction schedule for GRAHMC ('constant', 'tanh', etc.)
        update_freq: Update dual averaging every N steps (default: 100)
        learn_mass_matrix: If True, learn diagonal mass matrix via Welford. If False, use identity.

    Returns:
        (step_size, inv_mass_matrix, final_position, info_dict)
        inv_mass_matrix is None if learn_mass_matrix=False
    """
    n_chains, n_dim = initial_position.shape

    start_time = time.time()

    # ========================================================================
    # PHASE 1-2: Step Size Tuning + Mass Matrix Learning
    # ========================================================================
    # Use dimension-scaled initial step size for all samplers.
    # Starting from 1.0 causes issues on harmonic oscillators (like StandardNormal)
    # due to leapfrog resonances creating non-monotonic acceptance curves.
    # Starting small and letting DA increase step size works more reliably.
    initial_step = 0.5 / jnp.sqrt(n_dim)

    if sampler in ["grahmc", "rahmc"]:
        # Use conservative defaults for initial tuning
        gamma = 1.0
        steepness = 1.0 if schedule_type == "tanh" else 5.0
    else:
        gamma = None
        steepness = None

    da_state = da_init(initial_step)

    # Initial Mass Matrix (Identity)
    inv_mass_matrix = jnp.ones(n_dim)

    # Current position
    position = initial_position

    # Build Schedule
    schedule = build_schedule(num_warmup)

    print(f"Adaptation Schedule ({num_warmup} steps):")
    for s, e, t in schedule:
        print(f"  [{s:4d} - {e:4d}] {t}")

    # Run Windowed Adaptation
    welford_states_per_chain = None

    # If not learning mass matrix, keep identity throughout and skip Welford
    if not learn_mass_matrix:
        print("  [Mass matrix learning disabled - using identity]")

    for start_idx, end_idx, phase in schedule:
        window_len = end_idx - start_idx

        # Reset Welford at the start of a 'slow' window (only if learning mass matrix)
        # Maintain separate Welford state for each chain
        # Stan's approach - treats each chain as a separate sequence
        if phase == 'slow' and learn_mass_matrix:
            welford_states_per_chain = [welford_init(n_dim) for _ in range(n_chains)]

        # OPTIMIZATION: Run sampler in batches instead of 1 step at a time
        num_batches = max(1, window_len // update_freq)
        samples_per_batch = window_len // num_batches

        for batch_idx in range(num_batches):
            key, subkey = random.split(key)

            # Current parameters
            current_step_size = jnp.exp(da_state.log_step)

            # Get sampler-specific parameters from kwargs (but don't override gamma/steepness for GRAHMC)
            num_steps = kwargs.get("num_steps", 20)
            max_tree_depth = kwargs.get("max_tree_depth", 10)

            # Run Sampler for a batch of samples
            if sampler == "hmc":
                samples_batch, _, accept_rate, final_state = hmc_run(
                    subkey, target_log_prob, position,
                    step_size=float(current_step_size),
                    num_steps=num_steps,
                    num_samples=samples_per_batch,
                    burn_in=0,
                    inv_mass_matrix=inv_mass_matrix,
                )
            elif sampler == "nuts":
                samples_batch, _, accept_rate, final_state, _, mean_accept_probs = nuts_run(
                    subkey, target_log_prob, position,
                    step_size=float(current_step_size),
                    num_samples=samples_per_batch,
                    burn_in=0,
                    inv_mass_matrix=inv_mass_matrix,
                    max_tree_depth=max_tree_depth,
                )
                accept_rate = mean_accept_probs  # NUTS uses MH acceptance probability
            elif sampler in ["grahmc", "rahmc"]:
                from samplers.GRAHMC import get_friction_schedule
                friction_schedule = get_friction_schedule(schedule_type or "constant")
                samples_batch, _, accept_rate, final_state = rahmc_run(
                    subkey, target_log_prob, position,
                    step_size=float(current_step_size),
                    num_steps=num_steps,
                    gamma=float(gamma),
                    steepness=float(steepness),
                    num_samples=samples_per_batch,
                    burn_in=0,
                    friction_schedule=friction_schedule,
                    inv_mass_matrix=inv_mass_matrix,
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            # Update position for next batch
            position = final_state.position

            # Update Dual Averaging (Step Size)
            avg_accept = float(jnp.mean(accept_rate))
            da_state = da_update(da_state, avg_accept, target_accept)

            # Update Welford (Mass Matrix) if in slow phase and learning mass matrix
            if phase == 'slow' and learn_mass_matrix and sampler in ["hmc", "nuts", "grahmc", "rahmc"]:
                # Accumulate all samples from this batch
                # samples_batch shape: (num_samples, n_chains, n_dim)
                #
                # BUG FIX (Option B - Stan's approach):
                # Maintain separate Welford state for each chain, update each independently
                # This uses all chains but properly accounts for within-chain correlation
                for sample in samples_batch:
                    # sample shape: (n_chains, n_dim)
                    for chain_idx in range(n_chains):
                        welford_states_per_chain[chain_idx] = welford_update(
                            welford_states_per_chain[chain_idx],
                            sample[chain_idx]
                        )

        # End of Window Actions
        if phase == 'slow':
            if learn_mass_matrix:
                # 1. Estimate Variance (Stan's approach: compute per-chain, then average)
                # BUG FIX (Option B): Average variances across chains to properly handle correlation
                chain_variances = []
                for chain_idx in range(n_chains):
                    _, var = welford_covariance(welford_states_per_chain[chain_idx])
                    chain_variances.append(var)

                # Average variances across chains
                variance = jnp.mean(jnp.stack(chain_variances), axis=0)

                # Get sample count from first chain (all chains have same count)
                n_samples = welford_states_per_chain[0].count

                # 2. Regularize (Stan approach)
                # BUG FIX: Shrink toward identity (1.0), not toward ~0
                # Stan's formula: weighted average with prior on identity
                # Prior weight = 5, sample weight = n_samples
                shrinkage_weight = n_samples / (n_samples + 5.0)
                prior_weight = 5.0 / (n_samples + 5.0)
                variance = shrinkage_weight * variance + prior_weight * 1.0  # Shrink toward 1.0

                # Add numerical stability floor (separate from shrinkage)
                variance = jnp.maximum(variance, 1e-8)

                # 3. Update Mass Matrix
                inv_mass_matrix = variance
                print(f"  Window finished. Updated Mass Matrix. Range: [{jnp.min(variance):.4f}, {jnp.max(variance):.4f}]")
                print(f"    (n_samples_per_chain={n_samples:.0f}, n_chains={n_chains}, shrinkage={shrinkage_weight:.3f})")

                # 4. Reset Dual Averaging only when mass matrix changes
                # (geometry changed, so step size needs re-tuning from new baseline)
                da_state = da_reset(da_state)

    final_step_size = float(jnp.exp(da_state.log_step_bar))
    print(f"Warmup Complete. Final step_size: {final_step_size:.5f}")

    # ========================================================================
    # PHASE 3: GRAHMC Friction Refinement (AFTER mass matrix learning)
    # ========================================================================
    if sampler in ["grahmc", "rahmc"]:
        print(f"\n[Phase 3] Tuning GRAHMC friction on learned mass matrix...")

        from tuning.sequential_tune_grahmc import sequential_tune_grahmc

        # NOW tune gamma via grid search + ESJD using the learned mass matrix
        # This ensures friction is tuned for the sphered geometry
        num_steps = kwargs.get("num_steps", 20)  # Extract from kwargs
        tuned_step, tuned_gamma, tuned_steepness, tune_history = sequential_tune_grahmc(
            key=random.fold_in(key, 999),  # New key for tuning phase
            log_prob_fn=target_log_prob,
            grad_log_prob_fn=target_grad_log_prob,
            init_position=position,  # Use current warmed-up position
            num_steps=num_steps,  # Use the value from grid search
            schedule_type=schedule_type or "constant",
            target_accept=target_accept,
            max_iter_step=1000,  # Warmup iterations per gamma
            inv_mass_matrix=inv_mass_matrix,  # ← CRITICAL: Pass learned mass matrix
            gamma_coarse_values=None,  # Use default [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
            gamma_samples_per_eval=150,  # Samples for ESJD measurement per gamma
        )

        # from tuning.dual_averaging import joint_tune_grahmc
        # num_steps = kwargs.get("num_steps", 20)
        
        # Determine fixed steepness based on schedule
        fixed_steepness = 1.0 if schedule_type == "tanh" else 5.0
        
        # Run Joint Tuning
        # tuned_step, tuned_gamma, tuned_steepness, tune_history = joint_tune_grahmc(
        #     key=random.fold_in(key, 999),
        #     log_prob_fn=target_log_prob,
        #     grad_log_prob_fn=target_grad_log_prob,
        #     init_position=position,
        #     num_steps=num_steps,
        #     schedule_type=schedule_type or "constant",
        #     target_accept=target_accept,
        #     max_iter=1000, # usually sufficient
        #     inv_mass_matrix=inv_mass_matrix,
        #     current_step_size=final_step_size, # Pass the Phase 2 step size as a starting point
        #     fixed_steepness=fixed_steepness
        # )

        # Use the refined friction parameters
        gamma = tuned_gamma
        steepness = tuned_steepness

        final_step_size = tuned_step

        print(f"  Refined friction parameters:")
        print(f"    gamma={tuned_gamma:.5f}")
        print(f"    steepness={tuned_steepness:.5f}")
        print(f"    step_size={final_step_size:.5f} (re-tuned)")

    elapsed_time = time.time() - start_time

    info = {
        "elapsed_time": elapsed_time,
        "final_step_size": final_step_size,
        "inv_mass_matrix": inv_mass_matrix,
        "mass_matrix_learned": learn_mass_matrix,
    }

    # Add GRAHMC-specific parameters to info
    if sampler in ["grahmc", "rahmc"]:
        info["gamma"] = float(gamma) if gamma is not None else 1.0
        info["steepness"] = float(steepness) if steepness is not None else 5.0

    return final_step_size, inv_mass_matrix, position, info
