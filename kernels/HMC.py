from __future__ import annotations
from typing import Callable, Optional, Tuple, NamedTuple
import numpy as np
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, random, vmap, lax

# type aliases
Array = jnp.ndarray
LogProbFn = Callable[[Array], Array] # x -> log p(x)


class HMCState(NamedTuple):
    position: Array        # (n_chains, n_dim)
    log_prob: Array        # (n_chains,)
    grad_log_prob: Array   # (n_chains, n_dim)
    accept_count: Array    # (n_chains,)

def _ensure_batched(x: Array) -> Tuple[Array, bool]:
    """
    Guarantees consistent shape (n_chains, n_dim) so that vectorized operations work seamlessly.
    Accepts input of shape (n_dim,) or (n_chains, n_dim).
    Returns (n_chains, n_dim) and a boolean indicating if we added a batch dimension.
    """
    x = jnp.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    elif x.ndim == 2:
        return x, False
    else:
        raise ValueError("Input must have shape (n_dim,) or (n_chains, n_dim).")
    

def standard_normal_log_prob(x: Array) -> Array:
    """log N(0, I) for x with shape (..., D). Returns shape (...,)."""
    x = jnp.asarray(x)
    D = x.shape[-1]
    dtype = x.dtype
    two_pi = jnp.array(2.0 * jnp.pi, dtype=dtype)
    half = jnp.array(0.5, dtype=dtype)
    # -0.5 * (||x||^2 + D * log(2π))
    return -half * (jnp.sum(x**2, axis=-1) + D * jnp.log(two_pi))



def hmc_init(init_position: Array, log_prob_fn: LogProbFn) -> HMCState:
    pos, _ = _ensure_batched(init_position) # pos is now guaranteed (n_chains, n_dim)
    n_chains = pos.shape[0]
    log_prob, grad_log_prob = vmap(jax.value_and_grad(log_prob_fn))(pos)
    log_prob = log_prob.astype(jnp.float64)
    grad_log_prob = grad_log_prob.astype(pos.dtype)
    return HMCState(
        position=pos,
        log_prob=log_prob,
        grad_log_prob=grad_log_prob,
        accept_count=jnp.zeros(n_chains, dtype=jnp.int32)
    )


@partial(jit, static_argnames=("num_steps", "log_prob_fn"))
def leapfrog(
    position: Array, 
    momentum: Array, 
    step_size: float, 
    lp: Array, 
    grad_lp: Array, 
    log_prob_fn: LogProbFn, 
    num_steps: int
) -> Tuple[Array, Array, Array, Array]:
    """ Perform num_steps of leapfrog integration. """
    pos_dtype = position.dtype
    lp_dtype = lp.dtype
    step_sz = jnp.asarray(step_size, dtype=pos_dtype)
    half = jnp.array(0.5, dtype=pos_dtype)

    def lf_step(carry, t):
        position, momentum, lp, grad_lp = carry
        momentum = momentum + half * step_sz * grad_lp
        position = position + step_sz * momentum
        new_lp, new_grad_lp = vmap(jax.value_and_grad(log_prob_fn))(position)
        new_lp = new_lp.astype(lp_dtype)
        new_grad_lp = new_grad_lp.astype(pos_dtype)
        momentum = momentum + half * step_sz * new_grad_lp
        return (position.astype(pos_dtype), momentum.astype(pos_dtype), new_lp, new_grad_lp), None
    
    (final_position, final_momentum, final_lp, final_grad_lp), _ = lax.scan(lf_step, (position, momentum, lp, grad_lp), jnp.arange(num_steps))

    return final_position.astype(pos_dtype), final_momentum.astype(pos_dtype), final_grad_lp.astype(pos_dtype), final_lp.astype(lp_dtype)


@partial(jit, static_argnames=("num_steps", "log_prob_fn", "return_proposal"))
def hmc_step(
    state: HMCState, 
    step_size: float, 
    num_steps: int, 
    key: Array, 
    log_prob_fn: LogProbFn,
    return_proposal: Optional[bool] = False
) -> Tuple[Array, HMCState] | Tuple[Array, HMCState, Array, Array, Array]:
    """ Single HMC step for all chains, with optional proposal tracking."""
    n_chains, n_dim = state.position.shape
    pos_dtype = state.position.dtype
    logprob_dtype = state.log_prob.dtype

    # next_key, k_momentum, k_accept = random.split(key, 3)
    key, step_key = random.split(key)
    k_momentum, k_accept = random.split(step_key, 2)

    momentum = random.normal(k_momentum, shape=(n_chains, n_dim), dtype=pos_dtype)
    step_size_arr = jnp.asarray(step_size, dtype=pos_dtype)

    kinetic_initial = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_initial = -state.log_prob + kinetic_initial.astype(logprob_dtype)

    current_position = state.position
    grad_lp = state.grad_log_prob
    log_prob = state.log_prob

    current_position, momentum, grad_lp, log_prob = leapfrog(
        current_position,
        momentum,
        step_size_arr,
        log_prob,
        grad_lp,
        log_prob_fn=log_prob_fn,
        num_steps=num_steps,
    )

    momentum = -momentum 

    kinetic_final = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_final = -log_prob + kinetic_final.astype(logprob_dtype)
    # overflow protection
    hamiltonian_final = jnp.where(jnp.isfinite(hamiltonian_final), hamiltonian_final, jnp.array(1e10, dtype=logprob_dtype))


    log_alpha = hamiltonian_initial - hamiltonian_final
    delta_H = hamiltonian_final - hamiltonian_initial

    u = random.uniform(k_accept, shape=(n_chains,), dtype=logprob_dtype)
    zero = jnp.array(0.0, dtype=logprob_dtype)
    accept = jnp.log(u) < jnp.minimum(zero, log_alpha)

    new_position = jnp.where(accept[:, None], current_position, state.position)
    new_log_prob = jnp.where(accept, log_prob, state.log_prob)
    new_grad_log_prob = jnp.where(accept[:, None], grad_lp, state.grad_log_prob)
    new_accept_count = state.accept_count + accept.astype(jnp.int32)

    new_state = HMCState(new_position, new_log_prob, new_grad_log_prob, new_accept_count)

    if return_proposal:
        return key, new_state, current_position, log_prob, delta_H
    else:
        return key, new_state


@partial(jit, static_argnames=("log_prob_fn", "num_steps", "num_samples", "burn_in", "track_proposals"))
def hmc_run(
    key: Array, 
    log_prob_fn: LogProbFn, 
    init_position: Array, 
    step_size: float, 
    num_steps: int, 
    num_samples: int, 
    burn_in: int = 0,
    track_proposals: bool = False
) -> Tuple:
    init_state = hmc_init(init_position, log_prob_fn)
    n_chains, n_dim = init_state.position.shape
    state = init_state
    step_size_arr = jnp.asarray(step_size, dtype=init_state.position.dtype)
    
    if burn_in > 0:
        def burnin_step(carry, t):
            k, s = carry
            k, s = hmc_step(s, step_size_arr, num_steps, k, log_prob_fn, return_proposal=False)
            return (k, s), None
        
        (key, init_state), _ = lax.scan(burnin_step, (key, init_state), jnp.arange(burn_in))
        
        state = init_state._replace(accept_count=jnp.zeros(n_chains, dtype=jnp.int32))
    else:
        state = init_state
    
    if track_proposals:
        def body_with_proposals(carry, _):
            k, s = carry
            pre_pos, pre_lp = s.position, s.log_prob
            k, s, prop_pos, prop_lp, delta_H = hmc_step(
                s, step_size_arr, num_steps, k, log_prob_fn, return_proposal=True
            )
            return (k, s), (pre_pos, pre_lp, prop_pos, prop_lp, delta_H, s.position, s.log_prob)
        
        (key, state), (pre_positions, pre_lps, prop_positions, prop_lps, deltas_H, post_positions, post_lps) = lax.scan(
            body_with_proposals, (key, state), length=num_samples
        )
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return (
            post_positions, post_lps,
            accept_rate, state,
            pre_positions, pre_lps,
            prop_positions, prop_lps, deltas_H
        )
    else:
        def body(carry, _):
            k, s = carry
            k, s = hmc_step(s, step_size_arr, num_steps, k, log_prob_fn, return_proposal=False)
            return (k, s), (s.position, s.log_prob)
        
        (key, state), (samples, lps) = lax.scan(body, (key, state), length=num_samples)
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return samples, lps, accept_rate, state

