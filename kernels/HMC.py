from __future__ import annotations
from typing import Callable, Tuple, NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import jax
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



def hmc_init(init_position: Array, log_prob_fn: LogProbFn) -> HMCState:

    init_position = jnp.asarray(init_position)
    n_chains = init_position.shape[0]
    
    log_prob, grad_log_prob = jax.value_and_grad(log_prob_fn)(init_position)
    
    return HMCState(
        position=init_position,
        log_prob=log_prob,
        grad_log_prob=grad_log_prob,
        accept_count=jnp.zeros(n_chains, dtype=jnp.int32)
    )


def leapfrog(position: Array, momentum: Array, step_size: float, grad_lp: Array, log_prob_fn: LogProbFn) -> Tuple[Array, Array, Array]:
    
    momentum = momentum + 0.5 * step_size * grad_lp
    position = position + step_size * momentum
    new_lp, new_grad_lp = jax.value_and_grad(log_prob_fn)(position)
    momentum = momentum + 0.5 * step_size * new_grad_lp

    return position, momentum, new_grad_lp, new_lp


def hmc_step(state: HMCState, step_size: float, num_steps: int, key: Array, log_prob_fn: LogProbFn) -> Tuple[Array, HMCState]:

    n_chains, n_dim = state.position.shape
    next_key, k_momentum, k_accept = random.split(key, 3)

    momentum = random.normal(k_momentum, shape=(n_chains, n_dim))

    kinetic_initial = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_initial = -state.log_prob + kinetic_initial

    current_position = state.position
    grad_lp = state.grad_log_prob
    log_prob = state.log_prob

    for _ in range(num_steps):
        current_position, momentum, grad_lp, log_prob = leapfrog(current_position, momentum, step_size, grad_lp, log_prob_fn)

    momentum = -momentum 

    kinetic_final = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_final = -log_prob + kinetic_final

    log_alpha = hamiltonian_initial - hamiltonian_final
    u = random.uniform(k_accept, shape=(n_chains,))
    accept = jnp.log(u) < jnp.minimum(0.0, log_alpha)

    new_position = jnp.where(accept[:, None], current_position, state.position)
    new_log_prob = jnp.where(accept, log_prob, state.log_prob)
    new_grad_log_prob = jnp.where(accept[:, None], grad_lp, state.grad_log_prob)
    new_accept_count = state.accept_count + accept.astype(jnp.int32)

    return next_key, HMCState(new_position, new_log_prob, new_grad_log_prob, new_accept_count)


def hmc_run(key: Array, log_prob_fn: LogProbFn, init_position: Array, step_size: float, num_steps: int, num_samples: int, burn_in: int) -> Tuple[Array, HMCState]:

    init_state = hmc_init(init_position, log_prob_fn)

    def body_fn(i, val):
        key, state, samples = val
        key, new_state = hmc_step(state, step_size, num_steps, key, log_prob_fn)
        samples = samples.at[i].set(new_state.position)
        return key, new_state, samples

    samples = jnp.zeros((num_samples, *init_position.shape))
    _, final_state, samples = lax.fori_loop(0, num_samples, body_fn, (key, init_state, samples))

    return samples[burn_in:], final_state

