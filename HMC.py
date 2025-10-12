from __future__ import annotations
from typing import Callable, Optional, Tuple, NamedTuple
import numpy as np
import arviz as az
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
    D = x.shape[-1]
    # -0.5 * (||x||^2 + D * log(2π))
    return -0.5 * (jnp.sum(x**2, axis=-1) + D * jnp.log(2.0 * jnp.pi))



def hmc_init(init_position: Array, log_prob_fn: LogProbFn) -> HMCState:

    pos, _ = _ensure_batched(init_position) # pos is now guaranteed (n_chains, n_dim)
    n_chains = pos.shape[0]
    
    log_prob, grad_log_prob = vmap(jax.value_and_grad(log_prob_fn))(pos)
    
    return HMCState(
        position=pos,
        log_prob=log_prob,
        grad_log_prob=grad_log_prob,
        accept_count=jnp.zeros(n_chains, dtype=jnp.int32)
    )


@partial(jit, static_argnames=("num_steps", "log_prob_fn"))
def leapfrog(position: Array, momentum: Array, step_size: float, lp: Array, grad_lp: Array, log_prob_fn: LogProbFn, num_steps: int) -> Tuple[Array, Array, Array, Array]:
    """ Perform num_steps of leapfrog integration. """
    def step(carry, t):
        position, momentum, lp, grad_lp = carry
        momentum = momentum + 0.5 * step_size * grad_lp
        position = position + step_size * momentum
        new_lp, new_grad_lp = vmap(jax.value_and_grad(log_prob_fn))(position)
        # new_grad_lp = vmap(jax.grad(log_prob_fn))(position)
        momentum = momentum + 0.5 * step_size * new_grad_lp
        return (position, momentum, new_lp, new_grad_lp), None
    
    (final_position, final_momentum, final_lp, final_grad_lp), _ = lax.scan(step, (position, momentum, lp, grad_lp), jnp.arange(num_steps))

    # final_lp, _ = vmap(jax.value_and_grad(log_prob_fn))(final_position)
    # final_lp = vmap(log_prob_fn)(final_position)
    return final_position, final_momentum, final_grad_lp, final_lp


@partial(jit, static_argnames=("num_steps", "log_prob_fn"))
def hmc_step(state: HMCState, step_size: float, num_steps: int, key: Array, log_prob_fn: LogProbFn) -> Tuple[Array, HMCState]:
    """ Single HMC step for all chains. """
    n_chains, n_dim = state.position.shape
    next_key, k_momentum, k_accept = random.split(key, 3)

    momentum = random.normal(k_momentum, shape=(n_chains, n_dim))

    kinetic_initial = 0.5 * jnp.sum(momentum**2, axis=-1)
    hamiltonian_initial = -state.log_prob + kinetic_initial

    current_position = state.position
    grad_lp = state.grad_log_prob
    log_prob = state.log_prob

    # for _ in range(num_steps):
    #     current_position, momentum, grad_lp, log_prob = leapfrog(current_position, momentum, step_size, grad_lp, log_prob_fn)
    # 
    # JIT has to compile num_steps copies of the same operations for the for loop 
    # because it unrolls into a huge computation graph that is linear in num_steps.
    current_position, momentum, grad_lp, log_prob = leapfrog(current_position, momentum, step_size, log_prob, grad_lp, log_prob_fn=log_prob_fn, num_steps=num_steps)

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


@partial(jit, static_argnames=("log_prob_fn", "num_steps", "num_samples", "burn_in"))
def hmc_run(key: Array, log_prob_fn: LogProbFn, init_position: Array, step_size: float, num_steps: int, num_samples: int, burn_in: int = 0) -> Tuple[Array, HMCState]:

    init_state = hmc_init(init_position, log_prob_fn)
    n_chains, n_dim = init_state.position.shape
    
    if burn_in > 0:
        def burnin_step(carry, t):
            key, state = carry
            key, new_state = hmc_step(state, step_size, num_steps, key, log_prob_fn)
            return (key, new_state), None
        
        # key_burnin, key_sampling = random.split(key)
        (key, init_state), _ = lax.scan(burnin_step, (key, init_state), jnp.arange(burn_in))

        state = HMCState(
            position=init_state.position,
            log_prob=init_state.log_prob,
            grad_log_prob=init_state.grad_log_prob,
            accept_count=jnp.zeros(n_chains, dtype=jnp.int32)
        )
        # key = key_sampling
    
    # post burn-in
    samples = jnp.zeros((num_samples, n_chains, n_dim))
    lps = jnp.zeros((num_samples, n_chains))

    def step(carry, t):
        key, state, samples, lps = carry
        key, new_state = hmc_step(state, step_size, num_steps, key, log_prob_fn)
        samples = samples.at[t].set(new_state.position)
        lps = lps.at[t].set(new_state.log_prob)
        return (key, new_state, samples, lps), None

    (key, final_state, samples, lps), _ = lax.scan(step, (key, state, samples, lps), jnp.arange(num_samples))

    accept_rate = final_state.accept_count.astype(jnp.float32) / num_samples
    return samples, lps, final_state, accept_rate



if __name__ == "__main__":

    key = random.PRNGKey(30)
    D = 10        
    n_chains = 4
    burn_in = 1000
    num_samples = 4000

    # HMC hyperparams
    step_size = 0.25
    num_steps = 10      

    # init positions (batched): (chains, D)
    init = jnp.zeros((n_chains, D))

    # target
    log_prob_fn = standard_normal_log_prob 

    samples, lps, final_state, accept_rate = hmc_run(
        key=key,
        log_prob_fn=log_prob_fn,
        init_position=init,
        step_size=step_size,
        num_steps=num_steps,
        num_samples=num_samples,
        burn_in=burn_in,
    )
    print(f"Mean acceptance rate: {np.asarray(accept_rate).mean():.3f}")
    print(f"Per-chain acceptance: {np.asarray(accept_rate)}")


    samples_np = np.asarray(samples).transpose(1, 0, 2)  # (C, T, D)

    idata = az.from_dict(
        posterior={"x": samples_np},
        coords={"x_dim": np.arange(D)},
        dims={"x": ["x_dim"]},
    )


    print(az.summary(idata, var_names=["x"], coords={"x_dim": slice(0, 5)}, kind="all"))

    az.plot_trace(idata, var_names=["x"], coords={"x_dim": slice(0, 3)})
    plt.show()

    az.plot_autocorr(idata, var_names=["x"], coords={"x_dim": slice(0, 3)})
    plt.show()

