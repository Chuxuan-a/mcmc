from __future__ import annotations
from typing import Callable, Tuple, NamedTuple
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random, vmap, lax

# type aliases
Array = jnp.ndarray
LogProbFn = Callable[[Array], Array]  # x -> log p(x)


class RAHMCState(NamedTuple):
    position: Array        # (n_chains, n_dim)
    log_prob: Array        # (n_chains,)
    grad_log_prob: Array   # (n_chains, n_dim)
    accept_count: Array    # (n_chains,)


def _ensure_batched(x: Array) -> Tuple[Array, bool]:
    x = jnp.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    elif x.ndim == 2:
        return x, False
    else:
        raise ValueError("Input must have shape (n_dim,) or (n_chains, n_dim).")


def standard_normal_log_prob(x: Array) -> Array:
    x = jnp.asarray(x)
    D = x.shape[-1]
    two_pi = jnp.array(2.0 * jnp.pi, dtype=x.dtype)
    return -0.5 * (jnp.sum(x**2, axis=-1) + D * jnp.log(two_pi))


def rahmc_init(init_position: Array, log_prob_fn: LogProbFn) -> RAHMCState:
    pos, _ = _ensure_batched(init_position)
    n_chains = pos.shape[0]
    log_prob, grad_log_prob = vmap(jax.value_and_grad(log_prob_fn))(pos)
    log_prob = log_prob.astype(pos.dtype)
    grad_log_prob = grad_log_prob.astype(pos.dtype)
    return RAHMCState(
        position=pos,
        log_prob=log_prob,
        grad_log_prob=grad_log_prob,
        accept_count=jnp.zeros(n_chains, dtype=jnp.int32),
    )


@partial(jit, static_argnames=("log_prob_fn",))
def _conformal_leapfrog_step(
    position: Array,
    momentum: Array,
    step_size: float,
    gamma: float,
    log_prob: Array,
    grad_log_prob: Array,
    log_prob_fn: LogProbFn,
):
    """
    One conformal-leapfrog step:
      p <- e^{-gamma*eps/2} [p - (eps/2) * gradU(q)]
      q <- q + eps * p
      p <- e^{-gamma*eps/2} [p - (eps/2) * gradU(q_new)]
    gradU = -grad_log_prob, pass grad_log_prob to avoid recompute.
    """
    pos_dtype = position.dtype
    lp_dtype = log_prob.dtype
    eps = jnp.asarray(step_size, dtype=pos_dtype)
    g = jnp.asarray(gamma, dtype=pos_dtype)

    # use gradU = -grad_log_prob
    half_eps = jnp.array(0.5, dtype=pos_dtype) * eps
    scale = jnp.exp(-g * eps * jnp.array(0.5, dtype=pos_dtype))

    # first half kick + friction scaling
    momentum = (momentum - half_eps * (-grad_log_prob)) * scale
    # drift
    position = position + eps * momentum
    # refresh grads
    new_lp, new_grad_lp = vmap(jax.value_and_grad(log_prob_fn))(position)
    new_lp = new_lp.astype(lp_dtype)
    new_grad_lp = new_grad_lp.astype(pos_dtype)
    # second half kick + friction scaling
    momentum = (momentum - half_eps * (-new_grad_lp)) * scale

    return position.astype(pos_dtype), momentum.astype(pos_dtype), new_lp, new_grad_lp


# Half-trajectory scan (repelling or attracting)
@partial(jit, static_argnames=("num_steps", "log_prob_fn"))
def _half_trajectory(
    position: Array,
    momentum: Array,
    step_size: float,
    gamma: float,
    log_prob: Array,
    grad_log_prob: Array,
    num_steps: int,
    log_prob_fn: LogProbFn,
):
    def body(carry, _):
        q, p, lp, glp = carry
        q, p, lp, glp = _conformal_leapfrog_step(
            q, p, step_size, gamma, lp, glp, log_prob_fn
        )
        return (q, p, lp, glp), None

    (q, p, lp, glp), _ = lax.scan(
        body, (position, momentum, log_prob, grad_log_prob), jnp.arange(num_steps)
    )
    return q, p, lp, glp


@partial(jit, static_argnames=("num_steps", "log_prob_fn"))
def rahmc_step(
    state: RAHMCState,
    step_size: float,
    num_steps: int,
    gamma: float,
    key: Array,
    log_prob_fn: LogProbFn,
) -> Tuple[Array, RAHMCState]:
    """
    RA-HMC step:
      - sample p ~ N(0, I)
      - first ⌊L/2⌋ conformal steps with gamma_rep = -gamma (repelling)
      - then ⌊L/2⌋ steps with gamma_att = +gamma (attracting)
      - flip momentum
      - MH accept/reject (Jacobian=1)
    """
    n_chains, n_dim = state.position.shape
    next_key, k_mom, k_acc = random.split(key, 3)
    pos_dtype = state.position.dtype
    logprob_dtype = state.log_prob.dtype

    p0 = random.normal(k_mom, shape=(n_chains, n_dim), dtype=pos_dtype)

    # initial energies
    half = jnp.array(0.5, dtype=pos_dtype)
    kin0 = half * jnp.sum(p0**2, axis=-1)
    H0 = -state.log_prob + kin0

    L1 = num_steps // 2
    L2 = num_steps - L1  # allow odd L (put extra step in the second half)

    # repelling
    q, p, lp, glp = _half_trajectory(
        state.position, p0, step_size, -gamma, state.log_prob, state.grad_log_prob, L1, log_prob_fn
    )
    # attracting
    q, p, lp, glp = _half_trajectory(
        q, p, step_size, +gamma, lp, glp, L2, log_prob_fn
    )

    # flip momentum
    p = -p

    kin1 = half * jnp.sum(p**2, axis=-1)
    H1 = -lp + kin1

    # MH test
    log_alpha = H0 - H1
    u = random.uniform(k_acc, shape=(n_chains,), dtype=logprob_dtype)
    accept = jnp.log(u) < jnp.minimum(jnp.array(0.0, logprob_dtype), log_alpha)

    new_pos = jnp.where(accept[:, None], q, state.position)
    new_lp = jnp.where(accept, lp, state.log_prob)
    new_glp = jnp.where(accept[:, None], glp, state.grad_log_prob)
    new_acc = state.accept_count + accept.astype(state.accept_count.dtype)

    new_state = RAHMCState(new_pos, new_lp, new_glp, new_acc)
    return next_key, new_state


@partial(jit, static_argnames=("log_prob_fn", "num_steps", "num_samples", "burn_in"))
def rahmc_run(
    key: Array,
    log_prob_fn: LogProbFn,
    init_position: Array,
    step_size: float,
    num_steps: int,
    gamma: float,
    num_samples: int,
    burn_in: int = 0,
) -> Tuple[Array, Array, Array, RAHMCState]:
    state = rahmc_init(init_position, log_prob_fn)
    n_chains, n_dim = state.position.shape
    eps = jnp.asarray(step_size, dtype=state.position.dtype)
    gam = jnp.asarray(gamma, dtype=state.position.dtype)

    # burn-in
    if burn_in > 0:
        def burn_body(carry, _):
            k, s = carry
            k, s = rahmc_step(s, eps, num_steps, gam, k, log_prob_fn)
            return (k, s), None
        (key, state), _ = lax.scan(burn_body, (key, state), jnp.arange(burn_in))
        state = RAHMCState(
            position=state.position,
            log_prob=state.log_prob,
            grad_log_prob=state.grad_log_prob,
            accept_count=jnp.zeros(n_chains, dtype=jnp.int32),
        )

    samples = jnp.zeros((num_samples, n_chains, n_dim), dtype=state.position.dtype)
    lps = jnp.zeros((num_samples, n_chains), dtype=state.log_prob.dtype)

    def body(carry, t):
        k, s, xs, lvals = carry
        k, s = rahmc_step(s, eps, num_steps, gam, k, log_prob_fn)
        xs = xs.at[t].set(s.position)
        lvals = lvals.at[t].set(s.log_prob)
        return (k, s, xs, lvals), None

    (key, state, samples, lps), _ = lax.scan(body, (key, state, samples, lps), jnp.arange(num_samples))

    accept_rate = state.accept_count.astype(jnp.float32) / num_samples
    return samples, lps, accept_rate, state



if __name__ == "__main__":
    key = random.PRNGKey(30)
    d = 10
    n_chains = 4
    burn_in = 1000
    num_samples = 4000

    step_size = 0.25
    num_steps = 10
    gamma = 0.5

    init = jnp.zeros((n_chains, d))
    log_prob_fn = standard_normal_log_prob

    samples, lps, accept_rate, final_state = rahmc_run(
        key=key,
        log_prob_fn=log_prob_fn,
        init_position=init,
        step_size=step_size,
        num_steps=num_steps,
        gamma=gamma,
        num_samples=num_samples,
        burn_in=burn_in,
    )

    print(f"Mean acceptance rate: {np.asarray(accept_rate).mean():.3f}")
    print(f"Per-chain acceptance: {np.asarray(accept_rate)}")

    samples_np = np.asarray(samples).transpose(1, 0, 2)
    idata = az.from_dict(
        posterior={"x": samples_np},
        coords={"x_dim": np.arange(d)},
        dims={"x": ["x_dim"]},
    )
    print(az.summary(idata, var_names=["x"], coords={"x_dim": slice(0, 5)}, kind="all"))
    az.plot_trace(idata, var_names=["x"], coords={"x_dim": slice(0, 3)}); plt.show()
    az.plot_autocorr(idata, var_names=["x"], coords={"x_dim": slice(0, 3)}); plt.show()
