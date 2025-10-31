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
FrictionScheduleFn = Callable[[float, float], float] # (t, T) -> gamma(t)


class RAHMCState(NamedTuple):
    position: Array        # float32: (n_chains, n_dim)
    log_prob: Array        # float64: (n_chains,)
    grad_log_prob: Array   # float32: (n_chains, n_dim)
    accept_count: Array    # int32: (n_chains,)


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

# ============================================================================
# Friction Schedule Functions
# ============================================================================

def constant_friction(gamma: float) -> FrictionScheduleFn:
    """Original RAHMC: -gammafor the first half, +gamma for the second half."""
    def schedule(t: float, T: float) -> float:
        return jnp.where(t < T/2, -gamma, +gamma)
    return schedule

def tanh_friction(gamma_max: float, steepness: float = 5.0) -> FrictionScheduleFn:
    """Smooth tanh friction schedule from -gamma_max to +gamma_max."""
    def schedule(t: float, T: float) -> float:
        normalized_t = steepness * (2.0 * t / T - 1.0)
        return gamma_max * jnp.tanh(normalized_t)
    return schedule

def sigmoid_friction(gamma_max: float, steepness: float = 10.0) -> FrictionScheduleFn:
    """Smooth sigmoid friction schedule from -gamma_max to +gamma_max."""
    def schedule(t: float, T: float) -> float:
        normalized_t = steepness * (t / T - 0.5)
        return gamma_max * (2.0 / (1.0 + jnp.exp(-normalized_t)) - 1.0)
    return schedule

def linear_friction(gamma_max: float) -> FrictionScheduleFn:
    """Linear friction schedule from -gamma_max to +gamma_max."""
    def schedule(t: float, T: float) -> float:
        return -gamma_max + (2.0 * gamma_max * t / T)
    return schedule

def sine_friction(gamma_max: float) -> FrictionScheduleFn:
    """Sine friction schedule from -gamma_max to +gamma_max."""
    def schedule(t: float, T: float) -> float:
        return gamma_max * jnp.sin(jnp.pi * (t / T - 0.5))
    return schedule

def interpolate_schedules(schedule1, schedule2, alpha=0.5):
    """Blend two friction schedules."""
    def blended(t, T):
        return alpha * schedule1(t, T) + (1-alpha) * schedule2(t, T)
    return blended

# ============================================================================


def rahmc_init(init_position: Array, log_prob_fn: LogProbFn) -> RAHMCState:
    pos, _ = _ensure_batched(init_position)
    n_chains = pos.shape[0]
    pos_dtype = pos.dtype
    log_prob, grad_log_prob = vmap(jax.value_and_grad(log_prob_fn))(pos)
    log_prob = log_prob.astype(jnp.float64) # higher precision for log_prob for stability
    grad_log_prob = grad_log_prob.astype(pos_dtype)
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
    pos_dtype = position.dtype
    lp_dtype = log_prob.dtype
    eps = jnp.asarray(step_size, dtype=pos_dtype)
    gam = jnp.asarray(gamma, dtype=pos_dtype)

    half_eps = jnp.array(0.5, dtype=pos_dtype) * eps
    scale = jnp.exp(-gam * half_eps)

    # apply friction scaling
    momentum = momentum * scale
    # half kick  
    momentum = momentum + half_eps * grad_log_prob
    # drift
    position = position + eps * momentum
    # refresh grads
    new_lp, new_grad_lp = vmap(jax.value_and_grad(log_prob_fn))(position)
    new_lp = new_lp.astype(lp_dtype)
    new_grad_lp = new_grad_lp.astype(pos_dtype)
    # half kick
    momentum = momentum + half_eps * new_grad_lp
    # apply friction scaling
    momentum = momentum * scale

    return position, momentum, new_lp, new_grad_lp


# Half-trajectory scan (repelling or attracting)
@partial(jit, static_argnames=("num_steps", "log_prob_fn", "friction_schedule"))
def _trajectory_with_schedule(
    position: Array,
    momentum: Array,
    step_size: float,
    gamma_max: float,
    log_prob: Array,
    grad_log_prob: Array,
    num_steps: int,
    time_offset: float, # starting time for this half-trajectory
    total_time: float, # total trajectory length T
    log_prob_fn: LogProbFn,
    friction_schedule: FrictionScheduleFn,
):
    def body(carry, step_idx):
        q, p, lp, glp = carry
        
        # compute current time and friction at this time
        current_time = time_offset + step_idx * step_size
        gamma_t = friction_schedule(current_time, total_time)

        q, p, lp, glp = _conformal_leapfrog_step(
            q, p, step_size, gamma_t, lp, glp, log_prob_fn
        )
        return (q, p, lp, glp), None

    (q, p, lp, glp), _ = lax.scan(
        body, (position, momentum, log_prob, grad_log_prob), jnp.arange(num_steps)
    )
    return q, p, lp, glp


@partial(jit, static_argnames=("num_steps", "log_prob_fn", "friction_schedule"))
def rahmc_step(
    state: RAHMCState,
    step_size: float,
    num_steps: int,
    gamma_max: float,
    key: Array,
    log_prob_fn: LogProbFn,
    friction_schedule: FrictionScheduleFn = None,
) -> Tuple[Array, RAHMCState]:
    """
    RA-HMC step:
      - sample p ~ N(0, I)
      - first ⌊L/2⌋ conformal steps with gamma_rep = -gamma (repelling)
      - then ⌊L/2⌋ steps with gamma_att = +gamma (attracting)
      - flip momentum
      - MH accept/reject (Jacobian=1)
    """
    if friction_schedule is None:
        friction_schedule = constant_friction(gamma_max)

    n_chains, n_dim = state.position.shape
    pos_dtype = state.position.dtype
    logprob_dtype = state.log_prob.dtype

    # next_key, k_mom, k_acc = random.split(key, 3)
    key, step_key = random.split(key)
    k_mom, k_acc = random.split(step_key, 2)

    p0 = random.normal(k_mom, shape=(n_chains, n_dim), dtype=pos_dtype)

    kin0 = 0.5 * jnp.sum(p0**2, axis=-1) # JAX will broadcast to the right type
    H0 = -state.log_prob + kin0.astype(logprob_dtype)

    total_time = step_size * num_steps

    q, p, lp, glp = _trajectory_with_schedule(
        state.position, p0, step_size, gamma_max, 
        state.log_prob, state.grad_log_prob,
        num_steps, time_offset=0.0, total_time=total_time,
        log_prob_fn=log_prob_fn, friction_schedule=friction_schedule,
    )

    # flip momentum
    p = -p

    # compute final energies
    kin1 = 0.5 * jnp.sum(p**2, axis=-1)
    H1 = -lp + kin1.astype(logprob_dtype)

    # add overflow protection
    H1 = jnp.where(jnp.isfinite(H1), H1, jnp.array(1e10, dtype=logprob_dtype))
    # in extreme low-density region, H1 can become NaN or Inf.
    # we set it to a large number so log_alpha becomes very negative and proposal is rejected.

    # MH test
    log_alpha = H0 - H1 # sensitive, in float64

    u = random.uniform(k_acc, shape=(n_chains,), dtype=logprob_dtype)
    accept = jnp.log(u) < jnp.minimum(0.0, log_alpha)

    new_pos = jnp.where(accept[:, None], q, state.position)
    new_lp = jnp.where(accept, lp, state.log_prob)
    new_glp = jnp.where(accept[:, None], glp, state.grad_log_prob)
    new_acc = state.accept_count + accept.astype(jnp.int32)

    new_state = RAHMCState(new_pos, new_lp, new_glp, new_acc)
    return key, new_state # return fresh key


@partial(jit, static_argnames=("log_prob_fn","friction_schedule", "num_steps", "num_samples", "burn_in"))
def rahmc_run(
    key: Array,
    log_prob_fn: LogProbFn,
    init_position: Array,
    step_size: float,
    num_steps: int,
    gamma: float,
    num_samples: int,
    burn_in: int = 0,
    friction_schedule: FrictionScheduleFn = None,
) -> Tuple[Array, Array, Array, RAHMCState]:
    if friction_schedule is None:
        friction_schedule = constant_friction(gamma)
    state = rahmc_init(init_position, log_prob_fn)
    n_chains, n_dim = state.position.shape

    pos_type = state.position.dtype
    lp_type = state.log_prob.dtype

    eps = jnp.asarray(step_size, dtype=pos_type)
    gam = jnp.asarray(gamma, dtype=pos_type)

    # burn-in
    if burn_in > 0:
        def burn_body(carry, _):
            k, s = carry
            k, s = rahmc_step(s, eps, num_steps, gam, k, log_prob_fn, friction_schedule)
            return (k, s), None
        (key, state), _ = lax.scan(burn_body, (key, state), length=burn_in)
        # reset accept counter instead of manually reconstructing state
        state = state._replace(accept_count=jnp.zeros(n_chains, dtype=jnp.int32))

    def body(carry, _):
        k, s = carry
        k, s = rahmc_step(s, eps, num_steps, gam, k, log_prob_fn, friction_schedule)
        return (k, s), (s.position, s.log_prob)
    
    (key, state), (samples, lps) = lax.scan(body, (key, state), length=num_samples)
    # JAX automatically stacks the outputs
    
    accept_rate = state.accept_count.astype(jnp.float32) / num_samples
    return samples, lps, accept_rate, state

