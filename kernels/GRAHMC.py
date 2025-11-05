from __future__ import annotations
from typing import Callable, Tuple, NamedTuple
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random, vmap, lax

# from utils import Array, LogProbFn, FrictionScheduleFn, _ensure_batched
Array = jnp.ndarray
LogProbFn = Callable[[Array], Array]  # x -> log p(x)
FrictionScheduleFn = Callable[[float, float, float, float | None], float] # (t, T, gamma_max, steepness) -> gamma(t)


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


class RAHMCState(NamedTuple):
    position: Array        # float32: (n_chains, n_dim)
    log_prob: Array        # float64: (n_chains,)
    grad_log_prob: Array   # float32: (n_chains, n_dim)
    accept_count: Array    # int32: (n_chains,)


# ============================================================================
# Friction Schedule Functions
# ============================================================================

# This version creates a new function object every time. It's a closure that captures gamma and steepness.
# def tanh_friction(gamma_max: float, steepness: float = 5.0) -> FrictionScheduleFn:
#     """Smooth tanh friction schedule from -gamma_max to +gamma_max."""
#     def schedule(t: float, T: float) -> float:
#         normalized_t = steepness * (2.0 * t / T - 1.0)
#         return gamma_max * jnp.tanh(normalized_t)
#     return schedule

# parametrized friction schedule functions
def constant_schedule(t: float, T: float, gamma: float, steepness: float = None) -> float:
    # steepness unused for constant
    return jnp.where(t < T/2, -gamma, +gamma)

def tanh_schedule(t: float, T: float, gamma_max: float, steepness: float = 5.0) -> float:
    normalized_t = steepness * (2.0 * t / T - 1.0)
    return gamma_max * jnp.tanh(normalized_t)

def sigmoid_schedule(t: float, T: float, gamma_max: float, steepness: float = 10.0) -> float:
    normalized_t = steepness * (t / T - 0.5)
    return gamma_max * (2.0 / (1.0 + jnp.exp(-normalized_t)) - 1.0)

def linear_schedule(t: float, T: float, gamma_max: float, steepness: float = None) -> float:
    # steepness unused for linear
    return -gamma_max + (2.0 * gamma_max * t / T)

def sine_schedule(t: float, T: float, gamma_max: float, steepness: float = None) -> float:
    # steepness unused for sine
    return gamma_max * jnp.sin(jnp.pi * (t / T - 0.5))

def constant_schedule_default(t: float, T: float) -> float:
    """Constant schedule with default gamma=1.0 for plotting."""
    return constant_schedule(t, T, gamma=1.0, steepness=None)

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


@partial(jit, static_argnames=("log_prob_fn", "friction_schedule", "num_steps")) 
def _trajectory_with_schedule(
    position: Array,
    momentum: Array,
    step_size: float,
    gamma_max: float,
    steepness: float,
    log_prob: Array,
    grad_log_prob: Array,
    num_steps: int,
    # time_offset: float, # starting time for this half-trajectory
    # total_time: float, # total trajectory length T
    # these two are not necessary since our friction function's behavior is dictated by t/T.
    log_prob_fn: LogProbFn,
    friction_schedule: FrictionScheduleFn,
):
    """Run a full leapfrog trajectory with time-dependent friction schedule."""
    total_time = step_size * num_steps

    def body(carry, step_idx):
        q, p, lp, glp = carry
        # compute current time and friction at this time
        current_time = step_idx * step_size
        gamma_t = friction_schedule(current_time, total_time, gamma_max, steepness)
        q, p, lp, glp = _conformal_leapfrog_step(
            q, p, step_size, gamma_t, lp, glp, log_prob_fn
        )
        return (q, p, lp, glp), None

    (q, p, lp, glp), _ = lax.scan(
        body, (position, momentum, log_prob, grad_log_prob), jnp.arange(num_steps)
    )
    return q, p, lp, glp


@partial(jit, static_argnames=("log_prob_fn", "friction_schedule", "num_steps", "return_proposal"))
def rahmc_step(
    state: RAHMCState,
    step_size: float,
    num_steps: int,
    gamma_max: float,
    steepness: float,
    key: Array,
    log_prob_fn: LogProbFn,
    friction_schedule: FrictionScheduleFn = None,
    return_proposal: bool = False,
) -> Tuple[Array, RAHMCState]:

    if friction_schedule is None:
        friction_schedule = constant_schedule

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
        state.position, p0, step_size, gamma_max, steepness,
        state.log_prob, state.grad_log_prob,
        num_steps, 
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
    delta_H = H1 - H0

    u = random.uniform(k_acc, shape=(n_chains,), dtype=logprob_dtype)
    accept = jnp.log(u) < jnp.minimum(0.0, log_alpha)

    new_pos = jnp.where(accept[:, None], q, state.position)
    new_lp = jnp.where(accept, lp, state.log_prob)
    new_glp = jnp.where(accept[:, None], glp, state.grad_log_prob)
    new_acc = state.accept_count + accept.astype(jnp.int32)

    new_state = RAHMCState(new_pos, new_lp, new_glp, new_acc)
    
    if return_proposal:
        return key, new_state, q, lp, delta_H
    else:
        return key, new_state


@partial(jit, static_argnames=("log_prob_fn", "num_samples", "burn_in", "friction_schedule", "track_proposals", "num_steps"))
def rahmc_run(
    key: Array,
    log_prob_fn: LogProbFn,
    init_position: Array,
    step_size: float,
    num_steps: int,
    gamma: float,
    steepness: float,
    num_samples: int,
    burn_in: int = 0,
    friction_schedule: FrictionScheduleFn = None,
    track_proposals: bool = False,
) -> Tuple:
    
    if friction_schedule is None:
        friction_schedule = constant_schedule

    state = rahmc_init(init_position, log_prob_fn)
    n_chains = state.position.shape[0]

    pos_type = state.position.dtype

    eps = jnp.asarray(step_size, dtype=pos_type)
    gam = jnp.asarray(gamma, dtype=pos_type)
    steep = jnp.asarray(steepness, dtype=pos_type)

    # burn-in
    if burn_in > 0:
        def burn_body(carry, _):
            k, s = carry
            k, s = rahmc_step(s, eps, num_steps, gam, steep, k, log_prob_fn, friction_schedule, return_proposal=False)
            return (k, s), None
        (key, state), _ = lax.scan(burn_body, (key, state), length=burn_in)
        # reset accept counter instead of manually reconstructing state
        state = state._replace(accept_count=jnp.zeros(n_chains, dtype=jnp.int32))

    # sampling
    if track_proposals:
        def body_with_proposals(carry, _):
            k, s = carry
            pre_pos, pre_lp = s.position, s.log_prob
            k, s, prop_pos, prop_lp, delta_H = rahmc_step(
                s, eps, num_steps, gam, steep, k, log_prob_fn, friction_schedule, return_proposal=True
            )
            return (k, s), (pre_pos, pre_lp, prop_pos, prop_lp, delta_H, s.position, s.log_prob)
        
        # (key, state), (samples, lps, prop_positions, prop_lps, delta_H) = lax.scan(
        #     body_with_proposals, (key, state), length=num_samples
        # )
        (key, state), (pre_positions, pre_lps, prop_positions, prop_lps, deltas_H, post_positions, post_lps) = lax.scan(
            body_with_proposals, (key, state), length=num_samples
        )
        
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples

        return (
            post_positions, post_lps,          # samples, lps (post-MH)
            accept_rate, state,                # accept stats + final state
            pre_positions, pre_lps,            # pre-step state (for ESJD)
            prop_positions, prop_lps,          # proposals
            deltas_H
        )
    else:
        def body(carry, _):
            k, s = carry
            k, s = rahmc_step(s, eps, num_steps, gam, steep, k, log_prob_fn, friction_schedule, return_proposal=False)
            return (k, s), (s.position, s.log_prob)
        
        (key, state), (samples, lps) = lax.scan(body, (key, state), length=num_samples)
        
        accept_rate = state.accept_count.astype(jnp.float32) / num_samples
        return samples, lps, accept_rate, state



# ============================================================================

# =========================
# PROPOSAL TRACE + PLOTTING
# =========================

def rahmc_proposal_trace(
    key: Array,
    log_prob_fn: LogProbFn,
    q0: Array,
    step_size: float,
    num_steps: int,
    gamma_max: float,
    friction_schedule: FrictionScheduleFn | None = None,
    resample_p_between_segments: bool = False,
):
    """
    Run ONE raHMC proposal (no MH decision) and record per-substep (q,p,U,K,H).
    Uses your _conformal_leapfrog_step and friction_schedule (default constant_friction).
    q0 can be shape (n_dim,) – we trace a single chain for visualization.
    Returns:
      Qs, Ps  : (T+1, d)
      Us, Ks, Hs : (T+1,)
      split_idx : index where repel -> attract boundary occurs in arrays
      resample_idx : index where momentum was resampled (or None)
    """
    if friction_schedule is None:
        friction_schedule = constant_schedule_default

    q = jnp.atleast_1d(q0).astype(jnp.float32)
    d = q.shape[-1]
    key, k_p1, k_p2 = random.split(key, 3)

    # p ~ N(0,I) in position dtype
    p = random.normal(k_p1, shape=(d,), dtype=q.dtype)

    # initial energies (float64 log prob like your code)
    lp, glp = jax.value_and_grad(log_prob_fn)(q)
    lp = lp.astype(jnp.float64)
    glp = glp.astype(q.dtype)
    U = -lp
    K = 0.5 * jnp.sum(p**2).astype(jnp.float64)
    H = U + K

    Qs = [np.array(q)]; Ps = [np.array(p)]
    Us = [float(U)]; Ks = [float(K)]; Hs = [float(H)]

    L1 = int(num_steps // 2)
    L2 = int(num_steps - L1)
    total_time = step_size * num_steps

    # ---- Repelling segment ----
    for i in range(L1):
        t_now = i * step_size
        gamma_t = float(friction_schedule(t_now, total_time))
        q, p, lp, glp = _conformal_leapfrog_step(
            q[None, :], p[None, :], step_size, gamma_t,
            lp[None, ...], glp[None, :],  # batched to match your API
            log_prob_fn
        )
        # un-batch
        q = q[0]; p = p[0]; lp = lp[0]; glp = glp[0]
        U = -lp
        K = 0.5 * jnp.sum(p**2).astype(jnp.float64)
        H = U + K
        Qs.append(np.array(q)); Ps.append(np.array(p))
        Us.append(float(U)); Ks.append(float(K)); Hs.append(float(H))

    split_idx = len(Qs) - 1
    resample_idx = None

    # optional momentum resample between segments
    if resample_p_between_segments:
        resample_idx = len(Qs) - 1
        p = random.normal(k_p2, shape=(d,), dtype=q.dtype)
        U = -lp; K = 0.5 * jnp.sum(p**2).astype(jnp.float64); H = U + K
        Qs.append(np.array(q)); Ps.append(np.array(p))
        Us.append(float(U)); Ks.append(float(K)); Hs.append(float(H))

    # ---- Attracting segment ----
    for j in range(L2):
        t_now = (L1 + j) * step_size
        gamma_t = float(friction_schedule(t_now, total_time))
        q, p, lp, glp = _conformal_leapfrog_step(
            q[None, :], p[None, :], step_size, gamma_t,
            lp[None, ...], glp[None, :],
            log_prob_fn
        )
        q = q[0]; p = p[0]; lp = lp[0]; glp = glp[0]
        U = -lp
        K = 0.5 * jnp.sum(p**2).astype(jnp.float64)
        H = U + K
        Qs.append(np.array(q)); Ps.append(np.array(p))
        Us.append(float(U)); Ks.append(float(K)); Hs.append(float(H))

    return (
        np.stack(Qs), np.stack(Ps),
        np.array(Us), np.array(Ks), np.array(Hs),
        split_idx, resample_idx
    )

# ---------- Plot helpers (auto limits, shading) ----------
def _auto_limits(x, pad_ratio=0.10):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi == lo:
        lo -= 1.0; hi += 1.0
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad

def _shade_segments(ax, split_idx, T, a=0.10):
    ax.axvspan(0, split_idx, color="tab:orange", alpha=a, lw=0)
    ax.axvspan(split_idx, T-1, color="tab:blue", alpha=a, lw=0)

# ===================== 1) PHASE-SPACE (1D) =====================
def plot_phase_space_1d(
    log_prob_fn: LogProbFn,
    q0: float,
    key: Array = random.PRNGKey(0),
    step_size: float = 0.15,
    num_steps: int = 48,
    gamma: float = 1.3,
    friction_schedule: FrictionScheduleFn | None = None,
    resample_between: bool = False,
    title: str = "Phase-space (1D)"
):
    if friction_schedule is None:
        friction_schedule = constant_schedule_default

    Qs, Ps, Us, Ks, Hs, split_idx, res_idx = rahmc_proposal_trace(
        key, log_prob_fn, jnp.array([q0], dtype=jnp.float32),
        step_size, num_steps, gamma, friction_schedule,
        resample_p_between_segments=resample_between
    )
    q = Qs[:, 0]; p = Ps[:, 0]; T = len(q)
    colors = np.array(["tab:orange"] * (split_idx + 1) + ["tab:blue"] * (T - (split_idx + 1)))
    if res_idx is not None:
        colors[res_idx] = "0.25"

    # background H-level sets (local grid around path)
    qlo, qhi = _auto_limits(q, 0.12)
    plo, phi = _auto_limits(p, 0.12)
    qg = np.linspace(qlo, qhi, 180)
    pg = np.linspace(plo, phi, 180)
    Q, P = np.meshgrid(qg, pg)
    v_lp = jax.vmap(lambda z: log_prob_fn(jnp.array([z])))
    Ugrid = -np.array(v_lp(jnp.array(qg)))
    Hgrid = Ugrid[None, :] + 0.5 * (P**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.contour(Q, P, Hgrid, levels=12, colors="0.85", linewidths=0.8)
    ax1.plot(q, p, lw=1.4, color="0.35")
    step = max(1, T // 28)
    ax1.quiver(q[:-step:step], p[:-step:step],
               q[step::step]-q[:-step:step], p[step::step]-p[:-step:step],
               angles="xy", scale_units="xy", scale=1.0, color="0.35", width=0.004, alpha=0.7)
    ax1.scatter(q, p, c=colors, s=20, zorder=3)
    ax1.scatter([q[0]], [p[0]], c="white", edgecolors="k", s=90, zorder=5, label="start")
    ax1.scatter([q[-1]], [p[-1]], c="k", s=55, zorder=5, label="end")
    if res_idx is not None:
        ax1.scatter([q[res_idx]], [p[res_idx]], c="none", edgecolors="k", s=85, marker="D", label="resample")
    ax1.set_xlim(qlo, qhi); ax1.set_ylim(plo, phi)
    ax1.set_xlabel(r"$q$"); ax1.set_ylabel(r"$p$"); ax1.set_title(title)
    ax1.legend(frameon=False); ax1.grid(alpha=0.2)

    U, K, H = np.array(Us), np.array(Ks), np.array(Hs)
    H0 = np.min(H)  # offset to compress dynamic range
    ax2.plot(U - H0, label="U (offset)")
    ax2.plot(K - H0, label="K (offset)")
    ax2.plot(H - H0, label="H (offset)")
    ax2.axvline(split_idx, ls="--", lw=1.0, color="0.3")
    _shade_segments(ax2, split_idx, T, a=0.10)
    ax2.set_title("Energies per substep"); ax2.legend(frameon=False); ax2.grid(alpha=0.2)
    plt.tight_layout(); plt.show()

# ====================== 2) Q-SPACE (2D) ======================
def _contour_U_2d(ax, log_prob_fn, bounds, n=220):
    xs = np.linspace(bounds[0], bounds[1], n)
    ys = np.linspace(bounds[2], bounds[3], n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    v_lp2 = jax.vmap(lambda v: log_prob_fn(jnp.array(v)))
    U = -np.array(v_lp2(jnp.array(grid))).reshape(n, n)
    cs = ax.contour(X, Y, U, levels=18, linewidths=0.6, colors="0.65", alpha=0.95)
    return cs

def plot_q_space_2d(
    log_prob_fn: LogProbFn,
    q0: tuple[float, float],
    key: Array = random.PRNGKey(0),
    step_size: float = 0.12,
    num_steps: int = 60,
    gamma: float = 1.2,
    friction_schedule: FrictionScheduleFn | None = None,
    resample_between: bool = False,
    bounds: tuple[float, float, float, float] | None = None,
    title: str = "q-space (2D)"
):
    if friction_schedule is None:
        friction_schedule = constant_schedule_default

    Qs, Ps, Us, Ks, Hs, split_idx, res_idx = rahmc_proposal_trace(
        key, log_prob_fn, jnp.array(q0, dtype=jnp.float32),
        step_size, num_steps, gamma, friction_schedule,
        resample_p_between_segments=resample_between
    )
    q1, q2 = Qs[:, 0], Qs[:, 1]
    if bounds is None:
        xlo, xhi = _auto_limits(q1, 0.10); ylo, yhi = _auto_limits(q2, 0.10)
        bounds = (xlo, xhi, ylo, yhi)

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 5.4))
    _contour_U_2d(ax, log_prob_fn, bounds=bounds)

    colors = np.array(["tab:orange"] * (split_idx + 1) + ["tab:blue"] * (len(q1) - (split_idx + 1)))
    if res_idx is not None:
        colors[res_idx] = "0.25"

    ax.plot(q1, q2, lw=1.7, color="0.35", alpha=0.75)
    step = max(1, len(q1) // 30)
    ax.quiver(q1[:-step:step], q2[:-step:step],
              q1[step::step]-q1[:-step:step], q2[step::step]-q2[:-step:step],
              angles="xy", scale_units="xy", scale=1.0, width=0.0035, color="0.35", alpha=0.7)
    ax.scatter(q1, q2, c=colors, s=22, zorder=3)
    ax.scatter([q1[0]], [q2[0]], c="lime", edgecolors="k", s=90, zorder=5, label="start")
    ax.scatter([q1[-1]], [q2[-1]], c="red", edgecolors="k", s=55, zorder=5, label="end")
    if res_idx is not None:
        ax.scatter([q1[res_idx]], [q2[res_idx]], c="none", edgecolors="k", s=85, marker="D", label="resample")
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel(r"$q_1$"); ax.set_ylabel(r"$q_2$"); ax.set_title(title)
    ax.legend(frameon=False, loc="best"); ax.grid(alpha=0.15)
    plt.tight_layout(); plt.show()

# ----------------- Tiny example bimodal targets (optional) -----------------
def lp_1d_bimodal(q, m=2.5, s=0.6):
    q = jnp.atleast_1d(q); x = q[0]
    terms = jnp.stack([
        -0.5*((x - m)/s)**2 - jnp.log(s),
        -0.5*((x + m)/s)**2 - jnp.log(s),
    ])
    return jax.scipy.special.logsumexp(terms + jnp.log(0.5))

def lp_2d_bimodal(q):
    q = jnp.atleast_1d(q); x, y = q[0], q[1]
    def logN(mx, my, sx=0.7, sy=0.7, rho=0.0):
        inv = 1.0/(1 - rho**2)
        qf = inv*(((x-mx)/sx)**2 - 2*rho*((x-mx)/sx)*((y-my)/sy) + ((y-my)/sy)**2)
        return -0.5*qf - jnp.log(sx*sy)
    c1 = logN(-2.0, -1.2, rho= 0.2)
    c2 = logN(+2.0, +1.2, rho=-0.2)
    return jax.scipy.special.logsumexp(jnp.array([c1, c2]) + jnp.log(0.5))

# ----------------------------- Quick demos ---------------------------------
if __name__ == "__main__":
    k = random.PRNGKey(0)

    # 1D phase-space (unimodal & bimodal)
    plot_phase_space_1d(standard_normal_log_prob, 0.5, k, step_size=0.18, num_steps=48, gamma=1.3,
                        title="Phase-space (1D unimodal, constant γ)")
    plot_phase_space_1d(lp_1d_bimodal, -1.2, k, step_size=0.16, num_steps=56, gamma=1.5,
                        title="Phase-space (1D bimodal, constant γ)")

    # 2D q-space (unimodal & bimodal)
    def lp_2d_unimodal(q):
        # correlated normal example
        q = jnp.atleast_1d(q); x, y = q[0], q[1]
        rho = 0.6; inv = 1.0/(1 - rho**2)
        quad = inv * (x**2 - 2*rho*x*y + y**2)
        return -0.5 * quad

    plot_q_space_2d(lp_2d_unimodal, [-2.0, 1.0], k, step_size=0.12, num_steps=60, gamma=1.2,
                    title="q-space (2D unimodal, constant γ)")
    plot_q_space_2d(lp_2d_bimodal, [-2.0, -1.0], k, step_size=0.12, num_steps=70, gamma=1.4,
                    title="q-space (2D bimodal, constant γ)")
