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
    position: Array        # float32: (n_chains, n_dim)
    log_prob: Array        # float64: (n_chains,)
    grad_log_prob: Array   # float32: (n_chains, n_dim)
    accept_count: Array    # int32: (n_chains,)

# Energy calculations involve subtracting large numbers to get small differences, and in 
# high dimensions, errors accumulates. In MCMC, acceptance probability depends on these 
# tiny differences, so we use higher precision (float64) for log_prob to improve stability.


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
        body, (position, momentum, log_prob, grad_log_prob), length=num_steps
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
    pos_dtype = state.position.dtype
    logprob_dtype = state.log_prob.dtype

    # next_key, k_mom, k_acc = random.split(key, 3)
    key, step_key = random.split(key)
    k_mom, k_acc = random.split(step_key, 2)

    p0 = random.normal(k_mom, shape=(n_chains, n_dim), dtype=pos_dtype)

    # initial energies
    # half = jnp.array(0.5, dtype=pos_dtype)
    kin0 = 0.5 * jnp.sum(p0**2, axis=-1) # JAX will broadcast to the right type
    # H0 = -state.log_prob + kin0    mixed dtypes!
    # state.log_prob: float64, kin0: pos_dtype (float32)
    H0 = -state.log_prob + kin0.astype(logprob_dtype)

    L1 = num_steps // 2
    L2 = num_steps - L1  # allow odd L (extra step in the second half)

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

    pos_type = state.position.dtype
    lp_type = state.log_prob.dtype

    eps = jnp.asarray(step_size, dtype=pos_type)
    gam = jnp.asarray(gamma, dtype=pos_type)

    # burn-in
    if burn_in > 0:
        def burn_body(carry, _):
            k, s = carry
            k, s = rahmc_step(s, eps, num_steps, gam, k, log_prob_fn)
            return (k, s), None
        (key, state), _ = lax.scan(burn_body, (key, state), length=burn_in)
        # reset accept counter instead of manually reconstructing state
        state = state._replace(accept_count=jnp.zeros(n_chains, dtype=jnp.int32))

    # samples = jnp.zeros((num_samples, n_chains, n_dim), dtype=pos_type)
    # lps = jnp.zeros((num_samples, n_chains), dtype=lp_type)

    ## This is extremely inefficient memory usage, we are carrying the entire sample arrays!
    # def body(carry, t):
    #     k, s, xs, lvals = carry
    #     k, s = rahmc_step(s, eps, num_steps, gam, k, log_prob_fn)
    #     xs = xs.at[t].set(s.position)
    #     lvals = lvals.at[t].set(s.log_prob)
    #     return (k, s, xs, lvals), None
    # (key, state, samples, lps), _ = lax.scan(body, (key, state, samples, lps), jnp.arange(num_samples)) 
    ## can't use length= here since the loop body depends on the iteration index

    def body(carry, _):
        k, s = carry
        k, s = rahmc_step(s, eps, num_steps, gam, k, log_prob_fn)
        return (k, s), (s.position, s.log_prob)
    
    (key, state), (samples, lps) = lax.scan(body, (key, state), length=num_samples)
    # JAX automatically stacks the outputs
    
    accept_rate = state.accept_count.astype(jnp.float32) / num_samples
    return samples, lps, accept_rate, state



# if __name__ == "__main__":
#     key = random.PRNGKey(30)
#     d = 10
#     n_chains = 4
#     burn_in = 1000
#     num_samples = 4000

#     step_size = 0.25
#     num_steps = 10
#     gamma = 0.5

#     init = jnp.zeros((n_chains, d))
#     log_prob_fn = standard_normal_log_prob

#     samples, lps, accept_rate, final_state = rahmc_run(
#         key=key,
#         log_prob_fn=log_prob_fn,
#         init_position=init,
#         step_size=step_size,
#         num_steps=num_steps,
#         gamma=gamma,
#         num_samples=num_samples,
#         burn_in=burn_in,
#     )

#     print(f"Mean acceptance rate: {np.asarray(accept_rate).mean():.3f}")
#     print(f"Per-chain acceptance: {np.asarray(accept_rate)}")

#     samples_np = np.asarray(samples).transpose(1, 0, 2)
#     idata = az.from_dict(
#         posterior={"x": samples_np},
#         coords={"x_dim": np.arange(d)},
#         dims={"x": ["x_dim"]},
#     )
#     print(az.summary(idata, var_names=["x"], coords={"x_dim": slice(0, 5)}, kind="all"))
#     az.plot_trace(idata, var_names=["x"], coords={"x_dim": slice(0, 3)}); plt.show()
#     az.plot_autocorr(idata, var_names=["x"], coords={"x_dim": slice(0, 3)}); plt.show()


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("RA-HMC Test: 10D Standard Gaussian")
    print("=" * 60)
    
    # Setup
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

    # Run sampler (first run compiles)
    print(f"\nRunning RA-HMC:")
    print(f"  - Chains: {n_chains}")
    print(f"  - Dimensions: {d}")
    print(f"  - Burn-in: {burn_in}")
    print(f"  - Samples: {num_samples}")
    print(f"  - Step size: {step_size}, Steps: {num_steps}, Gamma: {gamma}")
    print("\nCompiling... (first run takes ~10-30s)")
    
    start = time.time()
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
    elapsed = time.time() - start
    
    print(f"✓ Done in {elapsed:.2f}s")
    
    # Basic diagnostics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nAcceptance rates:")
    print(f"  Mean: {np.asarray(accept_rate).mean():.3f}")
    print(f"  Per-chain: {np.asarray(accept_rate)}")
    
    # Check if acceptance rate is reasonable
    mean_acc = np.asarray(accept_rate).mean()
    if mean_acc < 0.3:
        print("  ⚠️  Warning: Low acceptance rate (<30%)")
    elif mean_acc > 0.9:
        print("  ⚠️  Warning: High acceptance rate (>90%)")
    else:
        print("  ✓ Acceptance rate looks good!")
    
    # Sample statistics
    samples_np = np.asarray(samples).transpose(1, 0, 2)
    sample_mean = samples_np.mean(axis=(0, 1))
    sample_std = samples_np.std(axis=(0, 1))
    
    print(f"\nSample statistics (first 5 dimensions):")
    print(f"  Mean: {sample_mean[:5]} (should be ≈ 0)")
    print(f"  Std:  {sample_std[:5]} (should be ≈ 1)")
    
    # ArviZ diagnostics
    print("\n" + "=" * 60)
    print("ARVIZ DIAGNOSTICS")
    print("=" * 60)
    
    idata = az.from_dict(
        posterior={"x": samples_np},
        coords={"x_dim": np.arange(d)},
        dims={"x": ["x_dim"]},
    )
    
    print("\nSummary statistics (first 5 dimensions):")
    print(az.summary(idata, var_names=["x"], coords={"x_dim": slice(0, 5)}, kind="stats"))
    
    # Check convergence
    summary = az.summary(idata, var_names=["x"])
    max_rhat = summary['r_hat'].max()
    min_ess = summary['ess_bulk'].min()
    
    print(f"\nConvergence diagnostics:")
    print(f"  Max R-hat: {max_rhat:.4f} (should be <1.01)")
    print(f"  Min ESS: {min_ess:.0f} (should be >400)")
    
    if max_rhat < 1.01 and min_ess > 400:
        print("  ✓ Chains converged successfully!")
    else:
        print("  ⚠️  Warning: Convergence issues detected")
    
    # Plots
    print("\nGenerating plots...")
    az.plot_trace(idata, var_names=["x"], coords={"x_dim": slice(0, 3)})
    plt.suptitle("Trace Plots (first 3 dimensions)")
    plt.tight_layout()
    plt.show()
    
    az.plot_autocorr(idata, var_names=["x"], coords={"x_dim": slice(0, 3)})
    plt.suptitle("Autocorrelation (first 3 dimensions)")
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Test complete! ✓")
    print("=" * 60)