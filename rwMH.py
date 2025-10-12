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
LogProbFn = Callable[[Array], Array] # x -> log p(x)

class RWMState(NamedTuple):
    position: Array   # (n_chains, n_dim)
    log_prob: Array   # (n_chains,)
    accept_count: Array   # (n_chains,)


def standard_normal_log_prob(x: Array) -> Array:
    """log N(0, I) for x with shape (..., D). Returns shape (...,)."""
    D = x.shape[-1]
    # -0.5 * (||x||^2 + D * log(2π))
    return -0.5 * (jnp.sum(x**2, axis=-1) + D * jnp.log(2.0 * jnp.pi))


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


def rwMH_init(logprob_fn: LogProbFn, init_position: Array) -> RWMState:
    """
    Initialize state for Random Walk M-H sampler.
    Ensures batching and computes initial log probabilities.
    """
    pos, _ = _ensure_batched(init_position)
    log_prob = vmap(logprob_fn)(pos)
    return RWMState(position=pos, 
                    log_prob=log_prob, 
                    accept_count=jnp.zeros(pos.shape[0], dtype=jnp.int32))


@partial(jit, static_argnames=("logprob_fn"))
def rwMH_step(key: Array, state: RWMState, logprob_fn: LogProbFn, scale: Array) -> Tuple[Array, RWMState]:
    """Performs one step of the Random Walk M-H algorithm."""
    # Current batch shape
    n_chains, D = state.position.shape

    # We need two independent random draws this step:
    #   - Gaussian noise for the proposal
    #   - Uniform(0,1) for the accept/reject test
    next_key, k_noise, k_u = random.split(key, 3)

    # random noise for each chain -> shape (n_chains, D)
    eps = random.normal(k_noise, shape=(n_chains, D))

    #  broadcast multiply to (n_chains, D).
    proposal = state.position + scale * eps

    # logprob for proposals
    new_lp = vmap(logprob_fn)(proposal)   # shape (n_chains,)
    # vmap(logprob_fn) lifts logprob_fn to batch over the first axis, calling
    # it once per chain and stacking the results.

    # MH log ratio
    log_alpha = new_lp - state.log_prob   # shape (n_chains,)

    # Accept/reject per chain (in parallel)
    u = random.uniform(k_u, shape=(n_chains,))
    # Accept if log(u) < min(0, log_alpha)
    accept = jnp.log(u) < jnp.minimum(0.0, log_alpha)   # bool array, (n_chains,)

    # JAX functional updates (no in-place mutations)
    new_pos = jnp.where(accept[:, None], proposal, state.position)
    new_lp2 = jnp.where(accept, new_lp, state.log_prob)
    new_accept_count = state.accept_count + accept.astype(jnp.int32)

    return next_key, RWMState(new_pos, new_lp2, new_accept_count)


@partial(jit, static_argnames=("logprob_fn", "num_samples", "burn_in"))
def rwMH_run(key: Array, logprob_fn: LogProbFn, init_position: Array, 
            num_samples: int, scale: float, burn_in: int = 0):
    """Runs multiple chains of Random Walk M-H in parallel."""
    state = rwMH_init(logprob_fn, init_position)
    n_chains, D = state.position.shape

    scale_arr = jnp.asarray(scale) # to avoid retracing

    if burn_in > 0:
        def burn_body(carry, _):
            key, st = carry
            key, st = rwMH_step(key, st, logprob_fn, scale_arr)
            return (key, st), None

        (key, state), _ = lax.scan(burn_body, (key, state), jnp.arange(burn_in))
        # reset accept counter after warmup
        state = RWMState(
            position=state.position,
            log_prob=state.log_prob,
            accept_count=jnp.zeros(n_chains, dtype=jnp.int32),
        )

    # Pre-allocate arrays to accumulate samples and log probs
    samples = jnp.zeros((num_samples, n_chains, D))
    lps     = jnp.zeros((num_samples, n_chains))

    def step(carry, t):
        key, state, samples, lps = carry
        # Take one step, get next key and updated state
        key, state = rwMH_step(key, state, logprob_fn, scale_arr)
        # Store current position and log prob
        samples = samples.at[t].set(state.position)
        lps     = lps.at[t].set(state.log_prob)
        return (key, state, samples, lps), None

    # Run the sampling loop with lax.scan for efficiency
    (key, final_state, samples, lps), _ = lax.scan(
        step, (key, state, samples, lps), xs=jnp.arange(num_samples)
    )
    accept_rate = final_state.accept_count.astype(jnp.float32) / num_samples
    return samples, lps, accept_rate, final_state




def run_sweep(key: Array, dims: list[int], scales: list[float], 
              n_chains: int = 4, num_samples: int = 3000, init_strategy: str = "zeros") -> dict:
    results = {}
    for D in dims:
        # initial positions
        if init_strategy == "zeros":
            init = jnp.zeros((n_chains, D))
        elif init_strategy == "overdispersed":
            subkey, key = random.split(key)
            init = random.normal(subkey, (n_chains, D)) * 3.0
        else:
            raise ValueError("Unknown init_strategy")

        for scale in scales:
            subkey, key = random.split(key)
            samples_jax, lps_jax, acc_rate_jax, _ = rwMH_run(
                subkey, standard_normal_log_prob, init, num_samples, scale
            )
            samples_np = np.array(samples_jax)
            acc_rate_np = np.array(acc_rate_jax)
            esjd_np = esjd(samples_np)
            results[(D, scale)] = {
                "accept_rate": acc_rate_np,
                "esjd": esjd_np,
                "samples": samples_np,
            }
            print(f"[D={D:>3}, scale={scale:>5.3f}] "
                  f"accept={acc_rate_np.mean():.3f} ± {acc_rate_np.std():.3f} | "
                  f"ESJD={esjd_np.mean():.3f}")
    return results



if __name__ == "__main__":

    key = random.PRNGKey(30)
    dims = [2, 10, 50, 100]
    scales = [0.1, 0.3, 0.6, 1.2, 2.4]

    results = run_sweep(
        key,
        dims=dims,
        scales=scales,
        n_chains=4,
        num_samples=4000,
        init_strategy="zeros",
    )

    # pick one config
    D_sel, scale_sel = 10, 0.6
    samples = results[(D_sel, scale_sel)]["samples"]  # shape (draws, chains, D)
    samples = np.asarray(samples).transpose(1, 0, 2)   # (chains, draws, D)

    idata = az.from_dict(
        posterior={"x": samples},
        coords={"x_dim": np.arange(samples.shape[-1])},
        dims={"x": ["x_dim"]},
    )

    print("arviz summary for first few dimensions:\n")
    print(az.summary(idata, var_names=["x"], coords={"x_dim": slice(0, 5)}, kind="all"))

    az.plot_trace(idata, var_names=["x"], coords={"x_dim": slice(0, 3)})
    plt.show()

    az.plot_autocorr(idata, var_names=["x"], coords={"x_dim": slice(0, 3)})
    plt.show()
