import jax
import jax.numpy as jnp
from jax import random, vmap, value_and_grad, jit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from functools import partial
from samplers.GRAHMC import FrictionScheduleFn
from typing import Callable, Tuple

# --- Imports from your files ---
# Assuming this script is in the PARENT directory, and HMC.py/GRAHMC.py are in samplers/
try:
    from samplers.GRAHMC import (
        # rahmc_proposal_trace,  <-- REMOVED
        # lp_2d_bimodal,       <-- REMOVED
        # _contour_U_2d,         <-- REMOVED
        constant_schedule,
        Array,
        LogProbFn
    )
    from samplers.HMC import hmc_init, leapfrog
except ImportError:
    print("Error: Make sure this script is in the directory *above* samplers/")
    print("And that samplers/HMC.py and samplers/GRAHMC.py exist.")
    exit()

print("JAX device:", jax.devices()[0])
jax.config.update("jax_enable_x64", True)

# --- 1. Define the Target Distribution (MOVED FROM GRAHMC.py) ---

# This is from samplers/GRAHMC.py
def lp_2d_bimodal(q: Array) -> Array:
    """Bimodal 2D target distribution."""
    q = jnp.atleast_1d(q); x, y = q[0], q[1]
    def logN(mx, my, sx=0.7, sy=0.7, rho=0.0):
        inv = 1.0/(1 - rho**2)
        qf = inv*(((x-mx)/sx)**2 - 2*rho*((x-mx)/sx)*((y-my)/sy) + ((y-my)/sy)**2)
        return -0.5*qf - jnp.log(sx*sy)
    c1 = logN(-2.0, -1.2, rho= 0.2)
    c2 = logN(+2.0, +1.2, rho=-0.2)
    return jax.scipy.special.logsumexp(jnp.array([c1, c2]) + jnp.log(0.5))

log_prob_fn = lp_2d_bimodal

# --- 2. HMC Trajectory Tracer (MOVED FROM GRAHMC.py) ---

# This is from samplers/GRAHMC.py
def _contour_U_2d(ax, log_prob_fn, bounds, n=220):
    """Helper to draw 2D contours of the potential U = -log_prob_fn."""
    xs = np.linspace(bounds[0], bounds[1], n)
    ys = np.linspace(bounds[2], bounds[3], n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    
    # Define a jitted vmap'd log_prob_fn for contours
    @jit
    @vmap
    def v_lp2(v_q: Array) -> Array:
        return log_prob_fn(jnp.array(v_q))
        
    U = -np.array(v_lp2(jnp.array(grid))).reshape(n, n)
    U = np.clip(U, -10, 30) # Clip for better visualization
    cs = ax.contour(X, Y, U, levels=18, linewidths=0.6, colors="0.65", alpha=0.95)
    return cs

# This is your HMC tracer, unchanged
@partial(jit, static_argnames=("log_prob_fn",))
def _hmc_leapfrog_step_logic(
    position: Array,
    momentum: Array,
    step_size: float,
    log_prob: Array,
    grad_log_prob: Array,
    log_prob_fn: LogProbFn,
) -> Tuple[Array, Array, Array, Array]:
    """
    Performs the logic of ONE leapfrog step,
    based on your samplers/HMC.py `lf_step` [cite: 3383, L79-L88].
    """
    pos_dtype = position.dtype
    lp_dtype = log_prob.dtype
    step_sz = jnp.asarray(step_size, dtype=pos_dtype)
    half = jnp.array(0.5, dtype=pos_dtype)

    # Half step for momentum
    mom = momentum + half * step_sz * grad_log_prob
    # Full step for position
    pos = position + step_sz * mom
    # Update gradient at new position
    new_lp, new_grad_lp = vmap(jax.value_and_grad(log_prob_fn))(pos)
    new_lp = new_lp.astype(lp_dtype)
    new_grad_lp = new_grad_lp.astype(pos_dtype)
    # Half step for momentum
    mom = mom + half * step_sz * new_grad_lp
    return pos, mom, new_lp, new_grad_lp

def hmc_proposal_trace(
    key: Array,
    log_prob_fn: LogProbFn,
    q0: Array,
    step_size: float,
    num_steps: int,
):
    """
    Generates a full HMC proposal trace (q, p, U, K, H) for visualization,
    using the logic from your HMC.py file [cite: 3383, L79-L88].
    """
    q = jnp.atleast_1d(q0).astype(jnp.float32)
    # Ensure it's batched for hmc_init
    if q.ndim == 1:
        q = q[None, :] # (1, d)

    # Use hmc_init to get initial lp and grad
    state = hmc_init(q, log_prob_fn)
    q, lp, glp = state.position, state.log_prob, state.grad_log_prob
    d = q.shape[1]

    # Sample initial momentum
    p = random.normal(key, shape=q.shape, dtype=q.dtype)

    # Initial energies
    U = -lp.astype(jnp.float64)
    K = 0.5 * jnp.sum(p**2, axis=-1).astype(jnp.float64)
    H = U + K

    # Store traces (as numpy arrays for matplotlib)
    Qs = [np.array(q[0])] # De-batch
    Ps = [np.array(p[0])]
    Us = [float(U[0])]
    Ks = [float(K[0])]
    Hs = [float(H[0])]

    # Run the loop
    for _ in range(num_steps):
        q, p, lp, glp = _hmc_leapfrog_step_logic(
            q, p, step_size, lp, glp, log_prob_fn
        )
        
        U = -lp.astype(jnp.float64)
        K = 0.5 * jnp.sum(p**2, axis=-1).astype(jnp.float64)
        H = U + K
        
        Qs.append(np.array(q[0]))
        Ps.append(np.array(p[0]))
        Us.append(float(U[0]))
        Ks.append(float(K[0]))
        Hs.append(float(H[0]))

    return (
        np.stack(Qs), np.stack(Ps),
        np.array(Us), np.array(Ks), np.array(Hs)
    )


# --- 3. RAHMC Trajectory Tracer (MOVED FROM GRAHMC.py and FIXED) ---

def rahmc_proposal_trace(
    key: Array,
    log_prob_fn: LogProbFn,
    q0: Array,
    step_size: float,
    num_steps: int,
    gamma_max: float,
    steepness: float | None,  # <-- ADDED steepness to signature
    friction_schedule: FrictionScheduleFn, # <-- Now accepts the 4-arg fn
    resample_p_between_segments: bool = False,
):
    """
    Run ONE raHMC proposal (no MH decision) and record per-substep (q,p,U,K,H).
    This is a *LOCAL* helper for visualization.
    """
    q = jnp.atleast_1d(q0).astype(jnp.float32)
    d = q.shape[-1]
    key, k_p1, k_p2 = random.split(key, 3)

    # Identity mass matrix for visualization (no adaptation)
    inv_mass_matrix = jnp.ones(d, dtype=q.dtype)

    p = random.normal(k_p1, shape=(d,), dtype=q.dtype)

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

    # Get the jitted conformal leapfrog step from GRAHMC.py
    # We must import it
    from samplers.GRAHMC import _conformal_leapfrog_step

    # ---- Repelling segment ----
    for i in range(L1):
        t_now = i * step_size
        
        # *** THE FIX IS HERE ***
        # Call schedule with all 4 args
        gamma_t = float(friction_schedule(t_now, total_time, gamma_max, steepness))
        
        q, p, lp, glp = _conformal_leapfrog_step(
            q[None, :], p[None, :], step_size, gamma_t,
            lp[None, ...], glp[None, :],
            log_prob_fn,
            inv_mass_matrix
        )
        q = q[0]; p = p[0]; lp = lp[0]; glp = glp[0]
        U = -lp
        K = 0.5 * jnp.sum(p**2).astype(jnp.float64)
        H = U + K
        Qs.append(np.array(q)); Ps.append(np.array(p))
        Us.append(float(U)); Ks.append(float(K)); Hs.append(float(H))

    split_idx = len(Qs) - 1
    resample_idx = None

    if resample_p_between_segments:
        resample_idx = len(Qs) - 1
        p = random.normal(k_p2, shape=(d,), dtype=q.dtype)
        U = -lp; K = 0.5 * jnp.sum(p**2).astype(jnp.float64); H = U + K
        Qs.append(np.array(q)); Ps.append(np.array(p))
        Us.append(float(U)); Ks.append(float(K)); Hs.append(float(H))

    # ---- Attracting segment ----
    for j in range(L2):
        t_now = (L1 + j) * step_size
        
        # *** THE FIX IS HERE ***
        # Call schedule with all 4 args
        gamma_t = float(friction_schedule(t_now, total_time, gamma_max, steepness))
        
        q, p, lp, glp = _conformal_leapfrog_step(
            q[None, :], p[None, :], step_size, gamma_t,
            lp[None, ...], glp[None, :],
            log_prob_fn,
            inv_mass_matrix
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


# --- 4. Set Up Parameters ---
key = random.PRNGKey(30)
key, hmc_key, rahmc_key = random.split(key, 3)

# Starting position near the first mode
start_q = jnp.array([-2.6, -1.6], dtype=jnp.float32)

# Parameters for the animation
STEP_SIZE = 0.1
NUM_STEPS = 50  # Total number of frames in the animation
RAHMC_GAMMA = 1.5 # Friction for RAHMC

print("Generating HMC trajectory (using samplers/HMC.py)...")
# --- 5. Generate HMC Trajectory (Actor 1) ---
Qs_hmc, Ps_hmc, Us_hmc, Ks_hmc, Hs_hmc = hmc_proposal_trace(
    key=hmc_key,
    log_prob_fn=log_prob_fn,
    q0=start_q,
    step_size=STEP_SIZE,
    num_steps=NUM_STEPS,
)
print("Generating RAHMC trajectory (using samplers/GRAHMC.py)...")
# --- 6. Generate RAHMC Trajectory (Actor 2) ---
    
# **FIX:** Call signature is now correct.
Qs_ra, Ps_ra, Us_ra, Ks_ra, Hs_ra, split_idx_ra, _ = rahmc_proposal_trace(
    key=rahmc_key,
    log_prob_fn=log_prob_fn,
    q0=start_q,
    step_size=STEP_SIZE,
    num_steps=NUM_STEPS,
    gamma_max=RAHMC_GAMMA,
    friction_schedule=constant_schedule, # <-- Pass the 4-arg function
    steepness=None # <-- Pass steepness (None is fine for constant_schedule)
)
print(f"RAHMC split index: {split_idx_ra} / {NUM_STEPS}")

# --- 7. Set Up the Animation Plot ---
fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
ax.set_facecolor('#fafafa')

# Define plot bounds based on trajectories
all_qs = np.vstack([Qs_hmc, Qs_ra])
x_min, y_min = all_qs.min(axis=0) - 1
x_max, y_max = all_qs.max(axis=0) + 1
bounds = (x_min, x_max, y_min, y_max)

# Draw the static contour background
print("Drawing contour background...")
_contour_U_2d(ax, log_prob_fn, bounds=bounds, n=150)

# Initialize animated plot elements with muted colors
line_hmc, = ax.plot([], [], lw=2.5, color='#6b8e23', alpha=0.85, linestyle='-')
line_ra_repel, = ax.plot([], [], lw=2.5, color='#d4a574', alpha=0.85, linestyle='-')
line_ra_attract, = ax.plot([], [], lw=2.5, color='#5f9ea0', alpha=0.85, linestyle='-')
dot_hmc, = ax.plot([], [], 'o', color='#6b8e23', markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=5)
dot_ra, = ax.plot([], [], 'o', color='#d4a574', markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=5)

# Mark starting position
ax.plot(start_q[0], start_q[1], 's', color='#8b4545', markersize=10,
        markeredgecolor='white', markeredgewidth=1.5, label='Start', zorder=4)

# Static text and legend
title_text = ax.set_title("HMC vs. RAHMC: Trajectory Comparison", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('$q_1$', fontsize=16, fontweight='semibold')
ax.set_ylabel('$q_2$', fontsize=16, fontweight='semibold', rotation=0, ha='right', labelpad=15)
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_aspect('equal')
ax.tick_params(labelsize=12)

# Create custom legend handles with better formatting
legend_patches = [
    mpatches.Patch(color='#6b8e23', label='HMC ($\\gamma=0$) - Trapped'),
    mpatches.Patch(color='#d4a574', label=f'RAHMC ($\\gamma=-{RAHMC_GAMMA}$) - Repel'),
    mpatches.Patch(color='#5f9ea0', label=f'RAHMC ($\\gamma=+{RAHMC_GAMMA}$) - Attract'),
    mpatches.Patch(color='#8b4545', label='Starting Position')
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=15, framealpha=0.95,
          edgecolor='gray', fancybox=True, shadow=True)

# --- 8. Define the Animation Function ---
def animate(frame):
    # Animate HMC
    line_hmc.set_data(Qs_hmc[:frame+1, 0], Qs_hmc[:frame+1, 1])
    dot_hmc.set_data([Qs_hmc[frame, 0]], [Qs_hmc[frame, 1]])

    # Animate RAHMC
    # Use <= to include the split point in the repel phase
    if frame <= split_idx_ra:
        # --- REPEL PHASE ---
        # 1. Repel line grows
        line_ra_repel.set_data(Qs_ra[:frame+1, 0], Qs_ra[:frame+1, 1])
        # 2. FIX: Attract line is explicitly hidden
        line_ra_attract.set_data([], [])
        
        dot_ra.set_data([Qs_ra[frame, 0]], [Qs_ra[frame, 1]])
        dot_ra.set_color('#d4a574')
        title_text.set_text(f"Step {frame} / {NUM_STEPS}  |  RAHMC Phase: REPEL ($\\gamma < 0$, Adding Energy)")
    else:
        # --- ATTRACT PHASE ---
        # 1. FIX: Repel line is now static and fully drawn
        line_ra_repel.set_data(Qs_ra[:split_idx_ra+1, 0], Qs_ra[:split_idx_ra+1, 1])
        # 2. Attract line grows from the split point
        line_ra_attract.set_data(Qs_ra[split_idx_ra:frame+1, 0], Qs_ra[split_idx_ra:frame+1, 1])

        dot_ra.set_data([Qs_ra[frame, 0]], [Qs_ra[frame, 1]])
        dot_ra.set_color('#5f9ea0')
        title_text.set_text(f"Step {frame} / {NUM_STEPS}  |  RAHMC Phase: ATTRACT ($\\gamma > 0$, Removing Energy)")

    return line_hmc, line_ra_repel, line_ra_attract, dot_hmc, dot_ra, title_text

# --- 9. Create and Save the Animation ---
print("Creating animation... this may take a moment.")
total_frames = NUM_STEPS + 1
anim = FuncAnimation(
    fig,
    animate,
    frames=total_frames,
    interval=80,  # 80ms per frame
    blit=True
)

# Save the animation
output_filename = 'hmc_vs_rahmc_animation.mp4'
anim.save(output_filename, writer='ffmpeg', dpi=300, bitrate=5000)

print(f"\nSuccessfully saved animation to '{output_filename}'")
plt.show()