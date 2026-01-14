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


try:
    from samplers.GRAHMC import (
        # rahmc_proposal_trace,  <-- REMOVED
        # lp_2d_bimodal,       <-- REMOVED
        # _contour_U_2d,         <-- REMOVED
        constant_schedule,
        tanh_schedule,
        sine_schedule,
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
key, hmc_key = random.split(key, 2)

# Starting position near the first mode
start_q = jnp.array([-2.2, -1.0], dtype=jnp.float32)

# Parameters for the animation
STEP_SIZE = 0.08
NUM_STEPS = 72  # Total number of frames in the animation
RAHMC_GAMMA = 1.5 # Friction for RAHMC
STEEPNESS = 5.0   # Steepness for smooth schedules

print("Generating HMC trajectory (using samplers/HMC.py)...")
# --- 5. Generate HMC Trajectory (Baseline) ---
Qs_hmc, Ps_hmc, Us_hmc, Ks_hmc, Hs_hmc = hmc_proposal_trace(
    key=hmc_key,
    log_prob_fn=log_prob_fn,
    q0=start_q,
    step_size=STEP_SIZE,
    num_steps=NUM_STEPS,
)

print("Generating RAHMC trajectories with different friction schedules...")
# --- 6. Generate RAHMC Trajectories with Different Schedules ---

# Need separate keys for each trajectory
key, key1, key2, key3 = random.split(key, 4)

# Constant schedule (original RAHMC)
print("  - Constant schedule...")
Qs_const, Ps_const, Us_const, Ks_const, Hs_const, split_const, _ = rahmc_proposal_trace(
    key=key1,
    log_prob_fn=log_prob_fn,
    q0=start_q,
    step_size=STEP_SIZE,
    num_steps=NUM_STEPS,
    gamma_max=RAHMC_GAMMA,
    friction_schedule=constant_schedule,
    steepness=None
)

# Tanh schedule
print("  - Tanh schedule...")
Qs_tanh, Ps_tanh, Us_tanh, Ks_tanh, Hs_tanh, split_tanh, _ = rahmc_proposal_trace(
    key=key2,
    log_prob_fn=log_prob_fn,
    q0=start_q,
    step_size=STEP_SIZE,
    num_steps=NUM_STEPS,
    gamma_max=RAHMC_GAMMA,
    friction_schedule=tanh_schedule,
    steepness=STEEPNESS
)

# Sine schedule
print("  - Sine schedule...")
Qs_sine, Ps_sine, Us_sine, Ks_sine, Hs_sine, split_sine, _ = rahmc_proposal_trace(
    key=key3,
    log_prob_fn=log_prob_fn,
    q0=start_q,
    step_size=STEP_SIZE,
    num_steps=NUM_STEPS,
    gamma_max=RAHMC_GAMMA,
    friction_schedule=sine_schedule,
    steepness=None
)

print(f"Split indices - Constant: {split_const}, Tanh: {split_tanh}, Sine: {split_sine}")

# --- 7. Set Up the Animation Plot (3-panel layout) ---
fig = plt.figure(figsize=(12, 6), facecolor='white')
gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 1.2, 0.75], hspace=0.13, wspace=0.10)

ax_friction = fig.add_subplot(gs[0, 0])
ax_target = fig.add_subplot(gs[0, 1])
ax_energy = fig.add_subplot(gs[0, 2])

# Reduce left and right margins
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12)


ax_friction.set_facecolor('#fafafa')
ax_target.set_facecolor('#fafafa')
ax_energy.set_facecolor('#fafafa')

# ========== LEFT PANEL: Friction Schedules γ(t) vs Time ==========
time_array = np.linspace(0, STEP_SIZE * NUM_STEPS, 200)
total_time = STEP_SIZE * NUM_STEPS

# Compute friction schedules over time
gamma_const = np.array([constant_schedule(t, total_time, RAHMC_GAMMA, None) for t in time_array])
gamma_tanh = np.array([tanh_schedule(t, total_time, RAHMC_GAMMA, STEEPNESS) for t in time_array])
gamma_sine = np.array([sine_schedule(t, total_time, RAHMC_GAMMA, None) for t in time_array])

# Plot static friction curves
ax_friction.plot(time_array, gamma_const, lw=2.5, color='#d4a574', linestyle='-', label='Constant', alpha=0.8)
ax_friction.plot(time_array, gamma_tanh, lw=2.5, color='#c97064', linestyle='--', label='Tanh', alpha=0.8)
ax_friction.plot(time_array, gamma_sine, lw=2.5, color='#b8860b', linestyle='-.', label='Sine', alpha=0.8)
ax_friction.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax_friction.set_xlabel('Time $t$', fontsize=13, fontweight='semibold')
ax_friction.set_ylabel('Friction $\\gamma(t)$', fontsize=13, fontweight='semibold')
ax_friction.set_title('Friction Schedules', fontsize=14, fontweight='bold', pad=10)
ax_friction.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax_friction.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax_friction.set_xlim(0, total_time)
ax_friction.set_ylim(-RAHMC_GAMMA * 1.3, RAHMC_GAMMA * 1.3)

# Time marker (vertical line)
time_marker_friction, = ax_friction.plot([], [], 'r-', lw=2, alpha=0.7)

# ========== MIDDLE PANEL: Target Space (q₁, q₂) ==========
# Define plot bounds based on all trajectories
all_qs = np.vstack([Qs_hmc, Qs_const, Qs_tanh, Qs_sine])
x_min, y_min = all_qs.min(axis=0) - 1
x_max, y_max = all_qs.max(axis=0) + 1
bounds = (x_min, x_max, y_min, y_max)

# Draw the static contour background
print("Drawing contour background...")
_contour_U_2d(ax_target, log_prob_fn, bounds=bounds, n=150)

# Initialize animated plot elements with muted colors
# HMC (baseline)
line_hmc, = ax_target.plot([], [], lw=2, color='#6b8e23', alpha=0.7, linestyle=':', label='HMC')
dot_hmc, = ax_target.plot([], [], 'o', color='#6b8e23', markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=5)

# Constant schedule - solid lines
line_const_repel, = ax_target.plot([], [], lw=2.5, color='#d4a574', alpha=0.85, linestyle='-')
line_const_attract, = ax_target.plot([], [], lw=2.5, color='#5f9ea0', alpha=0.85, linestyle='-')
dot_const, = ax_target.plot([], [], 'o', color='#d4a574', markersize=11, markeredgecolor='white', markeredgewidth=2, zorder=5)

# Tanh schedule - dashed lines
line_tanh_repel, = ax_target.plot([], [], lw=2.5, color='#c97064', alpha=0.85, linestyle='--')
line_tanh_attract, = ax_target.plot([], [], lw=2.5, color='#7b9aaa', alpha=0.85, linestyle='--')
dot_tanh, = ax_target.plot([], [], 's', color='#c97064', markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=5)

# Sine schedule - dash-dot lines
line_sine_repel, = ax_target.plot([], [], lw=2.5, color='#b8860b', alpha=0.85, linestyle='-.')
line_sine_attract, = ax_target.plot([], [], lw=2.5, color='#4682b4', alpha=0.85, linestyle='-.')
dot_sine, = ax_target.plot([], [], '^', color='#b8860b', markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=5)

# Mark starting position
ax_target.plot(start_q[0], start_q[1], '*', color='#8b4545', markersize=15,
        markeredgecolor='white', markeredgewidth=1.5, label='Start', zorder=6)

# Static text and legend
ax_target.set_title("Target Space Trajectories", fontsize=14, fontweight='bold', pad=10)
ax_target.set_xlabel('$q_1$', fontsize=13, fontweight='semibold')
ax_target.set_ylabel('$q_2$', fontsize=13, fontweight='semibold', rotation=0, ha='right', labelpad=15)
ax_target.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax_target.set_aspect('equal')
ax_target.tick_params(labelsize=11)

# Create custom legend handles with better formatting
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#6b8e23', lw=2, linestyle=':', label='HMC', alpha=0.7),
    Line2D([0], [0], color='#d4a574', lw=2.5, linestyle='-', label='Constant'),
    Line2D([0], [0], color='#c97064', lw=2.5, linestyle='--', label='Tanh'),
    Line2D([0], [0], color='#b8860b', lw=2.5, linestyle='-.', label='Sine'),
]
ax_target.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95,
          edgecolor='gray', fancybox=True, shadow=True)

# ========== RIGHT PANEL: Energy Evolution ==========
# Compute time points for each trajectory step
time_steps = np.arange(NUM_STEPS + 1) * STEP_SIZE

# Plot energy trajectories (static, will progressively reveal)
line_H_hmc, = ax_energy.plot([], [], lw=2, color='#6b8e23', linestyle=':', label='HMC (H)', alpha=0.7)
line_H_const, = ax_energy.plot([], [], lw=2.5, color='#8b7355', linestyle='-', label='Constant (H)', alpha=0.8)
line_H_tanh, = ax_energy.plot([], [], lw=2.5, color='#a0522d', linestyle='--', label='Tanh (H)', alpha=0.8)
line_H_sine, = ax_energy.plot([], [], lw=2.5, color='#8b6914', linestyle='-.', label='Sine (H)', alpha=0.8)

ax_energy.set_xlabel('Time $t$', fontsize=13, fontweight='semibold')
ax_energy.set_ylabel('Hamiltonian $H(t)$', fontsize=13, fontweight='semibold')
ax_energy.set_title('Energy Evolution', fontsize=14, fontweight='bold', pad=10)
ax_energy.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax_energy.legend(fontsize=10, framealpha=0.9, loc='upper left')
ax_energy.set_xlim(0, total_time)

# Set y-limits based on all energy values
all_H = np.concatenate([Hs_hmc, Hs_const, Hs_tanh, Hs_sine])
H_min, H_max = all_H.min(), all_H.max()
H_range = H_max - H_min
ax_energy.set_ylim(H_min - 0.1 * H_range, H_max + 0.1 * H_range)

# Time marker (vertical line)
time_marker_energy, = ax_energy.plot([], [], 'r-', lw=2, alpha=0.7)

# Set overall title
fig.suptitle('HMC vs. GRAHMC: Friction Schedules and Energy Dynamics',
             fontsize=16, fontweight='bold', y=0.98)

# --- 8. Define the Animation Function ---
def animate(frame):
    current_time = frame * STEP_SIZE

    # ========== LEFT PANEL: Update time marker on friction plot ==========
    time_marker_friction.set_data([current_time, current_time],
                                   [-RAHMC_GAMMA * 1.3, RAHMC_GAMMA * 1.3])

    # ========== MIDDLE PANEL: Animate trajectories in target space ==========
    # Helper function to animate a GRAHMC trajectory
    def animate_schedule(Qs, split_idx, line_repel, line_attract, dot, repel_color, attract_color):
        if frame <= split_idx:
            # Repel phase
            line_repel.set_data(Qs[:frame+1, 0], Qs[:frame+1, 1])
            line_attract.set_data([], [])
            dot.set_data([Qs[frame, 0]], [Qs[frame, 1]])
            dot.set_color(repel_color)
        else:
            # Attract phase
            line_repel.set_data(Qs[:split_idx+1, 0], Qs[:split_idx+1, 1])
            line_attract.set_data(Qs[split_idx:frame+1, 0], Qs[split_idx:frame+1, 1])
            dot.set_data([Qs[frame, 0]], [Qs[frame, 1]])
            dot.set_color(attract_color)

    # Animate HMC (baseline)
    line_hmc.set_data(Qs_hmc[:frame+1, 0], Qs_hmc[:frame+1, 1])
    dot_hmc.set_data([Qs_hmc[frame, 0]], [Qs_hmc[frame, 1]])

    # Animate constant schedule
    animate_schedule(Qs_const, split_const, line_const_repel, line_const_attract,
                    dot_const, '#d4a574', '#5f9ea0')

    # Animate tanh schedule
    animate_schedule(Qs_tanh, split_tanh, line_tanh_repel, line_tanh_attract,
                    dot_tanh, '#c97064', '#7b9aaa')

    # Animate sine schedule
    animate_schedule(Qs_sine, split_sine, line_sine_repel, line_sine_attract,
                    dot_sine, '#b8860b', '#4682b4')

    # ========== RIGHT PANEL: Update energy evolution curves ==========
    # Progressively reveal energy curves up to current frame
    line_H_hmc.set_data(time_steps[:frame+1], Hs_hmc[:frame+1])
    line_H_const.set_data(time_steps[:frame+1], Hs_const[:frame+1])
    line_H_tanh.set_data(time_steps[:frame+1], Hs_tanh[:frame+1])
    line_H_sine.set_data(time_steps[:frame+1], Hs_sine[:frame+1])

    # Update time marker on energy plot
    time_marker_energy.set_data([current_time, current_time],
                                 [H_min - 0.1 * H_range, H_max + 0.1 * H_range])

    return (time_marker_friction,
            line_hmc, dot_hmc,
            line_const_repel, line_const_attract, dot_const,
            line_tanh_repel, line_tanh_attract, dot_tanh,
            line_sine_repel, line_sine_attract, dot_sine,
            line_H_hmc, line_H_const, line_H_tanh, line_H_sine,
            time_marker_energy)

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
output_filename = 'hmc_vs_grahmc_3panel_animation.mp4'
anim.save(output_filename, writer='ffmpeg', dpi=300, bitrate=8000)

print(f"\nSuccessfully saved animation to '{output_filename}'")
plt.show()