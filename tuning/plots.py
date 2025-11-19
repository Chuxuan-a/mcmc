"""Plotting utilities for tuning diagnostics."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


def plot_tuning_history(history: Dict, sampler_name: str = "RWMH", output_file: str = None):
    """Plot dual averaging tuning history.

    Args:
        history: Dictionary containing tuning history
        sampler_name: Name of sampler for plot title
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")

    # Check if this is NUTS (has tree_depth_history)
    is_nuts = "tree_depth_history" in history
    n_plots = 3 if is_nuts else 2

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)

    # Get parameter name (scale for RWMH, step_size for NUTS)
    if "scale_history" in history:
        param_history = history["scale_history"]
        param_name = "Scale Parameter"
    else:
        param_history = history["step_size_history"]
        param_name = "Step Size"

    iterations = np.arange(1, len(param_history) + 1)
    converged_iter = history["converged_iter"]

    # Plot parameter evolution (top panel)
    axes[0].plot(iterations, param_history, 'b-', linewidth=1.5, label=param_name)
    axes[0].axvline(converged_iter, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (iter {converged_iter})')
    axes[0].set_ylabel(param_name, fontsize=12)
    axes[0].set_title(f'{sampler_name} Dual Averaging Tuning History', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot acceptance rate evolution (middle panel)
    axes[1].plot(iterations, history["accept_history"], 'g-', linewidth=1.5, label='Acceptance rate')
    axes[1].axhline(history["target_accept"], color='orange', linestyle='--', linewidth=1.5,
                    label=f'Target ({history["target_accept"]:.3f})')
    axes[1].axvline(converged_iter, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (iter {converged_iter})')
    if not is_nuts:
        axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Acceptance Rate', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Plot tree depth evolution for NUTS (bottom panel)
    if is_nuts:
        axes[2].plot(iterations, history["tree_depth_history"], 'm-', linewidth=1.5, label='Avg tree depth')
        axes[2].axhline(history["max_tree_depth"], color='red', linestyle='--', linewidth=1.5,
                        label=f'Max depth ({history["max_tree_depth"]})')
        axes[2].axvline(converged_iter, color='r', linestyle='--', linewidth=1.5,
                        label=f'Converged (iter {converged_iter})')
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('Average Tree Depth', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved tuning history plot to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_sampling_diagnostics(samples: np.ndarray, diagnostics: Dict,
                               sampler_name: str = "RWMH", output_file: str = None):
    """Plot sampling diagnostics including trace plots and marginals.

    Args:
        samples: Array of shape (n_samples, n_chains, n_dim)
        diagnostics: Dictionary containing diagnostic results
        sampler_name: Name of sampler for plot title
        output_file: Path to save plot (if None, displays interactively)
    """
    n_samples, n_chains, n_dim = samples.shape

    # Plot only first 4 dimensions to keep it manageable
    plot_dims = min(4, n_dim)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(plot_dims, 2, figsize=(12, 3 * plot_dims))
    if plot_dims == 1:
        axes = axes.reshape(1, -1)

    samples_np = np.array(samples)

    for i in range(plot_dims):
        # Trace plots (left column)
        for chain in range(n_chains):
            axes[i, 0].plot(samples_np[:, chain, i], alpha=0.6, linewidth=0.5, label=f'Chain {chain+1}')
        axes[i, 0].set_ylabel(f'x[{i}]', fontsize=10)
        axes[i, 0].set_title(f'Trace Plot (dim {i})', fontsize=10)
        if i == 0:
            axes[i, 0].legend(loc='upper right', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        # Marginal distributions (right column)
        for chain in range(n_chains):
            axes[i, 1].hist(samples_np[:, chain, i], bins=30, alpha=0.4, density=True, label=f'Chain {chain+1}')

        # Overlay true standard normal
        x_range = np.linspace(-4, 4, 100)
        axes[i, 1].plot(x_range, np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi),
                        'k--', linewidth=2, label='True N(0,1)')
        axes[i, 1].set_xlabel(f'x[{i}]', fontsize=10)
        axes[i, 1].set_ylabel('Density', fontsize=10)
        axes[i, 1].set_title(f'Marginal (dim {i})', fontsize=10)
        if i == 0:
            axes[i, 1].legend(loc='upper right', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

    if plot_dims > 1:
        axes[-1, 0].set_xlabel('Iteration', fontsize=10)
    else:
        axes[0, 0].set_xlabel('Iteration', fontsize=10)

    plt.suptitle(f'{sampler_name} Sampling Diagnostics', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved sampling diagnostics plot to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_grid_comparison(grid_results: list, num_steps_grid: list, output_file: str = None):
    """Plot grid search comparison of ESS per gradient vs num_steps.

    Args:
        grid_results: List of result dictionaries from grid search
        num_steps_grid: List of num_steps values that were tried
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract metrics
    ess_per_grad = [r['ess_per_gradient'] for r in grid_results]
    ess_per_sample = [r['ess_per_sample'] for r in grid_results]
    total_grad_calls = [r['total_gradient_calls'] for r in grid_results]
    step_sizes = [r['step_size'] for r in grid_results]

    # Find best
    best_idx = np.argmax(ess_per_grad)
    best_L = num_steps_grid[best_idx]

    # Plot 1: ESS per gradient (KEY METRIC)
    axes[0, 0].plot(num_steps_grid, ess_per_grad, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 0].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[0, 0].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[0, 0].set_ylabel('ESS per Gradient Call', fontsize=12)
    axes[0, 0].set_title('Computational Efficiency (KEY METRIC)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: ESS per sample (for comparison)
    axes[0, 1].plot(num_steps_grid, ess_per_sample, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 1].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[0, 1].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[0, 1].set_ylabel('ESS per Sample', fontsize=12)
    axes[0, 1].set_title('ESS per Sample (ignores cost)', fontsize=12)
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Total gradient calls to reach target ESS
    axes[1, 0].bar(range(len(num_steps_grid)), total_grad_calls, color='purple', alpha=0.6)
    axes[1, 0].set_xticks(range(len(num_steps_grid)))
    axes[1, 0].set_xticklabels(num_steps_grid)
    axes[1, 0].axvline(best_idx, color='r', linestyle='--', linewidth=2, label=f'Best: L={best_L}')
    axes[1, 0].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[1, 0].set_ylabel('Total Gradient Calls', fontsize=12)
    axes[1, 0].set_title('Computational Cost to Reach Target ESS', fontsize=12)
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Tuned step sizes
    axes[1, 1].plot(num_steps_grid, step_sizes, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 1].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[1, 1].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[1, 1].set_ylabel('Tuned Step Size', fontsize=12)
    axes[1, 1].set_title('Step Size vs Trajectory Length', fontsize=12)
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('HMC Grid Search: Trajectory Length Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved grid comparison plot to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_grahmc_grid_comparison(grid_results: list, num_steps_grid: list,
                                  schedule_type: str = 'constant',
                                  has_steepness: bool = False,
                                  output_file: str = None):
    """Plot GRAHMC grid search comparison with schedule-specific parameters.

    Args:
        grid_results: List of result dictionaries from grid search
        num_steps_grid: List of num_steps values that were tried
        schedule_type: Friction schedule type
        has_steepness: Whether schedule uses steepness parameter
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")

    # 2x3 layout for schedules with steepness, 2x2 for schedules without
    if has_steepness:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

    # Extract metrics
    ess_per_grad = [r['ess_per_gradient'] for r in grid_results]
    ess_per_sample = [r['ess_per_sample'] for r in grid_results]
    total_grad_calls = [r['total_gradient_calls'] for r in grid_results]
    step_sizes = [r['step_size'] for r in grid_results]
    gammas = [r['gamma'] for r in grid_results]
    if has_steepness:
        steepnesses = [r['steepness'] for r in grid_results]

    # Find best
    best_idx = np.argmax(ess_per_grad)
    best_L = num_steps_grid[best_idx]

    # Plot 1: ESS per gradient (KEY METRIC)
    axes[0].plot(num_steps_grid, ess_per_grad, 'o-', linewidth=2, markersize=8, color='green')
    axes[0].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[0].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[0].set_ylabel('ESS per Gradient Call', fontsize=12)
    axes[0].set_title('Computational Efficiency (KEY METRIC)', fontsize=12, fontweight='bold')
    axes[0].set_xscale('log', base=2)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: ESS per sample
    axes[1].plot(num_steps_grid, ess_per_sample, 'o-', linewidth=2, markersize=8, color='blue')
    axes[1].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[1].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[1].set_ylabel('ESS per Sample', fontsize=12)
    axes[1].set_title('ESS per Sample (ignores cost)', fontsize=12)
    axes[1].set_xscale('log', base=2)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Total gradient calls
    axes[2].bar(range(len(num_steps_grid)), total_grad_calls, color='purple', alpha=0.6)
    axes[2].set_xticks(range(len(num_steps_grid)))
    axes[2].set_xticklabels(num_steps_grid)
    axes[2].axvline(best_idx, color='r', linestyle='--', linewidth=2, label=f'Best: L={best_L}')
    axes[2].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[2].set_ylabel('Total Gradient Calls', fontsize=12)
    axes[2].set_title('Computational Cost to Reach Target ESS', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3, axis='y')

    # Plot 4: Tuned step sizes
    axes[3].plot(num_steps_grid, step_sizes, 'o-', linewidth=2, markersize=8, color='orange')
    axes[3].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
    axes[3].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
    axes[3].set_ylabel('Tuned Step Size', fontsize=12)
    axes[3].set_title('Step Size vs Trajectory Length', fontsize=12)
    axes[3].set_xscale('log', base=2)
    axes[3].legend(loc='best')
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Tuned gamma values
    if has_steepness:
        ax_gamma = axes[4]
    else:
        ax_gamma = axes[2] if len(axes) > 4 else None
        if ax_gamma is None:
            # Create new subplot position
            ax_gamma = fig.add_subplot(2, 2, 3)

    if ax_gamma is not None:
        ax_gamma.plot(num_steps_grid, gammas, 'o-', linewidth=2, markersize=8, color='cyan')
        ax_gamma.axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
        ax_gamma.set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
        ax_gamma.set_ylabel('Tuned Gamma (friction)', fontsize=12)
        ax_gamma.set_title('Gamma vs Trajectory Length', fontsize=12)
        ax_gamma.set_xscale('log', base=2)
        ax_gamma.legend(loc='best')
        ax_gamma.grid(True, alpha=0.3)

    # Plot 6: Tuned steepness (if applicable)
    if has_steepness:
        axes[5].plot(num_steps_grid, steepnesses, 'o-', linewidth=2, markersize=8, color='magenta')
        axes[5].axvline(best_L, color='r', linestyle='--', linewidth=1.5, label=f'Best: L={best_L}')
        axes[5].set_xlabel('Number of Leapfrog Steps (L)', fontsize=12)
        axes[5].set_ylabel('Tuned Steepness', fontsize=12)
        axes[5].set_title('Steepness vs Trajectory Length', fontsize=12)
        axes[5].set_xscale('log', base=2)
        axes[5].legend(loc='best')
        axes[5].grid(True, alpha=0.3)

    plt.suptitle(f'GRAHMC ({schedule_type.upper()}) Grid Search: Trajectory Length Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved grid comparison plot to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_coordinate_tuning_history(history: Dict, output_file: str = None):
    """Plot coordinate-wise tuning history for GRAHMC.

    Args:
        history: Dictionary containing cycle history from coordinate_wise_tune_grahmc
        output_file: Path to save plot (if None, displays interactively)
    """
    sns.set_style("whitegrid")

    cycle_history = history['cycle_history']
    schedule_type = history['schedule_type']
    has_steepness = history['has_steepness']
    num_steps = history['num_steps']
    converged_cycle = history['converged_cycle']

    # Extract data
    cycles = np.arange(1, len(cycle_history) + 1)
    step_sizes = [c['step_size'] for c in cycle_history]
    gammas = [c['gamma'] for c in cycle_history]
    steepnesses = [c['steepness'] for c in cycle_history]
    accept_rates = [c['accept_rate'] for c in cycle_history]

    # Create subplots
    n_plots = 4 if has_steepness else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)

    # Plot 1: Step size evolution
    axes[0].plot(cycles, step_sizes, 'o-', linewidth=2, markersize=6, color='orange')
    axes[0].axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (cycle {converged_cycle})')
    axes[0].set_ylabel('Step Size', fontsize=12)
    axes[0].set_title(f'GRAHMC ({schedule_type.upper()}, L={num_steps}) Coordinate-wise Tuning History',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Gamma evolution
    axes[1].plot(cycles, gammas, 'o-', linewidth=2, markersize=6, color='cyan')
    axes[1].axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                    label=f'Converged (cycle {converged_cycle})')
    axes[1].set_ylabel('Gamma (friction)', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Steepness evolution (if applicable)
    if has_steepness:
        axes[2].plot(cycles, steepnesses, 'o-', linewidth=2, markersize=6, color='magenta')
        axes[2].axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                        label=f'Converged (cycle {converged_cycle})')
        axes[2].set_ylabel('Steepness', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

    # Plot 4 (or 3): Acceptance rate evolution
    ax_accept = axes[3] if has_steepness else axes[2]
    ax_accept.plot(cycles, accept_rates, 'o-', linewidth=2, markersize=6, color='green')
    ax_accept.axhline(history['target_accept'], color='orange', linestyle='--', linewidth=1.5,
                      label=f'Target ({history["target_accept"]:.3f})')
    ax_accept.axvline(converged_cycle, color='r', linestyle='--', linewidth=1.5,
                      label=f'Converged (cycle {converged_cycle})')
    ax_accept.set_xlabel('Cycle', fontsize=12)
    ax_accept.set_ylabel('Acceptance Rate', fontsize=12)
    ax_accept.legend(loc='best')
    ax_accept.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved coordinate tuning history plot to {output_file}")
    else:
        plt.show()
    plt.close()