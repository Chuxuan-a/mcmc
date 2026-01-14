"""Grid search analysis and visualization for trajectory length (L) selection."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import filter_by_sampler, filter_by_target, filter_with_grid_search


# Plotting style configuration
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'tab10'
MARKER_SIZE = 60
LINE_WIDTH = 2


def plot_L_selection_analysis(
    results: List[Dict],
    output_dir: str,
    sampler: Optional[str] = None,
    target: Optional[str] = None,
    save_format: str = 'png',
) -> None:
    """Plot comprehensive L selection analysis showing impact on all metrics.

    Creates a 2x3 grid of plots showing how trajectory length (L) affects:
    - Row 1: Efficiency metrics (ESS/grad, W2, ESS_tail)
    - Row 2: Stability and cost (R-hat, accept rate, warmup time)

    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save plots
        sampler: Optional sampler filter (if None, plots all samplers separately)
        target: Optional target filter (if None, plots all targets separately)
        save_format: Output format ('png', 'pdf', or 'both')

    Note:
        Only works with results that have grid_search_info data.
        Older results without enhanced logging will be skipped.
    """
    # Filter to grid search results only
    results = filter_with_grid_search(results)

    if not results:
        print("WARNING: No results with grid search data found!")
        print("Grid search analysis requires benchmark results generated with enhanced logging.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Determine grouping
    if sampler is not None:
        results = filter_by_sampler(results, sampler)
    if target is not None:
        results = filter_by_target(results, target)

    if not results:
        print(f"No results found for sampler={sampler}, target={target}")
        return

    # Group results by (sampler, target) combinations
    groups = {}
    for r in results:
        key = (r["sampler"], r["target"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Generate plot for each group
    for (samp, targ), group_results in groups.items():
        if len(group_results) == 0:
            continue

        # Each group should have one result with multiple L values in grid_results
        # Take the first result and extract grid_results
        result = group_results[0]
        grid_results = result.get("grid_results", [])

        if len(grid_results) < 2:
            print(f"Skipping {samp}/{targ}: insufficient grid search data ({len(grid_results)} L values)")
            continue

        _plot_single_L_analysis(
            grid_results=grid_results,
            selected_L=result.get("selected_L"),
            sampler=samp,
            target=targ,
            output_dir=output_dir,
            save_format=save_format,
        )


def _plot_single_L_analysis(
    grid_results: List[Dict],
    selected_L: Optional[int],
    sampler: str,
    target: str,
    output_dir: str,
    save_format: str = 'png',
) -> None:
    """Plot L analysis for a single sampler-target pair.

    Internal function called by plot_L_selection_analysis.
    """
    # Extract data
    L_values = [r["num_steps"] for r in grid_results]
    ess_per_grad = [r.get("ess_per_gradient", 0) for r in grid_results]
    w2 = [r.get("sliced_w2") for r in grid_results]
    ess_tail = [r.get("ess_tail_min", 0) for r in grid_results]
    rhat_max = [r.get("rhat_max") for r in grid_results]
    accept_rate = [r.get("accept_rate") for r in grid_results]
    warmup_time = [r.get("warmup_time") for r in grid_results]

    # Create figure
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=PLOT_DPI)
    fig.suptitle(f'Trajectory Length Analysis: {sampler.upper()} on {target}',
                 fontsize=14, fontweight='bold')

    # Row 1: Efficiency metrics
    # (0,0) ESS/gradient vs L
    ax = axes[0, 0]
    ax.plot(L_values, ess_per_grad, marker='o', linewidth=LINE_WIDTH, markersize=6)
    if selected_L is not None and selected_L in L_values:
        idx = L_values.index(selected_L)
        ax.scatter([selected_L], [ess_per_grad[idx]], color='red', s=MARKER_SIZE,
                   zorder=5, label=f'Selected L={selected_L}', marker='*')
    ax.set_xlabel('Trajectory Length (L)')
    ax.set_ylabel('ESS / Gradient')
    ax.set_title('Efficiency: ESS per Gradient')
    ax.grid(True, alpha=0.3)
    if selected_L is not None:
        ax.legend()

    # (0,1) W2 distance vs L
    ax = axes[0, 1]
    w2_valid = [(l, w) for l, w in zip(L_values, w2) if w is not None]
    if w2_valid:
        L_w2, w2_vals = zip(*w2_valid)
        ax.plot(L_w2, w2_vals, marker='o', linewidth=LINE_WIDTH, markersize=6, color='orange')
        if selected_L is not None and selected_L in L_w2:
            idx = L_w2.index(selected_L)
            ax.scatter([selected_L], [w2_vals[idx]], color='red', s=MARKER_SIZE,
                       zorder=5, marker='*')
    ax.set_xlabel('Trajectory Length (L)')
    ax.set_ylabel('Sliced W2 Distance')
    ax.set_title('Quality: W2 Distance to Reference')
    ax.grid(True, alpha=0.3)

    # (0,2) ESS_tail vs L
    ax = axes[0, 2]
    ax.plot(L_values, ess_tail, marker='o', linewidth=LINE_WIDTH, markersize=6, color='green')
    if selected_L is not None and selected_L in L_values:
        idx = L_values.index(selected_L)
        ax.scatter([selected_L], [ess_tail[idx]], color='red', s=MARKER_SIZE,
                   zorder=5, marker='*')
    ax.set_xlabel('Trajectory Length (L)')
    ax.set_ylabel('ESS Tail (min)')
    ax.set_title('Tail Behavior: ESS Tail Min')
    ax.grid(True, alpha=0.3)

    # Row 2: Stability and cost
    # (1,0) R-hat vs L
    ax = axes[1, 0]
    rhat_valid = [(l, r) for l, r in zip(L_values, rhat_max) if r is not None]
    if rhat_valid:
        L_rhat, rhat_vals = zip(*rhat_valid)
        ax.plot(L_rhat, rhat_vals, marker='o', linewidth=LINE_WIDTH, markersize=6, color='purple')
        if selected_L is not None and selected_L in L_rhat:
            idx = L_rhat.index(selected_L)
            ax.scatter([selected_L], [rhat_vals[idx]], color='red', s=MARKER_SIZE,
                       zorder=5, marker='*')
    ax.axhline(y=1.01, color='darkgreen', linestyle='--', label='Quality threshold (1.01)', linewidth=1.5)
    ax.axhline(y=1.05, color='orange', linestyle='--', label='Usable threshold (1.05)', linewidth=1.5)
    ax.set_xlabel('Trajectory Length (L)')
    ax.set_ylabel('R-hat (max)')
    ax.set_title('Convergence: R-hat Maximum')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (1,1) Accept rate vs L
    ax = axes[1, 1]
    accept_valid = [(l, a) for l, a in zip(L_values, accept_rate) if a is not None]
    if accept_valid:
        L_accept, accept_vals = zip(*accept_valid)
        ax.plot(L_accept, accept_vals, marker='o', linewidth=LINE_WIDTH, markersize=6, color='brown')
        if selected_L is not None and selected_L in L_accept:
            idx = L_accept.index(selected_L)
            ax.scatter([selected_L], [accept_vals[idx]], color='red', s=MARKER_SIZE,
                       zorder=5, marker='*')
    # Target acceptance rate depends on sampler
    if sampler.lower() == 'rwmh':
        target_accept = 0.234
    else:  # HMC, NUTS, GRAHMC
        target_accept = 0.65
    ax.axhline(y=target_accept, color='gray', linestyle='--',
               label=f'Target ({target_accept:.3f})', linewidth=1.5)
    ax.set_xlabel('Trajectory Length (L)')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Tuning Health: Acceptance Rate')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (1,2) Warmup time vs L
    ax = axes[1, 2]
    warmup_valid = [(l, t) for l, t in zip(L_values, warmup_time) if t is not None]
    if warmup_valid:
        L_warmup, warmup_vals = zip(*warmup_valid)
        ax.plot(L_warmup, warmup_vals, marker='o', linewidth=LINE_WIDTH, markersize=6, color='teal')
        if selected_L is not None and selected_L in L_warmup:
            idx = L_warmup.index(selected_L)
            ax.scatter([selected_L], [warmup_vals[idx]], color='red', s=MARKER_SIZE,
                       zorder=5, marker='*')
    ax.set_xlabel('Trajectory Length (L)')
    ax.set_ylabel('Warmup Time (seconds)')
    ax.set_title('Cost: Warmup Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    filename = f"L_analysis_{sampler}_{target}.{save_format}"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {filepath}")

    if save_format == 'both':
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f"L_analysis_{sampler}_{target}.{fmt}")
            plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"Saved: {filepath}")

    plt.close(fig)


def plot_L_winner_distribution(
    results: List[Dict],
    output_dir: str,
    save_format: str = 'png',
) -> None:
    """Plot aggregate view of optimal L values across all targets.

    Creates:
    1. Histogram of winning L values
    2. Heatmap of (Target × Sampler) → optimal L

    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save plots
        save_format: Output format ('png', 'pdf', or 'both')
    """
    results = filter_with_grid_search(results)

    if not results:
        print("WARNING: No results with grid search data found!")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract selected L values
    selected_L_values = [r.get("selected_L") for r in results if r.get("selected_L") is not None]

    if not selected_L_values:
        print("No selected L values found in results")
        return

    # Create histogram
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=PLOT_DPI)

    ax.hist(selected_L_values, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Selected Trajectory Length (L)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Winning L Values Across All Benchmarks')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_L = np.mean(selected_L_values)
    median_L = np.median(selected_L_values)
    ax.axvline(mean_L, color='red', linestyle='--', label=f'Mean: {mean_L:.1f}', linewidth=2)
    ax.axvline(median_L, color='orange', linestyle='--', label=f'Median: {median_L:.1f}', linewidth=2)
    ax.legend()

    plt.tight_layout()

    # Save
    filename = f"L_winner_histogram.{save_format}"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {filepath}")

    if save_format == 'both':
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f"L_winner_histogram.{fmt}")
            plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')

    plt.close(fig)

    # Create heatmap of (Target × Sampler) → optimal L
    _plot_L_winner_heatmap(results, output_dir, save_format)


def _plot_L_winner_heatmap(
    results: List[Dict],
    output_dir: str,
    save_format: str = 'png',
) -> None:
    """Plot heatmap of optimal L values by target and sampler."""
    # Build matrix
    targets = sorted(set(r["target"] for r in results))
    samplers = sorted(set(r["sampler"] for r in results))

    matrix = np.full((len(targets), len(samplers)), np.nan)
    for i, target in enumerate(targets):
        for j, sampler in enumerate(samplers):
            matches = [r for r in results
                       if r["target"] == target and r["sampler"] == sampler]
            if matches:
                selected_L = matches[0].get("selected_L")
                if selected_L is not None:
                    matrix[i, j] = selected_L

    # Create heatmap
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(samplers)*1.5),
                                           max(6, len(targets)*0.6)),
                           dpi=PLOT_DPI)

    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='viridis',
                xticklabels=samplers, yticklabels=targets,
                cbar_kws={'label': 'Selected L'}, ax=ax)

    ax.set_xlabel('Sampler')
    ax.set_ylabel('Target')
    ax.set_title('Optimal Trajectory Length (L) by Target and Sampler')

    plt.tight_layout()

    # Save
    filename = f"L_winner_heatmap.{save_format}"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {filepath}")

    if save_format == 'both':
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f"L_winner_heatmap.{fmt}")
            plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')

    plt.close(fig)


def analyze_L_grid_coverage(results: List[Dict]) -> Dict:
    """Analyze whether the tested L grid is adequate.

    Args:
        results: List of benchmark result dictionaries

    Returns:
        Dictionary with coverage analysis statistics
    """
    results = filter_with_grid_search(results)

    if not results:
        print("WARNING: No results with grid search data found!")
        return {}

    boundary_winners = 0
    total_with_L = 0

    for r in results:
        selected_L = r.get("selected_L")
        grid_results = r.get("grid_results", [])

        if not grid_results or selected_L is None:
            continue

        total_with_L += 1
        L_values = [gr["num_steps"] for gr in grid_results]
        L_min, L_max = min(L_values), max(L_values)

        # Check if winner is at boundary
        if selected_L == L_min or selected_L == L_max:
            boundary_winners += 1
            print(f"  Boundary winner: {r['sampler']}/{r['target']} - "
                  f"selected L={selected_L} (range: [{L_min}, {L_max}])")

    boundary_rate = boundary_winners / total_with_L if total_with_L > 0 else 0

    analysis = {
        "total_runs": total_with_L,
        "boundary_winners": boundary_winners,
        "boundary_rate": boundary_rate,
        "recommendation": _get_grid_recommendation(boundary_rate),
    }

    # Print summary
    print("\n" + "="*60)
    print("L GRID COVERAGE ANALYSIS")
    print("="*60)
    print(f"Total runs analyzed: {total_with_L}")
    print(f"Winners at grid boundary: {boundary_winners} ({boundary_rate:.1%})")
    print(f"\nRecommendation: {analysis['recommendation']}")
    print("="*60 + "\n")

    return analysis


def _get_grid_recommendation(boundary_rate: float) -> str:
    """Generate recommendation based on boundary winner rate."""
    if boundary_rate < 0.05:
        return "Grid coverage is excellent (< 5% at boundary)"
    elif boundary_rate < 0.15:
        return "Grid coverage is good (< 15% at boundary)"
    elif boundary_rate < 0.30:
        return "Consider expanding grid range (15-30% at boundary)"
    else:
        return "Grid range likely insufficient (>30% at boundary) - expand tested L values"
