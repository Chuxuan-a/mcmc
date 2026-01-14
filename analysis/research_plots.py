"""Research-focused analysis and visualization functions.

High-impact plots for publications and presentations:
- Cross-sampler comparisons
- GRAHMC schedule analysis
- Efficiency vs quality trade-offs
- Winner matrices
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import (
    filter_by_sampler,
    filter_quality_only,
    filter_usable_only,
    get_unique_samplers,
    get_unique_targets,
    get_unique_schedules,
)

# Plotting configuration
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)


def plot_sampler_comparison(
    results: List[Dict],
    output_dir: str,
    metric: str = 'all',
    quality_only: bool = True,
    save_format: str = 'png',
) -> None:
    """Generate box plots comparing samplers on key metrics.

    Args:
        results: List of benchmark results
        output_dir: Output directory
        metric: Which metric to plot ('all', 'ess_per_gradient', 'w2', 'rhat', 'time')
        quality_only: If True, only include quality-pass runs
        save_format: Output format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter to quality runs if requested
    if quality_only:
        results = filter_quality_only(results)
        suffix = '_quality'
    else:
        results = filter_usable_only(results)
        suffix = '_usable'

    if not results:
        print(f"No {'quality' if quality_only else 'usable'} runs found!")
        return

    # Prepare data
    df_data = []
    for r in results:
        sampler_label = r['sampler']
        # For GRAHMC, just use the schedule name (not grahmc/schedule)
        if r['sampler'] in ['grahmc', 'rahmc'] and r.get('schedule'):
            sampler_label = r['schedule']

        df_data.append({
            'sampler': sampler_label,
            'target': r['target'],
            'ess_per_gradient': r.get('ess_per_gradient', 0),
            'sliced_w2': r.get('sliced_w2'),
            'rhat_max': r.get('rhat_max'),
            'total_time': r.get('total_time'),
            'ess_bulk_min': r.get('ess_bulk_min'),
        })

    df = pd.DataFrame(df_data)

    # Define metrics to plot
    if metric == 'all':
        metrics_to_plot = [
            ('ess_per_gradient', 'ESS / Gradient', False),  # Higher is better
            ('sliced_w2', 'Sliced W2 Distance', True),      # Lower is better
            ('rhat_max', 'R-hat (max)', True),              # Lower is better
            ('total_time', 'Total Time (s)', True),         # Lower is better
        ]
    else:
        metrics_to_plot = [(metric, metric.replace('_', ' ').title(), False)]

    plt.style.use(PLOT_STYLE)

    for metric_name, ylabel, lower_better in metrics_to_plot:
        # Skip if metric not available
        if df[metric_name].isna().all():
            print(f"Skipping {metric_name} - no data available")
            continue

        # Use white background instead of grey
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=PLOT_DPI, facecolor='white')
        ax.set_facecolor('white')

        # Create box plot - cleaner for publication
        # No title for LaTeX figure captions
        sns.boxplot(
            data=df,
            x='sampler',
            y=metric_name,
            hue='sampler',  # Required for palette in newer seaborn
            ax=ax,
            palette='Set2',
            showfliers=False,  # Hide outliers for cleaner, readable plot
            showmeans=False,   # Don't show mean markers
            legend=False,      # Don't show legend (redundant with x-axis)
        )

        # Publication-ready axis labels with larger fonts
        ax.set_xlabel('Sampler', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Cleaner grid
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.7)
        ax.set_axisbelow(True)  # Grid behind plot elements

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        filename = f"sampler_comparison_{metric_name}{suffix}.{save_format}"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)


def plot_schedule_comparison(
    results: List[Dict],
    output_dir: str,
    save_format: str = 'png',
) -> None:
    """Compare GRAHMC friction schedules head-to-head.

    Args:
        results: List of benchmark results
        output_dir: Output directory
        save_format: Output format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter to GRAHMC only
    grahmc_results = [r for r in results
                      if r['sampler'] in ['grahmc', 'rahmc']
                      and r.get('schedule')]

    if not grahmc_results:
        print("No GRAHMC results found!")
        return

    # Get schedules
    schedules = get_unique_schedules(grahmc_results)

    # Prepare data
    df_data = []
    for r in grahmc_results:
        df_data.append({
            'schedule': r['schedule'],
            'target': r['target'],
            'ess_per_gradient': r.get('ess_per_gradient', 0),
            'sliced_w2': r.get('sliced_w2'),
            'quality_pass': r.get('quality_pass', False),
            'usable': r.get('usable', False),
            'gamma': r.get('gamma'),
            'steepness': r.get('steepness'),
            'accept_rate': r.get('accept_rate'),
        })

    df = pd.DataFrame(df_data)

    plt.style.use(PLOT_STYLE)

    # Create 2x3 comparison grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=PLOT_DPI)
    fig.suptitle('GRAHMC Schedule Comparison', fontsize=16, fontweight='bold')

    # Plot 1: ESS/gradient
    ax = axes[0, 0]
    sns.boxplot(data=df, x='schedule', y='ess_per_gradient', ax=ax, palette='Set2')
    ax.set_title('Efficiency: ESS/Gradient')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: W2 distance (quality runs only)
    ax = axes[0, 1]
    df_quality = df[df['quality_pass'] == True]
    if len(df_quality) > 0:
        sns.boxplot(data=df_quality, x='schedule', y='sliced_w2', ax=ax, palette='Set2')
        ax.set_title('Quality: W2 Distance (Quality Runs)')
    else:
        ax.text(0.5, 0.5, 'No quality runs', ha='center', va='center',
               transform=ax.transAxes)
        ax.set_title('Quality: W2 Distance (No Data)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Win rate
    ax = axes[0, 2]
    win_counts = []
    for schedule in schedules:
        # Count wins per target (best ESS/grad per target)
        wins = 0
        targets = df['target'].unique()
        for target in targets:
            target_data = df[df['target'] == target]
            if len(target_data) > 0:
                best = target_data.loc[target_data['ess_per_gradient'].idxmax()]
                if best['schedule'] == schedule:
                    wins += 1
        win_counts.append(wins)

    ax.bar(schedules, win_counts, color=sns.color_palette('Set2', len(schedules)))
    ax.set_title('Win Rate (Best ESS/Grad per Target)')
    ax.set_ylabel('Number of Targets Won')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Quality pass rate
    ax = axes[1, 0]
    pass_rates = []
    for schedule in schedules:
        sched_data = df[df['schedule'] == schedule]
        pass_rate = sched_data['quality_pass'].mean() * 100
        pass_rates.append(pass_rate)

    ax.bar(schedules, pass_rates, color=sns.color_palette('Set2', len(schedules)))
    ax.set_title('Quality Pass Rate')
    ax.set_ylabel('% Passing Quality Gates')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Plot 5: Acceptance rate
    ax = axes[1, 1]
    sns.boxplot(data=df, x='schedule', y='accept_rate', ax=ax, palette='Set2')
    ax.axhline(y=0.65, color='red', linestyle='--', alpha=0.7, label='Target (0.65)')
    ax.set_title('Acceptance Rate Distribution')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Plot 6: Gamma values distribution
    ax = axes[1, 2]
    if df['gamma'].notna().any():
        sns.violinplot(data=df[df['gamma'].notna()], x='schedule', y='gamma',
                      ax=ax, palette='Set2')
        ax.set_title('Tuned Gamma Distribution')
    else:
        ax.text(0.5, 0.5, 'No gamma data', ha='center', va='center',
               transform=ax.transAxes)
        ax.set_title('Tuned Gamma (No Data)')
    ax.set_xlabel('Schedule')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = f"schedule_comparison.{save_format}"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close(fig)


def plot_efficiency_quality_tradeoff(
    results: List[Dict],
    output_dir: str,
    save_format: str = 'png',
) -> None:
    """Plot efficiency vs quality trade-off scatter.

    Args:
        results: List of benchmark results
        output_dir: Output directory
        save_format: Output format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter to runs with W2 data
    results_with_w2 = [r for r in results if r.get('sliced_w2') is not None]

    if not results_with_w2:
        print("No results with W2 data found!")
        return

    # Prepare data
    df_data = []
    for r in results_with_w2:
        sampler_label = r['sampler']
        if r['sampler'] in ['grahmc', 'rahmc'] and r.get('schedule'):
            sampler_label = f"{r['sampler']}/{r['schedule']}"

        df_data.append({
            'sampler': sampler_label,
            'target': r['target'],
            'ess_per_gradient': r.get('ess_per_gradient', 0),
            'sliced_w2': r.get('sliced_w2'),
            'quality_pass': r.get('quality_pass', False),
            'usable': r.get('usable', False),
        })

    df = pd.DataFrame(df_data)

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=PLOT_DPI)

    # Color by quality
    colors = df['quality_pass'].map({True: 'green', False: 'red'})
    markers = df['quality_pass'].map({True: 'o', False: 'x'})

    # Scatter plot
    for quality in [False, True]:
        subset = df[df['quality_pass'] == quality]
        label = 'Quality Pass' if quality else 'Failed Quality'
        marker = 'o' if quality else 'x'
        color = 'green' if quality else 'red'

        ax.scatter(subset['ess_per_gradient'], subset['sliced_w2'],
                  c=color, marker=marker, s=100, alpha=0.6,
                  label=label, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('ESS / Gradient (higher is better)', fontsize=12)
    ax.set_ylabel('Sliced W2 Distance (lower is better)', fontsize=12)
    ax.set_title('Efficiency vs Quality Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add ideal region annotation
    ax.text(0.95, 0.05, 'Ideal:\nHigh ESS/grad\nLow W2',
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)

    plt.tight_layout()

    filename = f"efficiency_quality_tradeoff.{save_format}"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close(fig)


def generate_winner_matrix(
    results: List[Dict],
    output_dir: str,
    save_format: str = 'png',
) -> None:
    """Generate winner matrix heatmap showing best sampler per target.

    Args:
        results: List of benchmark results
        output_dir: Output directory
        save_format: Output format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only consider quality runs
    quality_results = filter_quality_only(results)

    if not quality_results:
        print("No quality runs found!")
        return

    # Build winner matrix
    targets = sorted(get_unique_targets(quality_results))
    samplers = []

    # Determine winners per target
    winner_data = {}
    for target in targets:
        target_results = [r for r in quality_results if r['target'] == target]

        if not target_results:
            continue

        # Find best by ESS/gradient
        best = max(target_results, key=lambda r: r.get('ess_per_gradient', 0))

        sampler_label = best['sampler']
        if best['sampler'] in ['grahmc', 'rahmc'] and best.get('schedule'):
            sampler_label = f"{best['sampler']}/{best['schedule']}"

        if sampler_label not in samplers:
            samplers.append(sampler_label)

        winner_data[target] = {
            'winner': sampler_label,
            'ess_per_gradient': best.get('ess_per_gradient', 0),
            'w2': best.get('sliced_w2'),
        }

    # Create matrix for heatmap
    samplers = sorted(samplers)
    matrix = np.zeros((len(targets), len(samplers)))

    for i, target in enumerate(targets):
        if target in winner_data:
            winner = winner_data[target]['winner']
            j = samplers.index(winner)
            matrix[i, j] = winner_data[target]['ess_per_gradient']

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(samplers)*1.2),
                                           max(8, len(targets)*0.6)),
                           dpi=PLOT_DPI)

    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=samplers, yticklabels=targets,
                cbar_kws={'label': 'ESS/Gradient (Winner)'}, ax=ax)

    ax.set_xlabel('Sampler', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title('Winner Matrix: Best Sampler per Target (by ESS/Gradient)',
                fontsize=14, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    filename = f"winner_matrix.{save_format}"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close(fig)


def generate_summary_table(
    results: List[Dict],
    output_dir: str,
) -> None:
    """Generate summary statistics table.

    Args:
        results: List of benchmark results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect stats by sampler
    samplers_data = []

    # Get all sampler labels
    sampler_labels = set()
    for r in results:
        label = r['sampler']
        if r['sampler'] in ['grahmc', 'rahmc'] and r.get('schedule'):
            label = f"{r['sampler']}/{r['schedule']}"
        sampler_labels.add(label)

    for label in sorted(sampler_labels):
        # Filter results for this sampler
        if '/' in label:
            sampler, schedule = label.split('/', 1)
            sampler_results = [r for r in results
                             if r['sampler'] == sampler and r.get('schedule') == schedule]
        else:
            sampler_results = [r for r in results if r['sampler'] == label]

        if not sampler_results:
            continue

        # Calculate stats
        total = len(sampler_results)
        usable = sum(1 for r in sampler_results if r.get('usable', False))
        quality = sum(1 for r in sampler_results if r.get('quality_pass', False))

        ess_grads = [r.get('ess_per_gradient', 0) for r in sampler_results if r.get('ess_per_gradient')]
        w2s = [r.get('sliced_w2') for r in sampler_results if r.get('sliced_w2') is not None]

        samplers_data.append({
            'Sampler': label,
            'Total': total,
            'Usable': usable,
            'Quality': quality,
            'Quality %': f"{quality/total*100:.1f}%" if total > 0 else "0%",
            'Median ESS/Grad': f"{np.median(ess_grads):.6f}" if ess_grads else "N/A",
            'Median W2': f"{np.median(w2s):.4f}" if w2s else "N/A",
        })

    df = pd.DataFrame(samplers_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save to markdown
    md_path = os.path.join(output_dir, 'summary_table.md')
    with open(md_path, 'w') as f:
        f.write("# Benchmark Summary Statistics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved: {md_path}")
