#!/usr/bin/env python
"""Command-line interface for analyzing MCMC benchmark results.

This script provides various analysis and visualization tools for benchmark results,
with focus on grid search analysis (trajectory length selection).

Usage:
    # L analysis for all sampler-target pairs
    python analyze_benchmarks.py results_20d --L-analysis --output plots/

    # L analysis for specific sampler
    python analyze_benchmarks.py results_20d --L-analysis --sampler hmc --output plots/

    # L winner distribution
    python analyze_benchmarks.py results_20d --L-winners --output plots/

    # Grid coverage analysis
    python analyze_benchmarks.py results_20d --L-coverage

    # Run all L analyses
    python analyze_benchmarks.py results_20d --L-all --output plots/

    # Get summary statistics
    python analyze_benchmarks.py results_20d --summary
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analysis.utils import (
    load_benchmark_results,
    summarize_results,
    filter_with_grid_search,
)
from analysis.grid_analysis import (
    plot_L_selection_analysis,
    plot_L_winner_distribution,
    analyze_L_grid_coverage,
)
from analysis.research_plots import (
    plot_sampler_comparison,
    plot_schedule_comparison,
    plot_efficiency_quality_tradeoff,
    generate_winner_matrix,
    generate_summary_table,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze MCMC benchmark results with focus on grid search analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing benchmark_results.json",
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis options")
    analysis_group.add_argument(
        "--L-analysis",
        action="store_true",
        help="Generate L selection analysis plots (2x3 grid per sampler-target)",
    )
    analysis_group.add_argument(
        "--L-winners",
        action="store_true",
        help="Plot distribution of winning L values",
    )
    analysis_group.add_argument(
        "--L-coverage",
        action="store_true",
        help="Analyze L grid coverage (terminal output)",
    )
    analysis_group.add_argument(
        "--L-all",
        action="store_true",
        help="Run all L analyses (equivalent to --L-analysis --L-winners --L-coverage)",
    )
    analysis_group.add_argument(
        "--research",
        action="store_true",
        help="Generate research-focused plots (sampler comparison, schedule analysis, trade-offs)",
    )
    analysis_group.add_argument(
        "--sampler-comparison",
        action="store_true",
        help="Generate cross-sampler comparison box plots",
    )
    analysis_group.add_argument(
        "--schedule-comparison",
        action="store_true",
        help="Generate GRAHMC schedule comparison plots",
    )
    analysis_group.add_argument(
        "--tradeoff",
        action="store_true",
        help="Generate efficiency vs quality trade-off scatter plot",
    )
    analysis_group.add_argument(
        "--winner-matrix",
        action="store_true",
        help="Generate winner matrix heatmap",
    )
    analysis_group.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics of results",
    )

    # Filtering options
    filter_group = parser.add_argument_group("Filtering options")
    filter_group.add_argument(
        "--sampler",
        type=str,
        default=None,
        help="Filter to specific sampler (e.g., hmc, nuts, grahmc)",
    )
    filter_group.add_argument(
        "--target",
        type=str,
        default=None,
        help="Filter to specific target (e.g., StandardNormal10D, rosenbrock)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        default="analysis_output",
        help="Output directory for plots (default: analysis_output)",
    )
    output_group.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "both"],
        default="png",
        help="Output format for plots (default: png)",
    )

    args = parser.parse_args()

    # Validate: at least one analysis must be specified
    if not any([args.L_analysis, args.L_winners, args.L_coverage, args.L_all,
                args.research, args.sampler_comparison, args.schedule_comparison,
                args.tradeoff, args.winner_matrix, args.summary]):
        parser.error("At least one analysis option must be specified")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    print("="*70)
    print("MCMC BENCHMARK ANALYSIS")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print()

    # Load results
    try:
        results = load_benchmark_results(args.results_dir)
        print(f"Loaded {len(results)} benchmark results\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR loading results: {e}")
        return 1

    # Print summary if requested
    if args.summary:
        print("="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        summary = summarize_results(results)
        print(f"Total runs: {summary['total_runs']}")
        print(f"Usable runs: {summary['usable_runs']} ({summary['usable_rate']:.1%})")
        print(f"Quality runs: {summary['quality_runs']} ({summary['quality_rate']:.1%})")
        print(f"Grid search runs: {summary['grid_search_runs']} ({summary['grid_search_rate']:.1%})")
        print(f"\nSamplers: {', '.join(summary['samplers'])}")
        print(f"Targets: {', '.join(summary['targets'])}")
        if summary['schedules']:
            print(f"Schedules (GRAHMC): {', '.join(summary['schedules'])}")
        print()

    # Check if any grid search results exist
    grid_results = filter_with_grid_search(results)
    if not grid_results:
        print("WARNING: No results with grid search data found!")
        print()
        print("Grid search analysis requires benchmark results with enhanced logging.")
        print("These results were likely generated before grid_search_info was added.")
        print()
        print("To generate compatible results:")
        print("  python run_benchmarks.py --dim 20 --output-dir new_results")
        print()
        return 1

    print(f"Found {len(grid_results)} runs with grid search data\n")

    # Handle --L-all flag
    if args.L_all:
        args.L_analysis = True
        args.L_winners = True
        args.L_coverage = True

    # Run L selection analysis
    if args.L_analysis:
        print("="*70)
        print("L SELECTION ANALYSIS (2x3 grid plots)")
        print("="*70)
        try:
            plot_L_selection_analysis(
                results=results,
                output_dir=args.output,
                sampler=args.sampler,
                target=args.target,
                save_format=args.format,
            )
            print()
        except Exception as e:
            print(f"ERROR in L selection analysis: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Run L winner distribution analysis
    if args.L_winners:
        print("="*70)
        print("L WINNER DISTRIBUTION")
        print("="*70)
        try:
            plot_L_winner_distribution(
                results=results,
                output_dir=args.output,
                save_format=args.format,
            )
            print()
        except Exception as e:
            print(f"ERROR in L winner analysis: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Run L grid coverage analysis
    if args.L_coverage:
        try:
            analysis = analyze_L_grid_coverage(results)
        except Exception as e:
            print(f"ERROR in L coverage analysis: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Handle --research flag (run all research plots)
    if args.research:
        args.sampler_comparison = True
        args.schedule_comparison = True
        args.tradeoff = True
        args.winner_matrix = True

    # Run sampler comparison
    if args.sampler_comparison:
        print("="*70)
        print("CROSS-SAMPLER COMPARISON")
        print("="*70)
        try:
            plot_sampler_comparison(
                results=results,
                output_dir=args.output,
                metric='all',
                quality_only=True,
                save_format=args.format,
            )
            print()
        except Exception as e:
            print(f"ERROR in sampler comparison: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Run schedule comparison
    if args.schedule_comparison:
        print("="*70)
        print("GRAHMC SCHEDULE COMPARISON")
        print("="*70)
        try:
            plot_schedule_comparison(
                results=results,
                output_dir=args.output,
                save_format=args.format,
            )
            print()
        except Exception as e:
            print(f"ERROR in schedule comparison: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Run efficiency-quality trade-off plot
    if args.tradeoff:
        print("="*70)
        print("EFFICIENCY VS QUALITY TRADE-OFF")
        print("="*70)
        try:
            plot_efficiency_quality_tradeoff(
                results=results,
                output_dir=args.output,
                save_format=args.format,
            )
            print()
        except Exception as e:
            print(f"ERROR in trade-off plot: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Run winner matrix
    if args.winner_matrix:
        print("="*70)
        print("WINNER MATRIX")
        print("="*70)
        try:
            generate_winner_matrix(
                results=results,
                output_dir=args.output,
                save_format=args.format,
            )
            generate_summary_table(
                results=results,
                output_dir=args.output,
            )
            print()
        except Exception as e:
            print(f"ERROR in winner matrix: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Plots saved to: {args.output}/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
