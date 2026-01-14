"""Utility functions for loading and filtering benchmark results."""

import json
import os
from typing import Dict, List, Optional


def load_benchmark_results(results_path: str) -> List[Dict]:
    """Load and parse benchmark results from JSON file.

    Args:
        results_path: Directory containing benchmark_results.json

    Returns:
        List of result dictionaries with expanded grid_search_info

    Raises:
        FileNotFoundError: If benchmark_results.json doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    json_path = os.path.join(results_path, "benchmark_results.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No benchmark results found at {json_path}. "
            f"Run benchmarks first or check the path."
        )

    with open(json_path, 'r') as f:
        results = json.load(f)

    # Expand grid_search_info for easier access
    for r in results:
        if "grid_search_info" in r and r["grid_search_info"] is not None:
            r["grid_results"] = r["grid_search_info"].get("all_results", [])
            r["selected_L"] = r["grid_search_info"].get("selected_L")
            r["selection_tier"] = r["grid_search_info"].get("selection_tier")
            r["has_grid_search"] = len(r["grid_results"]) > 0
        else:
            r["grid_results"] = []
            r["selected_L"] = r.get("num_steps")  # Fallback for non-grid runs
            r["selection_tier"] = None
            r["has_grid_search"] = False

    return results


def filter_by_sampler(results: List[Dict], sampler: str) -> List[Dict]:
    """Filter results by sampler name.

    Args:
        results: List of result dictionaries
        sampler: Sampler name (e.g., 'hmc', 'nuts', 'grahmc')

    Returns:
        Filtered list of results
    """
    return [r for r in results if r.get("sampler") == sampler]


def filter_by_target(results: List[Dict], target: str) -> List[Dict]:
    """Filter results by target distribution name.

    Args:
        results: List of result dictionaries
        target: Target name (e.g., 'StandardNormal10D', 'rosenbrock')

    Returns:
        Filtered list of results

    Note:
        Target names are case-sensitive and include dimension suffix
    """
    return [r for r in results if r.get("target") == target]


def filter_by_schedule(results: List[Dict], schedule: str) -> List[Dict]:
    """Filter GRAHMC results by friction schedule type.

    Args:
        results: List of result dictionaries
        schedule: Schedule type (e.g., 'tanh', 'constant', 'sine')

    Returns:
        Filtered list of results (only GRAHMC/RAHMC with matching schedule)
    """
    return [r for r in results
            if r.get("sampler") in ["grahmc", "rahmc"]
            and r.get("schedule") == schedule]


def filter_usable_only(results: List[Dict]) -> List[Dict]:
    """Filter to only usable runs (pass hard quality gates).

    Args:
        results: List of result dictionaries

    Returns:
        List of results where usable=True

    Note:
        Usable criteria: rhat < 1.05, ess_bulk >= 400, ess_tail >= 100
    """
    return [r for r in results if r.get("usable", False)]


def filter_quality_only(results: List[Dict]) -> List[Dict]:
    """Filter to only high-quality runs (pass strict quality gates).

    Args:
        results: List of result dictionaries

    Returns:
        List of results where quality_pass=True

    Note:
        Quality criteria: rhat < 1.01, ess_bulk >= 400, ess_tail >= 400,
        divergence < 1%, z-score test pass
    """
    return [r for r in results if r.get("quality_pass", False)]


def filter_with_grid_search(results: List[Dict]) -> List[Dict]:
    """Filter to only runs with grid search data.

    Args:
        results: List of result dictionaries

    Returns:
        List of results with non-empty grid_results

    Note:
        Only results generated after grid_search_info enhancement will have this data
    """
    return [r for r in results if r.get("has_grid_search", False)]


def get_unique_samplers(results: List[Dict]) -> List[str]:
    """Extract unique sampler names from results.

    Args:
        results: List of result dictionaries

    Returns:
        Sorted list of unique sampler names
    """
    samplers = set(r.get("sampler") for r in results if r.get("sampler"))
    return sorted(samplers)


def get_unique_targets(results: List[Dict]) -> List[str]:
    """Extract unique target names from results.

    Args:
        results: List of result dictionaries

    Returns:
        Sorted list of unique target names
    """
    targets = set(r.get("target") for r in results if r.get("target"))
    return sorted(targets)


def get_unique_schedules(results: List[Dict]) -> List[str]:
    """Extract unique GRAHMC schedule types from results.

    Args:
        results: List of result dictionaries

    Returns:
        Sorted list of unique schedule names (only from GRAHMC/RAHMC runs)
    """
    schedules = set(
        r.get("schedule") for r in results
        if r.get("sampler") in ["grahmc", "rahmc"] and r.get("schedule")
    )
    return sorted(schedules)


def summarize_results(results: List[Dict]) -> Dict:
    """Generate summary statistics for result dataset.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with summary statistics
    """
    total_runs = len(results)
    usable_runs = len(filter_usable_only(results))
    quality_runs = len(filter_quality_only(results))
    grid_search_runs = len(filter_with_grid_search(results))

    return {
        "total_runs": total_runs,
        "usable_runs": usable_runs,
        "quality_runs": quality_runs,
        "grid_search_runs": grid_search_runs,
        "usable_rate": usable_runs / total_runs if total_runs > 0 else 0,
        "quality_rate": quality_runs / total_runs if total_runs > 0 else 0,
        "grid_search_rate": grid_search_runs / total_runs if total_runs > 0 else 0,
        "samplers": get_unique_samplers(results),
        "targets": get_unique_targets(results),
        "schedules": get_unique_schedules(results),
    }
