#!/usr/bin/env python3
"""Aggregate completed JESSI-S2R and DWA run metrics for one campaign."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import POLICIES, atomic_write_json, read_json, utc_now  # type: ignore
else:
    from .common import POLICIES, atomic_write_json, read_json, utc_now


LEGACY_POLICIES = ("JESSI", "DWA")


METRIC_COHORTS = (
    ("time_to_goal_s", "successful"),
    ("path_length_m", "successful"),
    ("average_jerk_m_s3", "all"),
    ("average_jerk_m_s3", "successful"),
    ("space_compliance", "all"),
    ("space_compliance", "successful"),
    ("minimum_human_clearance_m", "all"),
    ("minimum_human_clearance_m", "successful"),
)


def finite_values(rows, policy, metric, cohort):
    values = []
    for row in rows:
        if row["policy"] != policy or (cohort == "successful" and not row["success"]):
            continue
        if not row.get("synchronization_valid", True):
            continue
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values)


def bootstrap_mean_ci(values, rng, samples):
    if not len(values):
        return None, None
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def descriptive(values, rng, bootstrap_samples):
    if not len(values):
        return {key: None for key in ("mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high")}
    low, high = bootstrap_mean_ci(values, rng, bootstrap_samples)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "ci95_low": low,
        "ci95_high": high,
    }


def bootstrap_difference(jessi, dwa, rng, samples):
    if not len(jessi) or not len(dwa):
        return {"mean_difference": None, "ci95_low": None, "ci95_high": None}
    j_draw = rng.choice(jessi, size=(samples, len(jessi)), replace=True).mean(axis=1)
    d_draw = rng.choice(dwa, size=(samples, len(dwa)), replace=True).mean(axis=1)
    differences = j_draw - d_draw
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(jessi) - np.mean(dwa)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def aggregate(campaign_dir: Path) -> dict:
    config = read_json(campaign_dir / "campaign_config.json")
    bootstrap_seed = int(config.get("bootstrap_seed", 20260831))
    bootstrap_samples = int(config.get("bootstrap_samples", 10000))
    rows = [read_json(path) for path in sorted(campaign_dir.glob("run_*/metrics.json"))]
    if not rows:
        raise ValueError("No run metrics found")
    observed_policies = {row["policy"] for row in rows}
    policies = POLICIES if POLICIES[0] in observed_policies else LEGACY_POLICIES
    learned_policy, baseline_policy = policies

    all_fields = sorted({key for row in rows for key in row if not isinstance(row[key], dict)})
    with (campaign_dir / "campaign_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in all_fields} for row in rows)

    rng = np.random.default_rng(bootstrap_seed)
    summaries, comparisons = [], {}
    for metric, cohort in METRIC_COHORTS:
        comparison_key = f"{metric}:{cohort}"
        policy_values = {}
        for policy in policies:
            values = finite_values(rows, policy, metric, cohort)
            policy_values[policy] = values
            summaries.append(
                {
                    "metric": metric,
                    "cohort": cohort,
                    "policy": policy,
                    "n": len(values),
                    **descriptive(values, rng, bootstrap_samples),
                }
            )
        comparisons[comparison_key] = bootstrap_difference(
            policy_values[learned_policy], policy_values[baseline_policy], rng, bootstrap_samples
        )

    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        for metric, predicate in (
            ("success_rate", lambda row: row["success"]),
            ("operator_collision_rate", lambda row: row["operator_collision"]),
            ("timeout_rate", lambda row: row["timeout"]),
        ):
            values = np.asarray([float(predicate(row)) for row in policy_rows])
            summaries.append(
                {
                    "metric": metric,
                    "cohort": "all",
                    "policy": policy,
                    "n": len(values),
                    **descriptive(values, rng, bootstrap_samples),
                }
            )

    summary_fields = ("metric", "cohort", "policy", "n", "mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high")
    with (campaign_dir / "policy_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "runs_found": len(rows),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        (
            "jessi_s2r_minus_dwa"
            if learned_policy == "JESSI-S2R"
            else "jessi_minus_dwa"
        ): comparisons,
    }
    atomic_write_json(campaign_dir / "policy_comparison.json", result)
    plot_results(rows, campaign_dir / "policy_comparison.png", policies)
    return result


def plot_results(rows, output_path, policies=POLICIES):
    plots = (
        ("time_to_goal_s", "Time to goal [s]", "successful"),
        ("path_length_m", "Path length [m]", "successful"),
        ("average_jerk_m_s3", "Average jerk [m/s³]", "all"),
        ("space_compliance", "Space compliance", "all"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (metric, label, cohort) in zip(axes.flat, plots):
        data = [finite_values(rows, policy, metric, cohort) for policy in policies]
        if all(len(values) for values in data):
            axis.boxplot(data, tick_labels=policies, showmeans=True)
        axis.set_title(label)
        axis.grid(True, axis="y", linestyle="--", alpha=0.5)
    figure.suptitle("TurtleBot4 corridor experiment")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate(args.campaign_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
