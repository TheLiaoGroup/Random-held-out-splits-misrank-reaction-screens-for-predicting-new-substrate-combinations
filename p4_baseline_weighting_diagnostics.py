"""Recompute the saved-RF median-baseline and weighting diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEEDS = (42, 20261007, 20261008, 20261009, 20261010)
PROTOCOLS = ("row", "group")


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)


def rho(row: pd.Series, group: pd.Series) -> float | None:
    if len(row) < 2 or row.nunique() < 2 or group.nunique() < 2:
        return None
    return float(spearmanr(row, group).statistic)


def pair_weighted_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Give each recorded pair equal weight while retaining all test records."""
    pair_counts = frame.pair_key.value_counts()
    inverse_frequency = frame.pair_key.map(frame.pair_key.value_counts()).rdiv(1.0)
    weights = inverse_frequency.to_numpy(float)
    target = frame.y_true.to_numpy(float)
    prediction = frame.y_pred.to_numpy(float)
    weighted_mean = float(np.average(target, weights=weights))
    weighted_sse = float(np.sum(weights * (target - prediction) ** 2))
    weighted_sst = float(np.sum(weights * (target - weighted_mean) ** 2))
    return {
        "pairs": int(pair_counts.size),
        "r2": float(1.0 - weighted_sse / weighted_sst)
        if weighted_sst > 0 else None,
        "mae": float(np.average(frame.abs_error, weights=weights)),
        "rmse": float(np.sqrt(np.average(frame.sq_error, weights=weights))),
    }


def load_truth_frames(package: Path, population):
    from p4_rf_oof_diagnostics import _truth_frames
    return _truth_frames(package, population)


def run(package: Path, results: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sys.path.insert(0, str(package))
    from score_predictions import load_population

    population = load_population(package)
    truth_frames = load_truth_frames(package, population)
    baseline_rows = []
    weighting_rows = []

    for task in population.tasks:
        screen = task.name
        predictions = pd.read_csv(
            results / "sources" / slug(screen) / "oof_predictions.csv.gz",
            low_memory=False,
        )
        truth = truth_frames[screen].set_index("row_uid").loc[list(task.row_uids)].reset_index()
        pair_by_uid = truth.set_index("row_uid")["pair_key"]
        both_seen_by_seed = {}
        for seed_index, seed in enumerate(SEEDS):
            group = predictions.loc[
                predictions.protocol.eq("group") & predictions.seed.eq(seed)
            ].set_index("row_uid").loc[list(task.row_uids)]
            both_seen_by_seed[seed] = set(group.index[group.n_unseen.eq(0)])

        for protocol in PROTOCOLS:
            for seed_index, seed in enumerate(SEEDS):
                assigned = task.assignment[protocol][seed_index]
                subset = predictions.loc[
                    predictions.protocol.eq(protocol) & predictions.seed.eq(seed)
                ].set_index("row_uid").loc[list(task.row_uids)].reset_index()
                subset["pair_key"] = subset.row_uid.map(pair_by_uid)
                subset["abs_error"] = np.abs(subset.y_true - subset.y_pred)
                subset["sq_error"] = (subset.y_true - subset.y_pred) ** 2
                median_by_fold = {
                    fold: float(np.median(task.y[assigned != fold]))
                    for fold in range(5)
                }
                subset["training_median"] = subset.fold.astype(int).map(median_by_fold)
                subset["median_abs_error"] = np.abs(
                    subset.y_true - subset.training_median
                )
                model_mae = float(subset.abs_error.mean())
                median_mae = float(subset.median_abs_error.mean())
                baseline_rows.append({
                    "screen": screen,
                    "protocol": protocol,
                    "seed": seed,
                    "rows": int(len(subset)),
                    "pairs": int(subset.pair_key.nunique()),
                    "model_mae_percentage_points": model_mae * 100,
                    "training_median_mae_percentage_points": median_mae * 100,
                    "model_mae_improvement_over_training_median": (
                        1.0 - model_mae / median_mae if median_mae > 0 else None
                    ),
                })
                weighted = pair_weighted_metrics(subset)
                cohort = subset.loc[subset.row_uid.isin(both_seen_by_seed[seed])].copy()
                if len(cohort):
                    cohort_y = cohort.y_true.to_numpy(float)
                    cohort_p = cohort.y_pred.to_numpy(float)
                    cohort_r2 = float(1.0 - np.sum((cohort_y - cohort_p) ** 2)
                                      / np.sum((cohort_y - cohort_y.mean()) ** 2))
                    cohort_mae = float(cohort.abs_error.mean() * 100)
                else:
                    cohort_r2, cohort_mae = None, None
                y = subset.y_true.to_numpy(float)
                model = subset.y_pred.to_numpy(float)
                r2 = 1.0 - np.sum((y - model) ** 2) / np.sum((y - y.mean()) ** 2)
                weighting_rows.append({
                    "screen": screen,
                    "protocol": protocol,
                    "seed": seed,
                    "well_weighted_r2": float(r2),
                    "well_weighted_mae_percentage_points": float(subset.abs_error.mean() * 100),
                    "pair_equal_r2": weighted["r2"],
                    "pair_equal_mae_percentage_points": weighted["mae"] * 100,
                    "pair_equal_rmse_percentage_points": weighted["rmse"] * 100,
                    "both_seen_rows": int(len(cohort)),
                    "both_seen_well_r2": cohort_r2,
                    "both_seen_well_mae_percentage_points": cohort_mae,
                })

    baseline = pd.DataFrame(baseline_rows)
    weighting = pd.DataFrame(weighting_rows)
    screens11 = {task.name for task in population.tasks if task.metadata["mean_yield_fraction"] >= 0.05}
    summary = {"subsets": {}}
    for name, selected in (("all_18", {task.name for task in population.tasks}),
                           ("mean_yield_at_least_5pct", screens11)):
        summary["subsets"][name] = {}
        b = baseline.loc[baseline.screen.isin(selected)]
        w = weighting.loc[weighting.screen.isin(selected)]
        bmean = b.groupby(["screen", "protocol"], as_index=False).mean(numeric_only=True)
        wmean = w.groupby(["screen", "protocol"], as_index=False).mean(numeric_only=True)
        for metric in ("model_mae_percentage_points", "training_median_mae_percentage_points",
                       "model_mae_improvement_over_training_median"):
            table = bmean.pivot(index="screen", columns="protocol", values=metric)
            summary["subsets"][name][f"baseline_{metric}_rho"] = rho(table.row, table.group)
        for metric in ("well_weighted_r2", "well_weighted_mae_percentage_points",
                       "pair_equal_r2",
                       "pair_equal_mae_percentage_points",
                       "both_seen_well_r2", "both_seen_well_mae_percentage_points"):
            table = wmean.pivot(index="screen", columns="protocol", values=metric)
            summary["subsets"][name][f"weighting_{metric}_rho"] = rho(table.row, table.group)
    summary["grignard_acylation"] = {}
    for screen in ("Grignard", "Acylation"):
        table = baseline.loc[baseline.screen.eq(screen)].groupby("protocol").mean(numeric_only=True)
        summary["grignard_acylation"][screen] = table[
            ["model_mae_percentage_points", "training_median_mae_percentage_points",
             "model_mae_improvement_over_training_median"]
        ].to_dict(orient="index")
    return baseline, weighting, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline, weighting, summary = run(args.package_dir.resolve(), args.results_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(args.output_dir / "rf_median_baseline_by_seed.csv", index=False)
    weighting.to_csv(args.output_dir / "rf_weighting_by_seed.csv", index=False)
    (args.output_dir / "rf_baseline_weighting_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
