"""Independently validate and summarize the unified RF held-out predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SEEDS = (42, 20261007, 20261008, 20261009, 20261010)
STRATA = ("both_seen", "one_unseen", "both_unseen")
METRICS = ("r2", "mae", "rmse")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def signature(value):
    content = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def loss_metrics(y_true, y_pred, training_mean):
    y, p, b = [np.asarray(values, dtype=float) for values in (y_true, y_pred, training_mean)]
    if any(values.ndim != 1 for values in (y, p, b)) or not (len(y) == len(p) == len(b)):
        raise ValueError("Loss arrays must be one-dimensional and have matching lengths")
    if not all(np.isfinite(values).all() for values in (y, p, b)):
        raise ValueError("Nonfinite true yield, prediction or reference")
    n = len(y)
    if not n:
        return {"rows": 0, "sse": 0.0, "sae": 0.0, "sst": 0.0,
                **{key: None for key in ("r2", "mae", "rmse", "mean_yield_fraction",
                                         "yield_sd", "zero_fraction", "training_mean_baseline_sse",
                                         "training_mean_baseline_mae", "training_mean_baseline_rmse",
                                         "training_mean_baseline_r2", "skill_vs_training_mean")}}
    residual, baseline_residual = y - p, y - b
    sse, sae = float(np.sum(residual ** 2)), float(np.sum(np.abs(residual)))
    sst = float(np.sum((y - y.mean()) ** 2))
    baseline_sse = float(np.sum(baseline_residual ** 2))
    defined_r2 = not bool(np.all(y == y[0])) and sst > 0
    return {"rows": n, "sse": sse, "sae": sae, "sst": sst,
            "r2": 1 - sse / sst if defined_r2 else None,
            "mae": sae / n, "rmse": float(np.sqrt(sse / n)),
            "mean_yield_fraction": float(y.mean()), "yield_sd": float(y.std(ddof=0)),
            "zero_fraction": float(np.mean(y == 0)),
            "training_mean_baseline_sse": baseline_sse,
            "training_mean_baseline_mae": float(np.mean(np.abs(baseline_residual))),
            "training_mean_baseline_rmse": float(np.sqrt(baseline_sse / n)),
            "training_mean_baseline_r2": 1 - baseline_sse / sst if defined_r2 else None,
            "skill_vs_training_mean": 1 - sse / baseline_sse if baseline_sse > 0 else None}


def stratified_metrics(frame):
    if not frame.n_unseen.isin([0, 1, 2]).all():
        raise ValueError("Invalid role-specific substrate exposure category")
    total = loss_metrics(frame.y_true, frame.y_pred, frame.training_mean)
    rows = []
    for number, label in enumerate(STRATA):
        subset = frame.loc[frame.n_unseen == number]
        metrics = loss_metrics(subset.y_true, subset.y_pred, subset.training_mean)
        metrics.update({"n_unseen": number, "stratum": label,
                        "row_share": len(subset) / len(frame) if len(frame) else None,
                        "sse_share": metrics["sse"] / total["sse"]
                        if len(subset) and total["sse"] > 0 else None,
                        "sae_share": metrics["sae"] / total["sae"]
                        if len(subset) and total["sae"] > 0 else None})
        rows.append(metrics)
    for loss in ("sae", "sse"):
        if not np.isclose(sum(row[loss] for row in rows), total[loss], rtol=1e-12, atol=1e-12):
            raise ValueError("Stratum losses do not recover pooled losses")
    return rows


def matched_cohort_metrics(frame):
    records = []
    for seed in sorted(frame.seed.unique()):
        group = frame.loc[frame.seed.eq(seed) & frame.protocol.eq("group")].set_index("row_uid")
        row = frame.loc[frame.seed.eq(seed) & frame.protocol.eq("row")].set_index("row_uid")
        if (not group.index.is_unique or not row.index.is_unique or not len(group)
                or set(group.index) != set(row.index)):
            raise ValueError("Paired protocols require identical unique record identities")
        row = row.reindex(group.index)
        if not np.array_equal(row.y_true, group.y_true):
            raise ValueError("Paired protocols require identical true yields")
        seed_records = []
        for number, label in enumerate(STRATA):
            selected = group.n_unseen.eq(number)
            record = {"screen": group.screen.iloc[0], "seed": int(seed),
                      "group_exposure_stratum": label, "n_unseen": number,
                      "rows": int(selected.sum()), "row_share": float(selected.mean())}
            losses = {}
            for kind, predictions in (("row", row), ("group", group)):
                subset = predictions.loc[selected]
                losses[kind] = loss_metrics(subset.y_true, subset.y_pred, subset.training_mean)
                record.update({f"{kind}_{key}": value for key, value in losses[kind].items()
                               if key != "rows"})
            for loss in ("sae", "sse"):
                record[f"group_minus_row_{loss}"] = losses["group"][loss] - losses["row"][loss]
            record["group_minus_row_mae"] = losses["group"]["mae"] - losses["row"]["mae"] if selected.any() else None
            seed_records.append(record)
        for loss in ("sae", "sse"):
            total = {kind: loss_metrics(data.y_true, data.y_pred, data.training_mean)[loss]
                     for kind, data in (("row", row), ("group", group))}
            if not np.isclose(sum(r[f"group_minus_row_{loss}"] for r in seed_records),
                              total["group"] - total["row"], rtol=1e-12, atol=1e-12):
                raise ValueError("Matched-cohort changes do not recover pooled loss change")
        records.extend(seed_records)
    return records


def validate_oof(frame, truth, assignments, screen):
    required = {"screen", "row_uid", "protocol", "seed", "fold", "y_true", "y_pred",
                "n_unseen", "training_mean"}
    if not required.issubset(frame.columns) or truth.row_uid.duplicated().any():
        raise ValueError("Missing OOF fields or duplicate frozen identities")
    if not frame.screen.eq(screen).all() or frame.duplicated(["row_uid", "protocol", "seed"]).any():
        raise ValueError("Wrong screen or duplicate OOF prediction")
    observed = set(zip(frame.protocol, frame.seed))
    if observed != set(assignments):
        raise ValueError("Unexpected or missing OOF protocol/seed")
    y = truth.yield_fraction.to_numpy(dtype=float)
    for (protocol, seed), assigned in assignments.items():
        subset = frame.loc[frame.protocol.eq(protocol) & frame.seed.eq(seed)].set_index("row_uid")
        if len(subset) != len(truth) or set(subset.index) != set(truth.row_uid):
            raise ValueError("OOF predictions do not cover the exact frozen population")
        subset = subset.reindex(truth.row_uid)
        numeric = subset[["y_true", "y_pred", "training_mean", "fold", "n_unseen"]].to_numpy(float)
        if not np.isfinite(numeric).all() or not np.allclose(subset.y_true, y, rtol=0, atol=1e-14):
            raise ValueError("Nonfinite OOF values or changed true yields")
        if not np.array_equal(subset.fold.to_numpy(), assigned):
            raise ValueError("OOF fold differs from frozen membership")
        for fold in np.unique(assigned):
            train, test = assigned != fold, assigned == fold
            if not train.any() or not test.any():
                raise ValueError("Empty training or test partition")
            exposure = np.zeros(test.sum(), dtype=np.int8)
            for column in ("sub_1_smiles", "sub_2_smiles"):
                exposure += ~truth.loc[test, column].isin(set(truth.loc[train, column])).to_numpy()
            if not np.array_equal(exposure, subset.loc[test, "n_unseen"].to_numpy()):
                raise ValueError("Exposure category differs from numerical training rows")
            if not np.allclose(subset.loc[test, "training_mean"], y[train].mean(), rtol=0, atol=1e-14):
                raise ValueError("Training reference was not calculated on its training fold")


def distribution(values):
    data = np.asarray([value for value in values if value is not None], dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite summary metric")
    if not len(data):
        return {"count": 0, "mean": None, "min": None, "max": None, "range": None}
    return {"count": len(data), "mean": float(data.mean()), "min": float(data.min()),
            "max": float(data.max()), "range": float(np.ptp(data))}


def screen_order(sources, protocol, metric, seed=None):
    direction = -1 if metric == "r2" else 1
    def score(source):
        if seed is None:
            return source["summary"][protocol][metric]["mean"]
        return next(row[metric] for row in source["pooled_by_seed"][protocol] if row["seed"] == seed)
    return [source["source_name"] for source in
            sorted(sources, key=lambda source: (direction * score(source), source["source_name"]))]


def rank_rho(sources, metric, seed=None):
    def score(source, protocol):
        if seed is None:
            return source["summary"][protocol][metric]["mean"]
        return next(row[metric] for row in source["pooled_by_seed"][protocol] if row["seed"] == seed)
    row = [score(source, "row") for source in sources]
    group = [score(source, "group") for source in sources]
    if len(sources) < 2 or len(set(row)) < 2 or len(set(group)) < 2:
        return None
    return float(spearmanr(row, group).statistic)


def rank_summary(sources):
    output = {}
    for name, subset in (("all_screens", sources),
                         ("mean_yield_at_least_5pct", [s for s in sources if s["mean_yield_fraction"] >= 0.05]),
                         ("mean_yield_at_least_10pct", [s for s in sources if s["mean_yield_fraction"] >= 0.10])):
        deltas = [source["summary"]["delta_r2"]["mean"] for source in subset]
        output[name] = {"screens": len(subset), "positive_delta_screens": sum(d > 0 for d in deltas),
                        "median_delta_r2": float(np.median(deltas)),
                        "metric_rho": {metric: rank_rho(subset, metric) for metric in METRICS},
                        "r2_order": {kind: screen_order(subset, kind, "r2") for kind in ("row", "group")},
                        "by_seed": [{"seed": seed,
                                     "metric_rho": {metric: rank_rho(subset, metric, seed) for metric in METRICS},
                                     "r2_order": {kind: screen_order(subset, kind, "r2", seed)
                                                  for kind in ("row", "group")}} for seed in SEEDS]}
    return output


def summarize_screen(frame, metadata):
    source = {key: metadata[key] for key in ("source_name", "reaction_family",
                                            "numeric_records", "mean_yield_fraction", "floor_source")}
    source["pooled_by_seed"], source["summary"] = {}, {}
    conditional = []
    for kind in ("row", "group"):
        pooled = []
        for seed in SEEDS:
            subset = frame.loc[frame.protocol.eq(kind) & frame.seed.eq(seed)]
            metrics = loss_metrics(subset.y_true, subset.y_pred, subset.training_mean)
            pooled.append({"seed": seed, **metrics})
            for row in stratified_metrics(subset):
                conditional.append({"screen": source["source_name"], "protocol": kind, "seed": seed, **row})
        source["pooled_by_seed"][kind] = pooled
        source["summary"][kind] = {metric: distribution([row[metric] for row in pooled])
                                    for metric in (*METRICS, "skill_vs_training_mean",
                                                   "training_mean_baseline_r2")}
    deltas = [{"seed": seed, "row_minus_group": row["r2"] - group["r2"]}
              for seed, row, group in zip(SEEDS, source["pooled_by_seed"]["row"], source["pooled_by_seed"]["group"])]
    source["summary"]["delta_r2"] = {**distribution([row["row_minus_group"] for row in deltas]),
                                       "by_seed": deltas}
    return source, conditional


def _truth_frames(package, population):
    """Per-screen released records, in the frozen row order of each scoring task."""
    from score_predictions import read_records
    records = read_records(package)
    frames = {}
    for task in population.tasks:
        frame = records.loc[records.screen.eq(task.name)].reset_index(drop=True)
        if tuple(frame.row_uid) != task.row_uids:
            raise ValueError(f"{task.name}: released records differ from the scoring population")
        frames[task.name] = frame
    return frames


def validate_output_dir(package, results, output):
    package, results, output = [Path(path).resolve() for path in (package, results, output)]
    if (output.is_relative_to(package) or package.is_relative_to(output)
            or results.is_relative_to(output)):
        raise ValueError("Diagnostics must use a separate output directory outside the frozen package")
    for protected in (package.parent / "submission", package.parent / "internal" / "source-construction",
                      package.parent / "internal" / "experimental-reference"):
        if output.is_relative_to(protected.resolve()):
            raise ValueError("Diagnostics must not overwrite submission material or frozen sources")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package, results, output = [path.resolve() for path in (args.package_dir, args.results_dir, args.output_dir)]
    validate_output_dir(package, results, output)
    sys.path.insert(0, str(package))
    from score_predictions import load_population, check_inputs
    from p4_saved_score_diagnostics_20261008 import build_diagnostics
    population = load_population(package)
    frames = _truth_frames(package, population)
    run_manifest = json.loads((results / "run_manifest.json").read_text("utf-8"))
    if run_manifest["status"] != "completed":
        raise ValueError("All RF fits must be complete before final diagnostics")
    configuration = run_manifest.get("configuration")
    configuration_signature = run_manifest.get("configuration_signature")
    if (not isinstance(configuration_signature, str) or len(configuration_signature) != 64
            or any(character not in "0123456789abcdef" for character in configuration_signature)):
        raise ValueError("Missing or invalid RF configuration signature")
    if configuration is not None and signature(configuration) != configuration_signature:
        raise ValueError("RF manifest configuration signature changed")
    input_hashes = {**population.input_hashes, "rf_run_manifest": sha256(results / "run_manifest.json"),
                    "rf_configuration_signature": configuration_signature}
    configuration_path = results / "frozen_rf_configuration.json"
    if configuration_path.exists():
        frozen_configuration = json.loads(configuration_path.read_text("utf-8"))
        if ((configuration is not None and frozen_configuration != configuration)
                or signature(frozen_configuration) != configuration_signature):
            raise ValueError("RF frozen configuration differs from the run manifest")
        input_hashes["rf_frozen_configuration"] = sha256(configuration_path)
    elif run_manifest.get("release_kind") != "final_oof_not_resumable":
        raise ValueError("Missing RF frozen configuration for resumable results")
    sources, conditional, paired_cohorts, comparison, hashes = [], [], [], [], {}
    entries = {entry["source_name"]: entry for entry in run_manifest["sources"]}
    if set(entries) != {task.name for task in population.tasks}:
        raise ValueError("RF results do not cover the frozen 18 screens")
    for task in population.tasks:
        artifact = entries[task.name]["oof_predictions"]
        path = (results / artifact["path"]).resolve()
        if not path.is_relative_to(results) or sha256(path) != artifact["sha256"]:
            raise ValueError("OOF prediction artifact path or hash changed")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != artifact["rows"]:
            raise ValueError("OOF prediction file row count changed")
        assignments = {(kind, seed): task.assignment[kind][index]
                       for kind in ("row", "group") for index, seed in enumerate(SEEDS)}
        validate_oof(frame, frames[task.name], assignments, task.name)
        source, strata = summarize_screen(frame, task.metadata)
        sources.append(source)
        conditional.extend(strata)
        paired_cohorts.extend(matched_cohort_metrics(frame))
        hashes[task.name] = artifact["sha256"]
        row = {"screen": task.name, "numeric_records": len(task.y),
               "mean_yield_percent": task.metadata["mean_yield_fraction"] * 100}
        for kind in ("row", "group"):
            for metric in METRICS:
                for name, item in (("original", task.metadata), ("rf", source)):
                    value = item["summary"][kind][metric]["mean"]
                    row[f"{name}_{kind}_{metric}"] = value * (1 if metric == "r2" else 100)
            row[f"rf_{kind}_skill_vs_training_mean"] = source["summary"][kind]["skill_vs_training_mean"]["mean"]
        row["original_delta_r2"] = task.metadata["summary"]["delta_r2"]["mean"]
        row["rf_delta_r2"] = source["summary"]["delta_r2"]["mean"]
        comparison.append(row)
    check_inputs(population)
    payload = {"status": "completed", "protocol": {"seeds": list(SEEDS), "model": "unified RF100",
               "script_sha256": sha256(Path(__file__)),
               "substrate_exposure_identity": "original recorded SMILES; same role; numerical training rows only"},
               "input_sha256": input_hashes,
               "sources": sources, "prediction_sha256": hashes}
    ranks = {"original": rank_summary(list(population.reference["sources"])), "unified_rf": rank_summary(sources)}
    output.mkdir(parents=True, exist_ok=True)
    pooled_path = output / "rf_pooled_model_results.json"
    pooled_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    diagnostics = build_diagnostics(payload, pooled_path, sha256(pooled_path))
    # The shared ranking helper names its original pooled producer by default.
    diagnostics["provenance"]["pooled_evaluation_script_path"] = Path(__file__).resolve().as_posix()
    diagnostics["provenance"]["pooled_evaluation_script_sha256_at_input_creation"] = payload["protocol"]["script_sha256"]
    for name, content in (("rf_saved_score_diagnostics.json", diagnostics), ("rf_model_comparison.json", ranks)):
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(comparison).to_csv(output / "rf_model_comparison.csv", index=False)
    pd.DataFrame(conditional).to_csv(output / "rf_substrate_strata_by_seed.csv", index=False)
    pd.DataFrame(paired_cohorts).to_csv(output / "rf_matched_cohorts_by_seed.csv", index=False)
    summary_rows = []
    table = pd.DataFrame(conditional)
    for (screen, protocol, stratum), subset in table.groupby(["screen", "protocol", "stratum"], sort=False):
        summary = {"screen": screen, "protocol": protocol, "stratum": stratum}
        for key in ("rows", "row_share", "mae", "rmse", "training_mean_baseline_mae",
                    "training_mean_baseline_rmse", "skill_vs_training_mean", "sae_share", "sse_share"):
            values = [None if pd.isna(value) else float(value) for value in subset[key]]
            summary.update({f"{key}_{stat}": value for stat, value in distribution(values).items()})
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(output / "rf_substrate_strata_summary.csv", index=False)
    report = {"screens": len(sources), "prediction_rows": sum(len(task.y) * 10 for task in population.tasks),
              "pooled_seed_protocol_groups": len(sources) * 10, "stratum_seed_protocol_groups": len(conditional),
              "original_baseline_sha256": sha256(package / "data/p4_screens.json"),
              "rank_results": ranks["unified_rf"]}
    (output / "validation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
