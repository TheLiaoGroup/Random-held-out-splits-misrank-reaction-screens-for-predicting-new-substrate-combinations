"""Score external predictions on the complete, frozen P4 evaluation tasks.

The tasks are read from the released data: data/p4_screens.json (per-screen metadata and
condition vocabularies), data/p4_records.csv.gz (the 340,178 numerical records) and
data/p4_grouped_folds.csv.gz (the frozen new-combination fold of every record and seed).
Held-out-well folds are shuffled five-fold KFold splits of each screen's records in file order.

CSV columns are screen,row_uid,protocol,seed,fold,y_pred. Predictions use yield
fractions; finite values outside [0, 1] are retained without clipping. A scored
file must cover every numerical record once under each of two protocols and
five seeds. A template may select a smaller task for generating predictions.

The scorer validates populations, held-out memberships and coverage. It does
not verify how an external model was trained or whether its training leaked
test information. No estimator is fitted and no baseline artifact is changed.
"""

from __future__ import annotations

import argparse
from array import array
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import platform
from typing import TextIO
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import ConstantInputWarning
import sklearn
from sklearn.model_selection import KFold

from p4_saved_score_diagnostics_20261008 import build_diagnostics


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 20261007, 20261008, 20261009, 20261010)
PROTOCOLS = ("row", "group")
N_SPLITS = 5
FIELDS = ("screen", "row_uid", "protocol", "seed", "fold", "y_pred")
EXPECTED_NUMERIC_ROWS = 340178
EXPECTED_PREDICTIONS = EXPECTED_NUMERIC_ROWS * len(SEEDS) * len(PROTOCOLS)
SCREENS = "data/p4_screens.json"
RECORDS = "data/p4_records.csv.gz"
FOLDS = "data/p4_grouped_folds.csv.gz"
TRAINING_LIMIT = "Coverage and held-out identity are verified; external model training is not verified."


FLOOR = 0.05


def distribution(values: list) -> dict:
    finite = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(finite):
        return {"count": 0, "mean": None, "sd": None, "min": None, "max": None, "range": None}
    if not np.isfinite(finite).all():
        raise ValueError("non-finite metric")
    return {
        "count": len(finite), "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(finite.min()), "max": float(finite.max()),
        "range": float(finite.max() - finite.min()),
    }


def cross_source_summary(sources: list[dict], exclude_floor: bool = False) -> dict:
    selected = [s for s in sources if not exclude_floor or s["mean_yield_fraction"] >= FLOOR]
    deltas = [s["summary"]["delta_r2"]["mean"] for s in selected]
    valid = [d for d in deltas if d is not None]
    seed_ranges = [s["summary"]["group"]["r2"].get("range") for s in selected]
    seed_ranges = [r for r in seed_ranges if r is not None]
    return {
        "sources": len(selected), "valid_delta_sources": len(valid),
        "positive_delta_sources": sum(d > 0 for d in valid),
        "median_delta_r2": float(np.median(valid)) if valid else None,
        "median_group_seed_r2_range": float(np.median(seed_ranges)) if seed_ranges else None,
        "negative_group_mean_r2_sources": [s["source_name"] for s in selected
                                           if s["summary"]["group"]["r2"]["mean"] is not None
                                           and s["summary"]["group"]["r2"]["mean"] < 0],
        "source_names": [s["source_name"] for s in selected],
        "weighting": "equal source weights; five overlapping seed estimates are descriptive",
    }


def csv_row(source: dict) -> dict:
    row = {k: source[k] for k in ("source_name", "reaction_family", "numeric_records", "floor_source")}
    row["mean_yield_percent"] = source["mean_yield_fraction"] * 100
    for kind in ("group", "row"):
        summary = source["summary"][kind]
        for statistic in ("mean", "min", "max", "range"):
            row[f"{kind}_pooled_r2_{statistic}"] = summary["r2"][statistic]
        for metric in ("mae", "rmse"):
            row[f"{kind}_pooled_{metric}_pp"] = summary[metric]["mean"] * 100
        row[f"{kind}_training_mean_baseline_r2"] = summary["training_mean_baseline_r2"]["mean"]
        row[f"{kind}_skill_vs_training_mean"] = summary["skill_vs_training_mean"]["mean"]
        row[f"{kind}_auxiliary_fold_r2_mean"] = summary["auxiliary_fold_r2"]["mean"]
    row.update({f"pooled_delta_r2_{k}": source["summary"]["delta_r2"][k] for k in ("mean", "min", "max")})
    return row


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_path(root: Path, relative: str) -> Path:
    """Resolve preserved Windows-relative manifest paths on either platform."""
    parts = PureWindowsPath(relative).parts
    if PureWindowsPath(relative).is_absolute() or ".." in parts:
        raise ValueError(f"Expected a package-relative path: {relative}")
    resolved = root.joinpath(*parts).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes the package: {relative}")
    return resolved


def open_csv(path: Path, mode: str = "rt"):
    opener = gzip.open if path.suffix.lower() == ".gz" else path.open
    if opener is gzip.open:
        return opener(path, mode, encoding="utf-8-sig" if mode == "rt" else "utf-8", newline="")
    return opener(mode, encoding="utf-8-sig" if mode == "rt" else "utf-8", newline="")


def checked_json(root: Path, relative: str, hashes: dict[str, str], expected: str | None = None) -> dict:
    path = local_path(root, relative)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Frozen artifact hash mismatch: {relative}")
    hashes[relative.replace("\\", "/")] = digest
    return json.loads(content)


def checked_artifact(root: Path, artifact: dict, hashes: dict[str, str]) -> Path:
    path = local_path(root, artifact["path"])
    if sha256(path) != artifact["sha256"]:
        raise ValueError(f"Frozen artifact hash mismatch: {artifact['path']}")
    hashes[artifact["path"].replace("\\", "/")] = artifact["sha256"]
    return path


@dataclass
class FoldTruth:
    rows: int
    train_rows: int
    sst: float
    constant: bool
    training_mean: float
    baseline_sse: float


@dataclass
class ScreenTask:
    metadata: dict
    row_uids: tuple[str, ...]
    uid_index: dict[str, int]
    y: np.ndarray
    assignment: dict[str, list[np.ndarray]]
    fold_truth: dict[tuple[str, int, int], FoldTruth]
    sst: float
    sum_y: float
    sum_y_squared: float

    @property
    def name(self) -> str:
        return self.metadata["source_name"]


@dataclass
class FrozenPopulation:
    root: Path
    tasks: tuple[ScreenTask, ...]
    input_hashes: dict[str, str]
    reference: dict


def _screen_task(metadata: dict, frame: pd.DataFrame) -> ScreenTask:
    uids = tuple(frame.row_uid.astype(str))
    uid_index = {uid: index for index, uid in enumerate(uids)}
    y = frame.yield_fraction.to_numpy(dtype=np.float64)
    if len(uid_index) != len(uids) or len(y) != metadata["numeric_records"] or not np.isfinite(y).all():
        raise ValueError(f"{metadata['source_name']}: invalid frozen numerical population")
    average = float(y.mean())
    sst = float(np.sum((y - average) ** 2))
    if not math.isclose(average, metadata["mean_yield_fraction"], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"{metadata['source_name']}: frozen mean yield changed")
    if not math.isclose(sst, metadata["target_sst"], rel_tol=1e-12, abs_tol=1e-12) or sst <= 0:
        raise ValueError(f"{metadata['source_name']}: frozen target variance changed")
    assignment = {kind: [np.full(len(y), -1, dtype=np.int8) for _ in SEEDS] for kind in PROTOCOLS}
    for seed_index, seed in enumerate(SEEDS):
        for fold, (_, test) in enumerate(KFold(N_SPLITS, shuffle=True, random_state=seed).split(np.arange(len(y)))):
            assignment["row"][seed_index][test] = fold
    return ScreenTask(metadata, uids, uid_index, y, assignment, {}, sst,
                      float(y.sum()), float(np.sum(y ** 2)))


def _load_registry(path: Path, tasks: list[ScreenTask], expected_rows: int) -> None:
    lookup = {uid: (task, index) for task in tasks for uid, index in task.uid_index.items()}
    if len(lookup) != sum(len(task.y) for task in tasks):
        raise ValueError("Registry tasks have overlapping row identifiers")
    seed_indices = {seed: index for index, seed in enumerate(SEEDS)}
    row_count = 0
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        if not {"row_uid", "seed", "fold"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing frozen registry columns: {path.name}")
        for row in reader:
            row_count += 1
            matched = lookup.get(row["row_uid"])
            if matched is None:
                continue
            task, index = matched
            seed, fold = int(row["seed"]), int(row["fold"])
            if seed not in seed_indices or fold not in range(N_SPLITS):
                raise ValueError(f"{task.name}: invalid frozen group assignment")
            if row.get("screen", row.get("reaction", task.name)) != task.name:
                raise ValueError(f"{task.name}: registry screen identity mismatch")
            assignment = task.assignment["group"][seed_indices[seed]]
            if assignment[index] != -1:
                raise ValueError(f"{task.name}: duplicate frozen group assignment")
            assignment[index] = fold
    if row_count != expected_rows:
        raise ValueError(f"Frozen registry row count changed: {path.name}")
    if any(np.any(assigned < 0) for task in tasks for assigned in task.assignment["group"]):
        raise ValueError("Frozen registry does not cover every numerical row and seed")


def _fold_truth(task: ScreenTask, plates: np.ndarray, pairs: np.ndarray) -> None:
    """Record fold statistics; grouped folds must share no plate or substrate pair across train and test."""
    for kind in PROTOCOLS:
        for seed_index, seed in enumerate(SEEDS):
            assigned = task.assignment[kind][seed_index]
            for fold in range(N_SPLITS):
                test = np.flatnonzero(assigned == fold)
                train = np.flatnonzero(assigned != fold)
                if not len(test) or not len(train):
                    raise ValueError(f"{task.name}: empty frozen fold")
                if kind == "group" and (set(plates[test]) & set(plates[train]) or set(pairs[test]) & set(pairs[train])):
                    raise ValueError(f"{task.name}: grouped fold shares a plate or pair between train and test")
                test_y, train_y = task.y[test], task.y[train]
                train_mean = float(train_y.mean())
                task.fold_truth[kind, seed, fold] = FoldTruth(
                    len(test), len(train), float(np.sum((test_y - test_y.mean()) ** 2)),
                    bool(np.all(test_y == test_y[0])), train_mean,
                    float(np.sum((test_y - train_mean) ** 2)),
                )


def read_records(root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the released numerical records with exact floating-point yields."""
    return pd.read_csv(local_path(root, RECORDS), usecols=columns, low_memory=False, float_precision="round_trip",
                       dtype={"screen": str, "row_uid": str, "plate_group": str, "component_group": str,
                              "pair_key": str, "condition_id": str})


def load_population(root: Path = ROOT) -> FrozenPopulation:
    """Load the released truth and frozen folds and validate them."""
    root = root.resolve()
    hashes: dict[str, str] = {}
    reference = checked_json(root, SCREENS, hashes)
    if (tuple(reference["protocol"]["seeds"]) != SEEDS
            or len(reference["sources"]) != 18 or len({s["source_name"] for s in reference["sources"]}) != 18
            or sum(s["numeric_records"] for s in reference["sources"]) != EXPECTED_NUMERIC_ROWS
            or len({s["reaction_family"] for s in reference["sources"]}) != 17):
        raise ValueError("Unexpected frozen P4 population")
    hashes[RECORDS] = sha256(local_path(root, RECORDS))
    records = read_records(root, ["screen", "row_uid", "plate_group", "pair_key", "yield_fraction"])
    if len(records) != EXPECTED_NUMERIC_ROWS or records.row_uid.duplicated().any():
        raise ValueError("Released record count or row identity changed")
    tasks, groups = [], {}
    for metadata in reference["sources"]:
        frame = records.loc[records.screen.eq(metadata["source_name"])]
        tasks.append(_screen_task(metadata, frame))
        groups[metadata["source_name"]] = (frame.plate_group.fillna("").to_numpy(), frame.pair_key.fillna("").to_numpy())
    if sum(len(task.y) for task in tasks) != len(records):
        raise ValueError("Records belong to screens outside the released metadata")
    hashes[FOLDS] = sha256(local_path(root, FOLDS))
    manifest_path = root / "RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("RELEASE_MANIFEST.json is missing; the released data cannot be verified")
    expected = {entry["path"]: entry["sha256"] for entry in json.loads(manifest_path.read_text("utf-8"))["files"]}
    for relative in (SCREENS, RECORDS, FOLDS):
        if expected.get(relative) != hashes[relative]:
            raise ValueError(f"{relative} does not match its SHA-256 in RELEASE_MANIFEST.json")
    _load_registry(local_path(root, FOLDS), tasks, EXPECTED_NUMERIC_ROWS * len(SEEDS))
    for task in tasks:
        _fold_truth(task, *groups[task.name])
    population = FrozenPopulation(root, tuple(tasks), hashes, reference)
    check_inputs(population)
    return population


def check_inputs(population: FrozenPopulation) -> None:
    for relative, expected in population.input_hashes.items():
        if sha256(local_path(population.root, relative)) != expected:
            raise ValueError(f"Frozen artifact changed during scoring: {relative}")


class CompensatedSum:
    """Small, order-robust accumulator for a streamed nonnegative loss sum."""

    __slots__ = ("total", "correction")

    def __init__(self):
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        combined = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - combined) + value
        else:
            self.correction += (value - combined) + self.total
        self.total = combined

    def value(self) -> float:
        result = self.total + self.correction
        if not math.isfinite(result):
            raise ValueError("Accumulated prediction loss is not finite")
        return result


def evaluate_csv(handle: TextIO, population: FrozenPopulation, model_name: str) -> dict:
    """Stream a complete prediction file; its order need not match the template."""
    reader = csv.reader(handle)
    headers = next(reader, None)
    if headers is None or len(headers) != len(FIELDS) or set(headers) != set(FIELDS):
        raise ValueError("Prediction header must contain exactly: " + ",".join(FIELDS))
    columns = {name: headers.index(name) for name in FIELDS}
    task_indices = {task.name: index for index, task in enumerate(population.tasks)}
    seed_indices = {seed: index for index, seed in enumerate(SEEDS)}
    seen = [array("H", [0]) * len(task.y) for task in population.tasks]
    losses = [[(CompensatedSum(), CompensatedSum()) for _ in range(len(PROTOCOLS) * len(SEEDS) * N_SPLITS)]
              for _ in population.tasks]
    accepted = 0
    for row in reader:
        where = f"Prediction CSV line {reader.line_num}"
        if len(row) != len(FIELDS):
            raise ValueError(f"{where}: expected six fields")
        name, uid, kind = (row[columns[key]] for key in ("screen", "row_uid", "protocol"))
        if name not in task_indices:
            raise ValueError(f"{where}: unknown screen {name!r}")
        task_index = task_indices[name]
        task = population.tasks[task_index]
        numeric_index = task.uid_index.get(uid)
        if numeric_index is None:
            raise ValueError(f"{where}: unknown or nonnumerical row_uid {uid!r} for {name}")
        if kind not in PROTOCOLS:
            raise ValueError(f"{where}: protocol must be row or group")
        try:
            seed, fold = int(row[columns["seed"]]), int(row[columns["fold"]])
            prediction = float(row[columns["y_pred"]])
        except ValueError as error:
            raise ValueError(f"{where}: seed/fold must be integers and y_pred a finite number") from error
        if seed not in seed_indices or fold not in range(N_SPLITS):
            raise ValueError(f"{where}: unknown seed or fold")
        if not math.isfinite(prediction):
            raise ValueError(f"{where}: y_pred must be finite")
        seed_index = seed_indices[seed]
        expected_fold = int(task.assignment[kind][seed_index][numeric_index])
        if fold != expected_fold:
            raise ValueError(f"{where}: wrong fold for {name}/{uid}/{kind}/{seed}; expected {expected_fold}")
        task_slot = PROTOCOLS.index(kind) * len(SEEDS) + seed_index
        bit = 1 << task_slot
        if seen[task_index][numeric_index] & bit:
            raise ValueError(f"{where}: duplicate prediction for {name}/{uid}/{kind}/{seed}")
        residual = float(task.y[numeric_index]) - prediction
        squared = residual * residual
        if not math.isfinite(residual) or not math.isfinite(squared):
            raise ValueError(f"{where}: prediction produces a nonfinite loss")
        seen[task_index][numeric_index] |= bit
        sse, sae = losses[task_index][task_slot * N_SPLITS + fold]
        sse.add(squared)
        sae.add(abs(residual))
        accepted += 1
    full_mask = (1 << (len(PROTOCOLS) * len(SEEDS))) - 1
    if accepted != EXPECTED_PREDICTIONS or any(any(mask != full_mask for mask in masks) for masks in seen):
        examples = []
        for task, masks in zip(population.tasks, seen):
            for index, mask in enumerate(masks):
                if mask != full_mask:
                    examples.append(f"{task.name}/{task.row_uids[index]}")
                    if len(examples) == 5:
                        break
            if len(examples) == 5:
                break
        raise ValueError(f"Incomplete coverage: {accepted:,}/{EXPECTED_PREDICTIONS:,} predictions; missing tasks include "
                         + ", ".join(examples))
    sources = []
    for task_index, task in enumerate(population.tasks):
        pooled, summaries = {}, {}
        for protocol_index, kind in enumerate(PROTOCOLS):
            seed_scores, auxiliary = [], []
            for seed_index, seed in enumerate(SEEDS):
                totals_sse, totals_sae, baselines = [], [], []
                undefined = 0
                for fold in range(N_SPLITS):
                    sse, sae = losses[task_index][(protocol_index * len(SEEDS) + seed_index) * N_SPLITS + fold]
                    fold_sse, fold_sae = sse.value(), sae.value()
                    truth = task.fold_truth[kind, seed, fold]
                    auxiliary.append(None if truth.constant else float(1 - fold_sse / truth.sst))
                    undefined += int(truth.constant)
                    totals_sse.append(fold_sse)
                    totals_sae.append(fold_sae)
                    baselines.append(truth.baseline_sse)
                sse, sae, baseline = math.fsum(totals_sse), math.fsum(totals_sae), math.fsum(baselines)
                seed_scores.append({
                    "seed": seed, "rows": len(task.y), "sse": sse, "sae": sae, "sst": task.sst,
                    "sum_y": task.sum_y, "sum_y_squared": task.sum_y_squared,
                    "r2": float(1 - sse / task.sst), "mae": float(sae / len(task.y)),
                    "rmse": float(math.sqrt(sse / len(task.y))), "training_mean_baseline_sse": baseline,
                    "training_mean_baseline_r2": float(1 - baseline / task.sst),
                    "skill_vs_training_mean": float(1 - sse / baseline) if baseline else None,
                    "undefined_fold_r2_count": undefined,
                })
            pooled[kind] = seed_scores
            summaries[kind] = {metric: distribution([record[metric] for record in seed_scores])
                               for metric in ("r2", "mae", "rmse", "training_mean_baseline_r2", "skill_vs_training_mean")}
            summaries[kind]["auxiliary_fold_r2"] = distribution(auxiliary)
        deltas = [{"seed": seed, "row_minus_group": r["r2"] - g["r2"]}
                  for seed, r, g in zip(SEEDS, pooled["row"], pooled["group"])]
        summaries["delta_r2"] = {**distribution([row["row_minus_group"] for row in deltas]), "by_seed": deltas}
        metadata = task.metadata
        sources.append({
            "source_name": task.name, "reaction_family": metadata["reaction_family"],
            "numeric_records": len(task.y),
            "mean_yield_fraction": float(task.y.mean()), "floor_source": bool(task.y.mean() < 0.05),
            "target_sst": task.sst, "yield_sd_percentage_points": 100 * math.sqrt(task.sst / len(task.y)),
            "provenance": {"frozen_population": SCREENS, "prediction_model": model_name},
            "pooled_by_seed": pooled, "summary": summaries,
        })
    return {
        "status": "completed", "model_name": model_name,
        "protocol": {
            "seeds": list(SEEDS), "n_splits": N_SPLITS, "script_sha256": sha256(Path(__file__)),
            "primary": "per-seed pooled out-of-fold R2, MAE and RMSE; five seed means and ranges",
            "prediction_schema": list(FIELDS), "prediction_units": "yield fraction; no clipping",
            "coverage": "all 18 screens, both protocols, five seeds; each numerical row exactly once",
            "sst": "sum((yield - whole numerical screen mean)**2)",
            "yield_sd": "population SD: 100 * sqrt(SST / numerical row count), in percentage points",
            "training_validation": TRAINING_LIMIT,
            "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                  "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__},
        },
        "input_sha256": dict(population.input_hashes), "prediction_rows": accepted,
        "sources": sources, "all_sources_summary": cross_source_summary(sources),
        "nonfloor_sources_summary": cross_source_summary(sources, exclude_floor=True),
    }


def score_file(path: Path, population: FrozenPopulation, model_name: str) -> dict:
    """Score one complete CSV or CSV.gz without writing any output."""
    before = sha256(path)
    with open_csv(path) as handle:
        result = evaluate_csv(handle, population, model_name)
    if sha256(path) != before:
        raise ValueError("Prediction file changed while being read")
    check_inputs(population)
    result["input_sha256"]["predictions"] = before
    result["prediction_file"] = str(path.resolve())
    return result


def build_external_diagnostics(payload: dict, pooled_path: Path, pooled_sha256: str) -> dict:
    """Reuse the saved-score analysis; tied rankings do not invalidate scores."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            result = build_diagnostics(payload, pooled_path, pooled_sha256)
    except ValueError as error:
        if not str(error).startswith("undefined Spearman correlation:"):
            raise
        return {
            "status": "unavailable", "reason": str(error),
            "scores_remain_valid": True, "input_sha256": pooled_sha256,
            "definition": "A rank correlation is undefined when either ranking is constant.",
        }
    result["status"] = "completed"
    provenance = result["provenance"]
    provenance.pop("pooled_evaluation_script_path", None)
    provenance.pop("pooled_evaluation_script_sha256_at_input_creation", None)
    provenance["score_producer_script_path"] = str(Path(__file__).resolve())
    provenance["score_producer_script_sha256"] = payload["protocol"]["script_sha256"]
    return result


def deny_frozen_destination(path: Path, root: Path, directory: bool) -> None:
    """Prevent generated files from entering or enclosing any frozen data area."""
    resolved, root = path.resolve(), root.resolve()
    protected = [root / "data", root / "output"]
    if directory and resolved == root:
        raise ValueError("Score output must use a separate directory")
    for area in protected:
        if resolved.is_relative_to(area) or (directory and area.is_relative_to(resolved)):
            raise ValueError(f"Output destination overlaps a frozen directory: {area}")


def write_template(population: FrozenPopulation, output: Path, screens: list[str] | None = None,
                   protocols: list[str] | None = None, seeds: list[int] | None = None) -> int:
    """Write known memberships and a blank prediction column, optionally scoped."""
    selected_screens = set(screens or [task.name for task in population.tasks])
    selected_protocols = set(protocols or PROTOCOLS)
    selected_seeds = set(seeds or SEEDS)
    if not selected_screens.issubset({task.name for task in population.tasks}):
        raise ValueError("Template requested an unknown screen")
    if not selected_protocols.issubset(PROTOCOLS) or not selected_seeds.issubset(SEEDS):
        raise ValueError("Template requested an unknown protocol or seed")
    deny_frozen_destination(output, population.root, directory=False)
    if output.exists():
        raise ValueError("Template output already exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_csv(output, "wt") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for task in population.tasks:
            if task.name not in selected_screens:
                continue
            for kind in PROTOCOLS:
                if kind not in selected_protocols:
                    continue
                for seed_index, seed in enumerate(SEEDS):
                    if seed not in selected_seeds:
                        continue
                    for index, uid in enumerate(task.row_uids):
                        writer.writerow([task.name, uid, kind, seed, int(task.assignment[kind][seed_index][index]), ""])
                        count += 1
    check_inputs(population)
    return count


def write_scores(payload: dict, population: FrozenPopulation, output: Path) -> dict:
    """Write new pooled, summary and diagnostic files only into a fresh directory."""
    deny_frozen_destination(output, population.root, directory=True)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Score output directory must be new or empty")
    pooled_path = output / "pooled_model_results.json"
    content = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    diagnostics = build_external_diagnostics(payload, pooled_path, hashlib.sha256(content).hexdigest())
    check_inputs(population)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Score output directory changed before writing")
    pooled_path.write_bytes(content)
    rows = [dict(csv_row(source), yield_sd_percentage_points=source["yield_sd_percentage_points"])
            for source in payload["sources"]]
    summary_path = output / "pooled_model_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostic_path = output / "saved_score_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"output_directory": str(output.resolve()), "prediction_rows": payload["prediction_rows"],
            "model_name": payload["model_name"], "diagnostics_status": diagnostics["status"],
            "training_validation": TRAINING_LIMIT}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="Write a blank CSV/CSV.gz prediction template")
    template.add_argument("--output", type=Path, required=True, help="New CSV path; .gz selects gzip compression")
    template.add_argument("--screen", action="append", help="Select a screen by exact name; repeat to select several")
    template.add_argument("--protocol", choices=PROTOCOLS, action="append", help="Select row or group; repeat as needed")
    template.add_argument("--seed", type=int, choices=SEEDS, action="append", help="Select a frozen seed; repeat as needed")
    score = commands.add_parser("score", help="Score complete coverage of all 18 screens, protocols and seeds")
    score.add_argument("--predictions", type=Path, required=True, help="Complete six-column CSV or CSV.gz")
    score.add_argument("--model-name", required=True, help="Label the external model in new result files")
    score.add_argument("--output-dir", type=Path, required=True, help="New/empty separate directory outside frozen data")
    args = parser.parse_args()
    try:
        if args.command == "score":
            deny_frozen_destination(args.output_dir, ROOT, directory=True)
            if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
                raise ValueError("Score output directory must be new or empty")
            if not args.model_name.strip():
                raise ValueError("Model name must not be blank")
        else:
            deny_frozen_destination(args.output, ROOT, directory=False)
            if args.output.exists():
                raise ValueError("Template output already exists; choose a new path")
        population = load_population()
        if args.command == "template":
            count = write_template(population, args.output, args.screen, args.protocol, args.seed)
            result = {"template": str(args.output.resolve()), "rows": count,
                      "prediction_units": "yield fraction; leave no blanks in a completed submission",
                      "complete_score_rows_required": EXPECTED_PREDICTIONS}
        else:
            payload = score_file(args.predictions, population, args.model_name)
            result = write_scores(payload, population, args.output_dir)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
