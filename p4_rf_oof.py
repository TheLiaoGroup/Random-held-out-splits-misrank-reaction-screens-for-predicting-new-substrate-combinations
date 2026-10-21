"""Independent, resumable fixed-RF OOF predictions on the frozen P4 tasks.

The existing feature builder, populations, row folds and grouped fold registry
are reused unchanged. Results belong in a separate internal output directory.
No search, tuning, target clipping or replacement of old scores is performed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

SEEDS = (42, 20261007, 20261008, 20261009, 20261010)
PROTOCOLS = ("group", "row")
N_SPLITS = 5
SCHEMA_VERSION = 1
MODEL_NAME = "RandomForestRegressor_fixed_Suzuki_parameters"
RF_CONFIG = {
    "n_estimators": 100, "max_depth": None, "max_features": 1.0,
    "bootstrap": True, "min_samples_leaf": 1, "min_samples_split": 2,
    "criterion": "squared_error", "random_state": 42,
}
OOF_COLUMNS = ("screen", "row_uid", "protocol", "seed", "fold", "y_true", "y_pred",
               "n_unseen", "training_mean")
REQUIRED_FEATURE_COLUMNS = ("row_uid", "yield_fraction", "plate_group", "pair_key",
                            "component_group", "sub_1_smiles", "sub_2_smiles",
                            "product_smiles", "condition_id")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def signature(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def uid_hash(values) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_csv(path: Path, frames) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            import io
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                header = True
                for frame in frames:
                    frame.to_csv(handle, index=False, header=header, float_format="%.17g")
                    header = False
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)
    return {"path": path.name, "sha256": sha256(path)}


def safe_artifact(directory: Path, artifact: dict) -> Path:
    relative = str(artifact["path"])
    windows = PureWindowsPath(relative)
    parts = windows.parts
    if windows.is_absolute() or ".." in parts:
        raise ValueError("prediction output path escapes its source directory")
    path = directory.joinpath(*parts).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing prediction output: {relative}")
    if sha256(path) != artifact["sha256"]:
        raise ValueError(f"prediction output hash mismatch: {relative}")
    return path


def validate_output_dir(package: Path, output: Path) -> None:
    package, output = package.resolve(), output.resolve()
    if output.is_relative_to(package) or package.is_relative_to(output):
        raise ValueError("RF output must be separate from the public package and its ancestors")
    for protected in (package.parent / "submission", package.parent / "internal" / "source-construction",
                      package.parent / "internal" / "experimental-reference"):
        if output.is_relative_to(protected.resolve()):
            raise ValueError("RF output must not overwrite frozen inputs or submission material")


@dataclass
class ScreenInput:
    metadata: dict
    frame: pd.DataFrame
    vocabulary: list[str]
    assignments: dict[str, dict[int, np.ndarray]]
    provenance: dict

    @property
    def name(self) -> str:
        return self.metadata["source_name"]

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name)


@dataclass
class Partition:
    seed: int
    protocol: str
    fold: int
    train: np.ndarray
    test: np.ndarray


def partitions(screen: ScreenInput, seeds: list[int] | tuple[int, ...]) -> list[Partition]:
    frame = screen.frame
    missing = set(REQUIRED_FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{screen.name}: missing model columns {sorted(missing)}")
    y = frame.yield_fraction.to_numpy(dtype=np.float64)
    if (len(frame) != screen.metadata["numeric_records"] or frame.row_uid.isna().any()
            or frame.row_uid.astype(str).duplicated().any() or not np.isfinite(y).all()):
        raise ValueError(f"{screen.name}: invalid numerical population or row identity")
    if not math.isclose(float(y.mean()), screen.metadata["mean_yield_fraction"],
                        rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"{screen.name}: numerical target mean changed")
    if len(set(seeds)) != len(seeds) or not seeds or not set(seeds).issubset(SEEDS):
        raise ValueError("seeds must be unique members of the frozen five-seed protocol")
    result = []
    for seed in seeds:
        for kind in PROTOCOLS:
            assigned = np.asarray(screen.assignments[kind][seed])
            if assigned.shape != (len(frame),) or not np.isin(assigned, np.arange(N_SPLITS)).all():
                raise ValueError(f"{screen.name}: incomplete or invalid frozen assignment")
            tests = []
            for fold in range(N_SPLITS):
                test, train = np.flatnonzero(assigned == fold), np.flatnonzero(assigned != fold)
                if not len(test) or not len(train):
                    raise ValueError(f"{screen.name}: empty frozen partition")
                if kind == "group":
                    for column in ("plate_group", "pair_key", "component_group"):
                        if set(frame.iloc[train][column]) & set(frame.iloc[test][column]):
                            raise ValueError(f"{screen.name}: grouped partition leaks {column}")
                tests.append(test)
                result.append(Partition(int(seed), kind, fold, train, test))
            if not np.array_equal(np.sort(np.concatenate(tests)), np.arange(len(frame))):
                raise ValueError(f"{screen.name}: frozen test partitions lack exact coverage")
    return result


def n_unseen(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    result = np.zeros(len(test), dtype=np.int8)
    for column in ("sub_1_smiles", "sub_2_smiles"):
        values = frame[column].fillna("").astype(str)
        observed = set(values.iloc[train])
        result += (~values.iloc[test].isin(observed)).to_numpy(dtype=np.int8)
    return result


def loss_metrics(y_true: np.ndarray, prediction: np.ndarray, training_mean: float) -> dict:
    y_true, prediction = np.asarray(y_true), np.asarray(prediction)
    if (y_true.ndim != 1 or prediction.shape != y_true.shape or not len(y_true)
            or not np.isfinite(y_true).all() or not np.isfinite(prediction).all()
            or not math.isfinite(training_mean)):
        raise ValueError("truth and prediction must be equal-length finite one-dimensional arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = y_true - prediction
        sse = float(np.sum(residual ** 2))
        sae = float(np.sum(np.abs(residual)))
        sst = float(np.sum((y_true - y_true.mean()) ** 2))
        baseline = float(np.sum((y_true - training_mean) ** 2))
    if not np.isfinite([sse, sae, sst, baseline]).all():
        raise ValueError("nonfinite prediction or baseline loss")
    defined_r2 = not bool(np.all(y_true == y_true[0])) and sst > 0
    return {"sse": sse, "sae": sae, "sst": sst,
            "training_mean_baseline_sse": baseline, "training_mean": float(training_mean),
            "metrics": {"r2": float(1 - sse / sst) if defined_r2 else None,
                        "mae": sae / len(y_true), "rmse": math.sqrt(sse / len(y_true))},
            "r2_status": "defined" if defined_r2 else "constant_test_target"}


def prediction_frame(screen: ScreenInput, part: Partition, prediction: np.ndarray) -> tuple[pd.DataFrame, dict]:
    target = screen.frame.yield_fraction.to_numpy(dtype=np.float64)
    training_mean = float(target[part.train].mean())
    losses = loss_metrics(target[part.test], prediction, training_mean)
    frame = pd.DataFrame({"screen": screen.name,
                          "row_uid": screen.frame.iloc[part.test].row_uid.astype(str).to_numpy(),
                          "protocol": part.protocol, "seed": part.seed, "fold": part.fold,
                          "y_true": target[part.test], "y_pred": prediction,
                          "n_unseen": n_unseen(screen.frame, part.train, part.test),
                          "training_mean": training_mean}, columns=OOF_COLUMNS)
    record = {"seed": part.seed, "protocol": part.protocol, "fold": part.fold,
              "train_rows": len(part.train), "test_rows": len(part.test),
              "test_row_uid_sha256": uid_hash(frame.row_uid), **losses}
    for column, label in (("plate_group", "plate"), ("pair_key", "pair"),
                          ("component_group", "component")):
        train_groups = set(screen.frame.iloc[part.train][column])
        test_groups = set(screen.frame.iloc[part.test][column])
        record[f"{label}_overlap"] = len(train_groups & test_groups)
    return frame, record


def close_values(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            close_values(actual[key], value) for key, value in expected.items())
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14)
    return actual == expected


def validate_prediction_frame(screen: ScreenInput, part: Partition, frame: pd.DataFrame, record: dict) -> None:
    if tuple(frame.columns) != OOF_COLUMNS or len(frame) != len(part.test):
        raise ValueError(f"{screen.name}: prediction columns or coverage mismatch")
    if not np.isfinite(frame[["y_true", "y_pred", "training_mean", "n_unseen"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{screen.name}: nonfinite saved prediction")
    expected, evidence = prediction_frame(screen, part, frame.y_pred.to_numpy(dtype=float))
    for column in ("screen", "row_uid", "protocol", "seed", "fold", "n_unseen"):
        if frame[column].tolist() != expected[column].tolist():
            raise ValueError(f"{screen.name}: saved prediction {column} identity or order mismatch")
    for column in ("y_true", "training_mean"):
        if not np.array_equal(frame[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f"{screen.name}: saved prediction {column} mismatch")
    for key, value in evidence.items():
        if key not in record or not close_values(record[key], value):
            raise ValueError(f"{screen.name}: checkpoint {key} disagrees with saved predictions")


def write_prediction(directory: Path, part: Partition, frame: pd.DataFrame) -> dict:
    relative = f"folds/{part.protocol}_{part.seed}_{part.fold}.csv.gz"
    path = directory / relative
    artifact = atomic_csv(path, [frame])
    artifact.update(path=relative, rows=len(frame))
    return artifact


def empty_checkpoint(screen: ScreenInput, protocol: dict) -> dict:
    return {"schema_version": SCHEMA_VERSION, "status": "running", "model_name": MODEL_NAME,
            "source_name": screen.name, "reaction_family": screen.metadata["reaction_family"],
            "numeric_records": len(screen.frame),
            "mean_yield_fraction": screen.metadata["mean_yield_fraction"],
            "protocol": protocol, "protocol_signature": signature(protocol), "results": []}


def validate_checkpoint(screen: ScreenInput, directory: Path, checkpoint: dict, protocol: dict) -> dict:
    if (checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("model_name") != MODEL_NAME
            or checkpoint.get("source_name") != screen.name
            or checkpoint.get("reaction_family") != screen.metadata["reaction_family"]
            or checkpoint.get("numeric_records") != len(screen.frame)
            or checkpoint.get("mean_yield_fraction") != screen.metadata["mean_yield_fraction"]):
        raise ValueError(f"{screen.name}: checkpoint schema or source identity mismatch")
    if (checkpoint.get("protocol_signature") != signature(checkpoint.get("protocol"))
            or checkpoint.get("protocol_signature") != signature(protocol)):
        raise ValueError(f"{screen.name}: checkpoint uses another protocol; refusing to resume")
    if checkpoint.get("status") not in ("running", "completed"):
        raise ValueError(f"{screen.name}: invalid checkpoint status")
    parts = {(p.protocol, p.seed, p.fold): p for p in partitions(screen, protocol["seeds"])}
    records = checkpoint.get("results", [])
    keys = [(r["protocol"], r["seed"], r["fold"]) for r in records]
    if len(set(keys)) != len(keys) or not set(keys).issubset(parts):
        raise ValueError(f"{screen.name}: duplicate or unexpected checkpoint folds")
    if checkpoint["status"] == "completed" and set(keys) != set(parts):
        raise ValueError(f"{screen.name}: completed checkpoint has missing folds")
    for key, record in zip(keys, records):
        artifact = record.get("prediction_file")
        if not artifact:
            raise ValueError(f"{screen.name}: checkpoint is missing its prediction file")
        path = safe_artifact(directory, artifact)
        if artifact.get("rows") != len(parts[key].test):
            raise ValueError(f"{screen.name}: saved prediction row count mismatch")
        frame = pd.read_csv(path, dtype={"screen": str, "row_uid": str, "protocol": str}, float_precision="round_trip")
        validate_prediction_frame(screen, parts[key], frame, record)
    summary, pooled = summarize(screen, records, protocol["seeds"])
    if "summary" in checkpoint and not close_values(checkpoint["summary"], summary):
        raise ValueError(f"{screen.name}: checkpoint summary mismatch")
    if "pooled_by_seed" in checkpoint and not close_values(checkpoint["pooled_by_seed"], pooled):
        raise ValueError(f"{screen.name}: checkpoint pooled loss mismatch")
    if checkpoint["status"] == "completed":
        artifact = checkpoint.get("oof_file")
        if not artifact:
            raise ValueError(f"{screen.name}: completed checkpoint has no collected OOF file")
        path = safe_artifact(directory, artifact)
        expected_rows = len(screen.frame) * len(PROTOCOLS) * len(protocol["seeds"])
        if artifact.get("rows") != expected_rows:
            raise ValueError(f"{screen.name}: collected OOF coverage mismatch")
        # Recheck the final export against the individual folds, including order.
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != OOF_COLUMNS:
                raise ValueError(f"{screen.name}: collected OOF header mismatch")
            count = 0
            for part in parts.values():
                chunk = [next(reader, None) for _ in part.test]
                if any(row is None for row in chunk):
                    raise ValueError(f"{screen.name}: truncated collected OOF file")
                frame = pd.DataFrame(chunk)
                for column in ("seed", "fold", "n_unseen"):
                    frame[column] = [int(value) for value in frame[column]]
                for column in ("y_true", "y_pred", "training_mean"):
                    frame[column] = [float(value) for value in frame[column]]
                key = (part.protocol, part.seed, part.fold)
                validate_prediction_frame(screen, part, frame, records[keys.index(key)])
                count += len(frame)
            if next(reader, None) is not None or count != expected_rows:
                raise ValueError(f"{screen.name}: extra collected OOF rows")
    return checkpoint


def distribution(values: list) -> dict:
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "sd": None, "min": None, "max": None, "range": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite summary metric")
    return {"count": len(values), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()), "range": float(np.ptp(values))}


def summarize(screen: ScreenInput, records: list[dict], seeds: list[int]) -> tuple[dict, dict]:
    target = screen.frame.yield_fraction.to_numpy(dtype=np.float64)
    sst = float(np.sum((target - target.mean()) ** 2))
    defined_r2 = not bool(np.all(target == target[0])) and sst > 0
    pooled, summary = {}, {}
    for kind in PROTOCOLS:
        pooled[kind] = []
        for seed in seeds:
            selected = [r for r in records if r["protocol"] == kind and r["seed"] == seed]
            if len(selected) != N_SPLITS:
                continue
            sse, sae = sum(r["sse"] for r in selected), sum(r["sae"] for r in selected)
            baseline = sum(r["training_mean_baseline_sse"] for r in selected)
            pooled[kind].append({"seed": seed, "rows": len(target), "sse": sse, "sae": sae, "sst": sst,
                                 "sum_y": float(target.sum()), "sum_y_squared": float(np.sum(target ** 2)),
                                 "r2": 1 - sse / sst if defined_r2 else None,
                                 "mae": sae / len(target), "rmse": math.sqrt(sse / len(target)),
                                 "training_mean_baseline_sse": baseline,
                                 "training_mean_baseline_r2": 1 - baseline / sst if defined_r2 else None,
                                 "skill_vs_training_mean": 1 - sse / baseline if baseline > 0 else None,
                                 "undefined_fold_r2_count": sum(r["metrics"]["r2"] is None for r in selected)})
        summary[kind] = {metric: distribution([r[metric] for r in pooled[kind]]) for metric in
                         ("r2", "mae", "rmse", "training_mean_baseline_r2", "skill_vs_training_mean")}
        summary[kind]["complete_seeds"] = len(pooled[kind])
        summary[kind]["auxiliary_fold_r2"] = distribution(
            [r["metrics"]["r2"] for r in records if r["protocol"] == kind])
    lookup = {kind: {r["seed"]: r for r in pooled[kind]} for kind in PROTOCOLS}
    differences = [{"seed": seed, "row_minus_group":
                    lookup["row"][seed]["r2"] - lookup["group"][seed]["r2"]
                    if lookup["row"][seed]["r2"] is not None and lookup["group"][seed]["r2"] is not None else None}
                   for seed in seeds if all(seed in lookup[kind] for kind in PROTOCOLS)]
    summary["delta_r2"] = {**distribution([r["row_minus_group"] for r in differences]), "by_seed": differences}
    return summary, pooled


def get_model(threads: int):
    from sklearn.ensemble import RandomForestRegressor
    if threads < 1:
        raise ValueError("threads must be positive")
    return RandomForestRegressor(**RF_CONFIG, n_jobs=threads)


def run_fold_tasks(parts, worker, commit, *, workers: int) -> None:
    """Fit concurrently; invoke every artifact commit on the calling thread."""
    if workers < 1:
        raise ValueError("fold workers must be positive")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p4_rf_fold") as executor:
        futures = {executor.submit(worker, part): part for part in parts}
        try:
            for future in as_completed(futures):
                commit(futures[future], future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def fit_fold(screen: ScreenInput, features: np.ndarray, part: Partition, *, threads: int):
    """Return fitted predictions and evidence without writing any artifacts."""
    started = time.perf_counter()
    training_x = np.asfortranarray(features[part.train], dtype=np.float32)
    model = get_model(threads)
    model.fit(training_x, screen.frame.yield_fraction.to_numpy(dtype=float)[part.train])
    prediction = model.predict(features[part.test])
    frame, record = prediction_frame(screen, part, prediction)
    record.update(elapsed_seconds=time.perf_counter() - started,
                  training_array_order="F", training_array_dtype="float32")
    return frame, record


def run_screen(screen: ScreenInput, directory: Path, protocol: dict, *, features: np.ndarray | None = None,
               feature_builder=None, max_new_folds: int | None = None, collect_only: bool = False) -> dict:
    if max_new_folds is not None and max_new_folds < 0:
        raise ValueError("max_new_folds must be nonnegative")
    parts = partitions(screen, protocol["seeds"])
    path = directory / "model_results.json"
    frozen = directory / "frozen_protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != protocol:
            raise ValueError(f"{screen.name}: frozen RF protocol differs")
    elif path.exists():
        raise ValueError(f"{screen.name}: missing frozen RF protocol")
    checkpoint = empty_checkpoint(screen, protocol)
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        validate_checkpoint(screen, directory, checkpoint, protocol)
    elif directory.exists() and any(item.name != "frozen_protocol.json" for item in directory.iterdir()):
        raise ValueError(f"{screen.name}: output directory contains files without a checkpoint")
    if collect_only or checkpoint["status"] == "completed":
        return checkpoint
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(frozen, protocol)
    atomic_json(path, checkpoint)
    completed = {(r["protocol"], r["seed"], r["fold"]) for r in checkpoint["results"]}
    if max_new_folds == 0:
        return checkpoint
    if features is None:
        if feature_builder is None:
            raise ValueError("a frozen feature builder is required for new fits")
        features = feature_builder(screen.frame, screen.vocabulary)
    if (features.ndim != 2 or len(features) != len(screen.frame) or features.dtype != np.float32
            or not np.isfinite(features).all()):
        raise ValueError(f"{screen.name}: invalid feature shape or nonfinite feature values")
    pending = [part for part in parts if (part.protocol, part.seed, part.fold) not in completed]
    if max_new_folds is not None:
        pending = pending[:max_new_folds]
    order = {(part.protocol, part.seed, part.fold): index for index, part in enumerate(parts)}

    def worker(part):
        return fit_fold(screen, features, part, threads=protocol["threads"])

    def commit(part, fitted):
        frame, record = fitted
        record["prediction_file"] = write_prediction(directory, part, frame)
        checkpoint["results"].append(record)
        checkpoint["results"].sort(key=lambda row: order[(row["protocol"], row["seed"], row["fold"])])
        checkpoint["summary"], checkpoint["pooled_by_seed"] = summarize(screen, checkpoint["results"], protocol["seeds"])
        atomic_json(path, checkpoint)
        print(json.dumps({"source": screen.name, "completed_folds": len(checkpoint["results"]),
                          "protocol": part.protocol, "seed": part.seed, "fold": part.fold,
                          "seconds": round(record["elapsed_seconds"], 2)}), flush=True)
    run_fold_tasks(pending, worker, commit, workers=protocol.get("fold_workers", 2))
    if len(checkpoint["results"]) == len(parts):
        records = {(r["protocol"], r["seed"], r["fold"]): r for r in checkpoint["results"]}
        def frames():
            for part in parts:
                record = records[(part.protocol, part.seed, part.fold)]
                yield pd.read_csv(safe_artifact(directory, record["prediction_file"]),
                                  dtype={"screen": str, "row_uid": str, "protocol": str}, float_precision="round_trip")
        checkpoint["oof_file"] = atomic_csv(directory / "oof_predictions.csv.gz", frames())
        checkpoint["oof_file"]["rows"] = len(screen.frame) * len(PROTOCOLS) * len(protocol["seeds"])
        checkpoint["status"] = "completed"
        atomic_json(path, checkpoint)
        validate_checkpoint(screen, directory, checkpoint, protocol)
    return checkpoint


def package_modules(package: Path):
    package = package.resolve()
    sys.path.insert(0, str(package))
    scorer = importlib.import_module("score_predictions")
    builder = importlib.import_module("p4_u0_model_analysis_20261007")
    for module in (scorer, builder):
        if Path(module.__file__).resolve().parent != package:
            raise ValueError("imported frozen adapter or feature builder belongs to another package")
    return scorer, builder


def load_inputs(package: Path, names: list[str] | None):
    scorer, builder = package_modules(package)
    population = scorer.load_population(package)
    known = {task.name for task in population.tasks}
    if names and (len(set(names)) != len(names) or not set(names).issubset(known)):
        raise ValueError("screen selectors must be unique frozen source names")
    selected = [task for task in population.tasks if not names or task.name in names]
    records = scorer.read_records(package)
    vocabularies = {source["source_name"]: source["condition_vocabulary"] for source in population.reference["sources"]}
    screens = []
    for task in selected:
        model = records.loc[records.screen.eq(task.name)].reset_index(drop=True)
        provenance = {name: {"path": path, "sha256": population.input_hashes[path]}
                      for name, path in (("records", scorer.RECORDS), ("grouped_folds", scorer.FOLDS),
                                         ("screens", scorer.SCREENS))}
        if tuple(model.row_uid.astype(str)) != task.row_uids:
            raise ValueError(f"{task.name}: feature row order differs from frozen scoring population")
        model["yield_fraction"] = pd.to_numeric(model.yield_fraction, errors="coerce")
        if not np.array_equal(model.yield_fraction.to_numpy(dtype=float), task.y):
            raise ValueError(f"{task.name}: feature targets differ from frozen scoring population")
        for column in ("plate_group", "pair_key", "component_group"):
            model[column] = model[column].fillna("").astype(str)
        model["condition_id"] = model.condition_id.astype(str)
        vocabulary = [str(value) for value in vocabularies[task.name]]
        if (vocabulary != sorted(vocabulary, key=builder.natural_condition) or len(set(vocabulary)) != len(vocabulary)
                or not set(model.condition_id).issubset(vocabulary)):
            raise ValueError(f"{task.name}: released condition vocabulary is invalid")
        assignments = {kind: {seed: task.assignment[kind][i] for i, seed in enumerate(SEEDS)} for kind in PROTOCOLS}
        screen = ScreenInput(task.metadata, model, vocabulary, assignments, provenance)
        partitions(screen, list(SEEDS))
        screens.append(screen)
    if len({s.slug for s in screens}) != len(screens):
        raise ValueError("screen output slugs collide")
    scorer.check_inputs(population)
    return population, screens, scorer, builder


def package_versions() -> dict:
    import sklearn
    import scipy
    import xgboost
    from rdkit import rdBase
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "xgboost": xgboost.__version__, "rdkit": rdBase.rdkitVersion}


def expected_protocol(screen: ScreenInput, config: dict) -> dict:
    return {"run_configuration_signature": signature(config), "seeds": config["seeds"],
            "protocols": list(PROTOCOLS), "n_splits": N_SPLITS, "threads": config["threads"],
            "fold_workers": config["fold_workers"], "execution": config["execution"],
            "model_parameters": config["model_parameters"],
            "feature_builder_sha256": config["implementation_sha256"]["feature_builder"],
            "script_sha256": config["implementation_sha256"]["rf_runner"], "versions": config["versions"],
            "row_uid_sha256": uid_hash(screen.frame.row_uid), "condition_vocabulary": screen.vocabulary,
            "provenance": screen.provenance,
            "assignment_sha256": {kind: {str(seed): signature(screen.assignments[kind][seed].tolist())
                                         for seed in config["seeds"]} for kind in PROTOCOLS},
            "target": "finite numeric yield fractions including explicit zeros; no clipping",
            "features": "unchanged Morgan radius 2, 1024 bits for both substrate roles and product; frozen all-record condition one-hot",
            "model_selection": "none; fixed Suzuki RF parameters applied to every screen",
            "n_unseen": "exact raw SMILES membership separately by substrate role in numeric training rows"}


def collect_manifest(output: Path, config: dict, screens: list[ScreenInput], protocols: dict) -> dict:
    sources = []
    files = {}
    for screen in screens:
        directory = output / "sources" / screen.slug
        checkpoint = run_screen(screen, directory, protocols[screen.name], collect_only=True)
        if (directory / "model_results.json").exists():
            relative = (directory / "model_results.json").relative_to(output).as_posix()
            files[relative] = sha256(directory / "model_results.json")
        if checkpoint["status"] == "completed":
            relative = (directory / checkpoint["oof_file"]["path"]).relative_to(output).as_posix()
            files[relative] = checkpoint["oof_file"]["sha256"]
        source = {**checkpoint, "source_slug": screen.slug}
        if checkpoint["status"] == "completed":
            source["oof_predictions"] = {**checkpoint["oof_file"], "path": relative}
        sources.append(source)
    completed = sum(source["status"] == "completed" for source in sources)
    deltas = [s["summary"]["delta_r2"]["mean"] for s in sources if s["status"] == "completed"]
    valid_deltas = [value for value in deltas if value is not None]
    result = {"schema_version": SCHEMA_VERSION, "status": "completed" if completed == len(sources) else "running",
              "model_name": MODEL_NAME, "configuration": config, "configuration_signature": signature(config),
              "expected_source_count": len(sources), "complete_source_count": completed,
              "expected_folds": len(sources) * len(config["seeds"]) * len(PROTOCOLS) * N_SPLITS,
              "complete_folds": sum(len(source["results"]) for source in sources),
              "complete_prediction_rows": sum(source["oof_file"]["rows"] for source in sources if source["status"] == "completed"),
              "output_file_sha256": files, "sources": sources,
              "cross_source_summary": {"weighting": "equal completed source weights; seeds are descriptive overlapping estimates",
                                       "completed_sources": completed, "valid_delta_sources": len(valid_deltas),
                                       "positive_delta_sources": sum(value > 0 for value in valid_deltas),
                                       "median_delta_r2": float(np.median(valid_deltas)) if valid_deltas else None}}
    atomic_json(output / "run_manifest.json", result)
    return result


def run(args: argparse.Namespace) -> dict:
    package, output = args.package_dir.resolve(), args.output_dir.resolve()
    validate_output_dir(package, output)
    if args.threads < 1 or args.fold_workers < 1 or (args.max_new_folds is not None and args.max_new_folds < 0):
        raise ValueError("threads and fold-workers must be positive; max-new-folds must be nonnegative")
    seeds = args.seed or list(SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        raise ValueError("seed selectors must be unique frozen seeds")
    population, screens, scorer, builder = load_inputs(package, args.screen)
    config = {"schema_version": SCHEMA_VERSION, "model_name": MODEL_NAME, "model_parameters": get_model(args.threads).get_params(),
              "threads": args.threads, "fold_workers": args.fold_workers,
              "execution": {"training_array_order": "F", "training_array_dtype": "float32", "test_array_order": "C",
                            "fold_workers": args.fold_workers, "tree_jobs_per_fold": args.threads,
                            "source_workers": 1, "checkpoint_writer": "parent_only",
                            "scheduler": "ThreadPoolExecutor over folds within one source",
                            "feature_matrix": "one unchanged C-order float32 matrix shared by fold tasks"},
              "seeds": seeds, "protocols": list(PROTOCOLS), "n_splits": N_SPLITS,
              "screen_names": [s.name for s in screens], "versions": package_versions(),
              "input_sha256": population.input_hashes,
              "implementation_sha256": {"rf_runner": sha256(Path(__file__)), "feature_builder": sha256(Path(builder.__file__)),
                                         "population_adapter": sha256(Path(scorer.__file__))}}
    protocols = {screen.name: expected_protocol(screen, config) for screen in screens}
    for screen in screens:
        partitions(screen, seeds)
    frozen = output / "frozen_rf_configuration.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != config:
            raise ValueError("output contains a different frozen RF configuration; refusing to overwrite")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output directory contains files without a frozen RF configuration")
    if args.dry_run:
        # Existing files are still audited by dry-run; a fresh dry-run writes nothing.
        for screen in screens:
            run_screen(screen, output / "sources" / screen.slug, protocols[screen.name], collect_only=True)
        return {"status": "dry_run", "configuration": config, "configuration_signature": signature(config),
                "expected_folds": len(screens) * len(seeds) * len(PROTOCOLS) * N_SPLITS,
                "numeric_records": sum(len(s.frame) for s in screens), "screen_names": [s.name for s in screens]}
    if args.collect_only and not frozen.exists():
        raise ValueError("collect-only requires an existing frozen RF configuration")
    if not args.collect_only:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(frozen, config)
        collect_manifest(output, config, screens, protocols)
        remaining = args.max_new_folds
        for screen in screens:
            directory = output / "sources" / screen.slug
            before = run_screen(screen, directory, protocols[screen.name], collect_only=True)
            completed_before = len(before["results"])
            result = run_screen(screen, directory, protocols[screen.name], feature_builder=builder.build_features,
                                max_new_folds=remaining)
            if remaining is not None:
                remaining -= len(result["results"]) - completed_before
            collect_manifest(output, config, screens, protocols)
            if remaining is not None and remaining <= 0:
                break
    scorer.check_inputs(population)
    return collect_manifest(output, config, screens, protocols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen", action="append", help="Repeat to select a screen; default all")
    parser.add_argument("--seed", action="append", type=int, help="Repeat to select frozen seeds; default all five")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fold-workers", type=int, default=2)
    parser.add_argument("--max-new-folds", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs, protocols and checkpoints without fitting or writing")
    mode.add_argument("--collect-only", action="store_true", help="Validate saved predictions and collect progress without fitting")
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, KeyError, ImportError) as error:
        parser.exit(2, f"RF OOF validation failed: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "expected_folds")}
                     | {key: result[key] for key in ("complete_folds", "complete_source_count", "numeric_records", "screen_names")
                        if key in result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
