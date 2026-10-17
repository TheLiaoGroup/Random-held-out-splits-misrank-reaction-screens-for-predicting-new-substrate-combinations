"""Common model evaluation for frozen P4 data views.

The data-freeze script writes one view per compressed CSV.  This module keeps
the model and feature definitions fixed across those views and writes one
row-to-fold registry derived from a declared base view. Pair identities can occur on multiple
plates, so the split group is a connected component of the plate--pair graph;
this prevents either a plate or a substrate pair from crossing train/test.

The default run is the preregistered five seeds by five folds.  ``--dry-run``
checks inputs and writes the fold registry without fitting models; ``--smoke``
uses one seed and two folds for a quick implementation check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from rdkit import Chem, rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "output" / "p4_u0_20261007"
DEFAULT_OUTPUT = ROOT / "p4_u0_model_analysis_20261007.json"
VIEW_NAMES = ["U0", "R0", "Rcorr", "Rcorr_pending", "Bcorr", "Bbound_retain", "Bbound_exclude"]
REACTIONS = ["Acylation", "Suzuki", "Buchwald-Hartwig"]
SEEDS = [42, 20261007, 20261008, 20261009, 20261010]
N_SPLITS = 5

REQUIRED_COLUMNS = {
    "row_uid",
    "reaction",
    "plate_group",
    "pair_key",
    "condition_id",
    "sub_1_smiles",
    "sub_2_smiles",
    "product_smiles",
    "yield_fraction",
    "yield_state",
    "include_numeric",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_condition(value: object) -> tuple[int, str]:
    text = str(value)
    match = re.fullmatch(r"C(\d+)", text)
    return (int(match.group(1)), "") if match else (10**9, text)


def load_view(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}")
    if frame["row_uid"].duplicated().any():
        raise ValueError(f"{name}: row_uid is not unique")
    frame = frame.copy()
    frame["reaction"] = frame["reaction"].astype(str)
    frame["plate_group"] = frame["plate_group"].fillna("").astype(str)
    frame["pair_key"] = frame["pair_key"].fillna("").astype(str)
    frame["condition_id"] = frame["condition_id"].astype(str)
    frame["yield_fraction"] = pd.to_numeric(frame["yield_fraction"], errors="coerce")
    frame["include_numeric"] = frame["include_numeric"].astype(bool)
    finite = np.isfinite(frame["yield_fraction"].to_numpy(dtype=float))
    frame["model_eligible"] = frame["include_numeric"].to_numpy() & finite
    return frame


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            if root_left < root_right:
                self.parent[root_right] = root_left
            else:
                self.parent[root_left] = root_right


def component_groups(frame: pd.DataFrame) -> pd.Series:
    """Return deterministic plate-pair connected-component labels."""
    union = UnionFind()
    for plate, pair in zip(frame["plate_group"], frame["pair_key"]):
        plate_node = "plate\x1f" + (plate or "<missing>")
        pair_node = "pair\x1f" + (pair or "<row>")
        union.union(plate_node, pair_node)
    roots = {node: union.find(node) for node in union.parent}
    root_labels = {root: f"component_{index:06d}" for index, root in enumerate(sorted(set(roots.values())))}
    labels = []
    for plate, pair in zip(frame["plate_group"], frame["pair_key"]):
        plate_node = "plate\x1f" + (plate or "<missing>")
        labels.append(root_labels[roots[union.find(plate_node)]])
    return pd.Series(labels, index=frame.index, dtype="string")


def fold_registry(
    base_view: pd.DataFrame,
    seeds: list[int],
    n_splits: int,
    reactions: list[str] = REACTIONS,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Freeze fold assignments on all rows in the declared base view."""
    registry_parts: list[pd.DataFrame] = []
    metadata: dict[str, dict] = {}
    for reaction in reactions:
        subset = base_view[base_view["reaction"] == reaction].copy()
        if subset.empty:
            raise ValueError(f"U0 has no rows for {reaction}")
        subset["component_group"] = component_groups(subset).to_numpy()
        groups = subset["component_group"].to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(f"{reaction}: {len(unique_groups)} groups < {n_splits} folds")
        reaction_meta = {
            "base_rows": int(len(subset)),
            "base_plates": int(subset["plate_group"].nunique()),
            "base_pairs": int(subset["pair_key"].nunique()),
            "component_groups": int(len(unique_groups)),
            "pairs_cross_plate": int(
                subset.groupby("pair_key")["plate_group"].nunique().gt(1).sum()
            ),
            "seeds": {},
        }
        for seed in seeds:
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            assignment = np.full(len(subset), -1, dtype=np.int16)
            seed_meta = {}
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(np.zeros(len(subset)), groups=groups)
            ):
                assignment[test_idx] = fold
                train_groups = set(groups[train_idx])
                test_groups = set(groups[test_idx])
                if train_groups & test_groups:
                    raise AssertionError(f"{reaction}: component overlap in seed {seed} fold {fold}")
                seed_meta[str(fold)] = {
                    "train_groups": int(len(train_groups)),
                    "test_groups": int(len(test_groups)),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                }
            if (assignment < 0).any():
                raise AssertionError(f"{reaction}: incomplete fold assignment")
            reaction_meta["seeds"][str(seed)] = seed_meta
            registry_parts.append(
                pd.DataFrame(
                    {
                        "row_uid": subset["row_uid"].to_numpy(),
                        "reaction": reaction,
                        "component_group": groups,
                        "seed": int(seed),
                        "fold": assignment,
                    }
                )
            )
        metadata[reaction] = reaction_meta
    return pd.concat(registry_parts, ignore_index=True), metadata


def make_model(reaction: str):
    if reaction == "Suzuki":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    return XGBRegressor(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def fingerprint_cache(values: Iterable[object]) -> dict[str, np.ndarray]:
    generator = GetMorganGenerator(radius=2, fpSize=1024)
    result: dict[str, np.ndarray] = {}
    for value in pd.Series(list(values)).fillna("").astype(str).unique():
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {value}")
        result[value] = np.asarray(generator.GetFingerprint(mol), dtype=np.float32)
    return result


def build_features(frame: pd.DataFrame, condition_vocab: list[str]) -> np.ndarray:
    caches = {
        column: fingerprint_cache(frame[column])
        for column in ("sub_1_smiles", "sub_2_smiles", "product_smiles")
    }
    molecular = [
        np.vstack([caches[column][str(value)] for value in frame[column].fillna("")])
        for column in ("sub_1_smiles", "sub_2_smiles", "product_smiles")
    ]
    condition_index = {value: index for index, value in enumerate(condition_vocab)}
    conditions = np.zeros((len(frame), len(condition_vocab)), dtype=np.float32)
    for row, value in enumerate(frame["condition_id"].astype(str)):
        index = condition_index.get(value)
        if index is not None:
            conditions[row, index] = 1.0
    return np.hstack([molecular[0], molecular[1], conditions, molecular[2]])


def metric_dict(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"r2": None, "mae": None, "rmse": None}
    values: dict[str, float | None] = {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
    }
    values["r2"] = float(r2_score(y_true, prediction)) if len(np.unique(y_true)) > 1 else None
    return values


def summarize(records: list[dict]) -> dict:
    result: dict[str, dict] = {"fold_count": len(records)}
    for metric in ("r2", "mae", "rmse"):
        values = np.asarray(
            [record["metrics"][metric] for record in records if record["metrics"][metric] is not None],
            dtype=float,
        )
        if not len(values):
            result[metric] = {"mean": None, "sd": None, "min": None, "max": None, "values": []}
        else:
            result[metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "values": [float(value) for value in values],
            }
    return result


def input_signature(frame: pd.DataFrame, reaction: str) -> str:
    """Hash the model rows so identical views can reuse fitted metrics."""
    subset = frame.loc[frame["reaction"] == reaction, ["row_uid", "yield_fraction"]].copy()
    subset = subset.sort_values("row_uid", kind="mergesort")
    return hashlib.sha256(subset.to_csv(index=False).encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> dict:
    input_dir = args.input_dir
    view_names = args.views
    active_reactions = args.reactions
    paths = {name: input_dir / f"{name}.csv.gz" for name in view_names}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
    views = {name: load_view(path, name) for name, path in paths.items()}
    base_name = args.base_view
    if base_name not in views:
        raise ValueError(f"{base_name} is required to freeze vocabularies and folds")
    base_view = views[base_name]
    vocabularies = {
            reaction: sorted(
            base_view.loc[base_view["reaction"] == reaction, "condition_id"].astype(str).unique().tolist(),
            key=natural_condition,
        )
        for reaction in active_reactions
    }
    registry, registry_meta = fold_registry(base_view, args.seeds, args.n_splits, active_reactions)
    reaction_tag = "_".join(re.sub(r"[^A-Za-z0-9]+", "", value).lower() for value in active_reactions)
    fold_path = args.output.with_name(f"p4_{base_name.lower()}_fold_registry_{reaction_tag}.csv.gz")
    registry.to_csv(fold_path, index=False, compression="gzip")

    output = {
        "status": "running",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "protocol": {
            "views": view_names,
            "reactions": active_reactions,
            "folds": "GroupKFold on plate-pair connected components",
            "n_splits": args.n_splits,
            "seeds": args.seeds,
            "base_view": base_name,
            "condition_vocab_source": f"{base_name} observed condition IDs per reaction",
            "target": "yield_fraction; finite numeric rows only, including explicit zero",
            "model_selection": "None; fixed TheLiaoGroup/10k_HTE parameters",
        },
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path), "rows": int(len(views[name]))}
            for name, path in paths.items()
        },
        "condition_vocabularies": {name: values for name, values in vocabularies.items()},
        "fold_registry": {
            "path": str(fold_path),
            "sha256": sha256(fold_path),
            "rows": int(len(registry)),
            "metadata": registry_meta,
        },
        "results": [],
        "summaries": {},
        "analysis_aliases": {},
    }

    if args.dry_run:
        output["status"] = "dry_run"
        output["view_inventory"] = {
            name: {
                "rows": int(len(frame)),
                "numeric_rows": int(frame["model_eligible"].sum()),
                "reactions": frame.groupby("reaction").size().astype(int).to_dict(),
            }
            for name, frame in views.items()
        }
        return output

    # Build each reaction's feature matrix once from the base view. Other
    # views reuse these rows where their model inputs are identical.
    feature_cache: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    for reaction in active_reactions:
        base = base_view[(base_view["reaction"] == reaction) & base_view["model_eligible"]].copy()
        base_ids = base["row_uid"].astype(str).to_numpy()
        feature_cache[reaction] = (
            {row_id: index for index, row_id in enumerate(base_ids)},
            build_features(base, vocabularies[reaction]),
        )
    registry_lookup = registry.set_index(["row_uid", "seed"])["fold"]
    completed_by_signature: dict[tuple[str, str], tuple[str, list[dict]]] = {}

    for name, frame in views.items():
        for reaction in active_reactions:
            subset = frame[(frame["reaction"] == reaction) & frame["model_eligible"]].copy()
            if subset.empty:
                continue
            signature = input_signature(frame, reaction)
            cache_key = (reaction, signature)
            if cache_key in completed_by_signature:
                source_name, source_records = completed_by_signature[cache_key]
                aliased_records = [
                    {**record, "view": name, "aliased_to": source_name}
                    for record in source_records
                ]
                output["results"].extend(aliased_records)
                output["summaries"].setdefault(name, {})[reaction] = summarize(aliased_records)
                output["analysis_aliases"][f"{name}/{reaction}"] = f"{source_name}/{reaction}"
                continue
            feature_indices, full_features = feature_cache[reaction]
            row_ids = subset["row_uid"].astype(str).to_numpy()
            missing_ids = [row_id for row_id in row_ids if row_id not in feature_indices]
            if missing_ids:
                raise ValueError(f"{name}/{reaction}: {len(missing_ids)} rows are absent from {base_name} feature cache")
            features = full_features[[feature_indices[row_id] for row_id in row_ids]]
            target = subset["yield_fraction"].to_numpy(dtype=float)
            reaction_records: list[dict] = []
            for seed in args.seeds:
                for fold in range(args.n_splits):
                    assigned = registry_lookup.reindex(
                        pd.MultiIndex.from_arrays(
                            [row_ids, np.full(len(row_ids), seed)],
                            names=["row_uid", "seed"],
                        )
                    ).to_numpy()
                    if pd.isna(assigned).any():
                        raise ValueError(f"{name}/{reaction}: incomplete fold assignment")
                    test_idx = np.flatnonzero(assigned == fold)
                    train_idx = np.flatnonzero(assigned != fold)
                    if len(train_idx) == 0 or len(test_idx) == 0:
                        continue
                    model = make_model(reaction)
                    model.fit(features[train_idx], target[train_idx])
                    prediction = model.predict(features[test_idx])
                    metrics = metric_dict(target[test_idx], prediction)
                    test_groups = set(subset.iloc[test_idx]["plate_group"].astype(str))
                    train_groups = set(subset.iloc[train_idx]["plate_group"].astype(str))
                    test_pairs = set(subset.iloc[test_idx]["pair_key"].astype(str))
                    train_pairs = set(subset.iloc[train_idx]["pair_key"].astype(str))
                    record = {
                        "view": name,
                        "reaction": reaction,
                        "seed": int(seed),
                        "fold": int(fold),
                        "algorithm": "rf" if reaction == "Suzuki" else "xgb",
                        "train_rows": int(len(train_idx)),
                        "test_rows": int(len(test_idx)),
                        "train_plates": int(len(train_groups)),
                        "test_plates": int(len(test_groups)),
                        "plate_overlap": int(len(train_groups & test_groups)),
                        "pair_overlap": int(len(train_pairs & test_pairs)),
                        "metrics": metrics,
                    }
                    if record["plate_overlap"] or record["pair_overlap"]:
                        raise AssertionError(f"{name}/{reaction}: group overlap in seed {seed} fold {fold}")
                    output["results"].append(record)
                    reaction_records.append(record)
            output["summaries"].setdefault(name, {})[reaction] = summarize(reaction_records)
            completed_by_signature[cache_key] = (name, reaction_records)
    output["status"] = "completed"
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--views", nargs="+", default=VIEW_NAMES, choices=VIEW_NAMES)
    parser.add_argument("--base-view", default="U0", choices=VIEW_NAMES)
    parser.add_argument("--reactions", nargs="+", default=REACTIONS, choices=REACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One seed and two folds; implementation check only")
    args = parser.parse_args()
    if args.smoke:
        args.views = [args.base_view]
        args.seeds = [args.seeds[0]]
        args.n_splits = 2
    result = run(args)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
