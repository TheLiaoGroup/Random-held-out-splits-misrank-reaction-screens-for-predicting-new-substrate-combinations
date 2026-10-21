"""Original screen-specific models on the released tasks (sensitivity analysis to the common random forest).

Each screen keeps its original estimator and parameters (output/p4_original_models/model_parameters.json):
random forests for Suzuki and Suzuki Bpin, XGBoost regressors otherwise. The Buchwald-Hartwig CMC array, which
had no original model, uses the Buchwald-Hartwig specification of the same reaction family. Data, features,
folds, checkpoints and validation are those of p4_rf_oof.py; only the estimator differs.

    python p4_original_models.py fit   --package-dir . --output-dir ../original-models [--screen NAME] [--threads 12]
    python p4_original_models.py score --package-dir . --output-dir ../original-models

`score` pools the saved out-of-fold predictions of every screen with score_predictions.py and writes
original_pooled_model_results.json, the input of p4_rf_oof_diagnostics.py --original-results.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import pandas as pd

FALLBACK = {"Buchwald-Hartwig CMC": "Buchwald-Hartwig"}
FIELDS = ("screen", "row_uid", "protocol", "seed", "fold", "y_pred")


def specifications(package: Path) -> dict[str, dict]:
    payload = json.loads((package / "output/p4_original_models/model_parameters.json").read_text("utf-8"))
    return {entry["screen"]: entry for entry in payload["models"]}


class Recorded:
    """An estimator whose recorded parameters are JSON-safe (XGBoost's default missing value is NaN)."""

    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def get_params(self, deep: bool = False) -> dict:
        return {key: "nan" if isinstance(value, float) and math.isnan(value) else value
                for key, value in self.model.get_params(deep=False).items()}


def estimator(entry: dict, threads: int):
    parameters = {key: float("nan") if value == "nan" else value for key, value in entry["parameters"].items()}
    if entry["estimator"] == "XGBRegressor":
        from xgboost import XGBRegressor
        parameters = {key: value for key, value in parameters.items() if value is not None}  # None records "unset"
        parameters["n_jobs"] = threads
        return Recorded(XGBRegressor(**parameters))
    if entry["estimator"] == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        parameters["n_jobs"] = threads
        return Recorded(RandomForestRegressor(**parameters))
    raise ValueError(f"Unknown original estimator: {entry['estimator']}")


def fit(args) -> None:
    package = args.package_dir.resolve()
    sys.path.insert(0, str(package))
    import p4_rf_oof
    import score_predictions
    specs = specifications(package)
    names = args.screen or [s["source_name"] for s in score_predictions.load_population(package).reference["sources"]]
    for name in names:
        entry = specs.get(name) or specs[FALLBACK[name]]
        p4_rf_oof.get_model = lambda threads, entry=entry: estimator(entry, threads)
        p4_rf_oof.MODEL_NAME = f"original screen-specific {entry['estimator']} ({entry['screen']} specification)"
        run_args = argparse.Namespace(package_dir=package, output_dir=args.output_dir / p4_rf_oof_slug(name),
                                      screen=[name], seed=None, threads=args.threads, fold_workers=1,
                                      max_new_folds=None, dry_run=False, collect_only=False)
        print(json.dumps(p4_rf_oof.run(run_args)["cross_source_summary"]), flush=True)


def p4_rf_oof_slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def score(args) -> None:
    package, output = args.package_dir.resolve(), args.output_dir.resolve()
    sys.path.insert(0, str(package))
    import score_predictions
    population = score_predictions.load_population(package)
    combined = output / "original_predictions_all.csv.gz"
    with gzip.open(combined, "wt", newline="", encoding="utf-8") as handle:
        handle.write(",".join(FIELDS) + "\n")
        for task in population.tasks:
            manifest = json.loads((output / p4_rf_oof_slug(task.name) / "run_manifest.json").read_text("utf-8"))
            if manifest["status"] != "completed":
                raise ValueError(f"{task.name}: original-model fits are incomplete")
            source = manifest["sources"][0]
            frame = pd.read_csv(output / p4_rf_oof_slug(task.name) / source["oof_predictions"]["path"],
                                dtype={"screen": str, "row_uid": str, "protocol": str})
            frame.loc[:, list(FIELDS)].to_csv(handle, header=False, index=False, float_format="%.17g")
    payload = score_predictions.score_file(combined, population, "original screen-specific models")
    (output / "original_pooled_model_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                                                                encoding="utf-8")
    print(f"scored {payload['prediction_rows']:,} predictions -> {output / 'original_pooled_model_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fit", "score"))
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen", action="append")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    (fit if args.command == "fit" else score)(args)


if __name__ == "__main__":
    main()
