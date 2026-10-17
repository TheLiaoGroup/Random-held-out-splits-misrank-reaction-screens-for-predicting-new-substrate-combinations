"""Fit a fixed multilayer perceptron on the frozen tasks and write a scoring-interface prediction file.

The network uses exactly the random forest's inputs: three 1,024-bit Morgan fingerprints
(substrate 1, substrate 2, product) and the one-hot condition vocabulary, built by the same
feature builder, with the same frozen row-level and grouped folds for all five split seeds.
One fixed specification is used for every screen and fold, without tuning:
hidden layers (256, 128), ReLU, Adam, L2 penalty 1e-4, batch size 512, learning rate 1e-3,
at most 60 epochs with early stopping on 10% of each training fold, random_state 42.

Output: a CSV.gz with the scoring-interface columns screen,row_uid,protocol,seed,fold,y_pred,
written to a directory outside the package. Score it with score_predictions.py.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

MLP_CONFIG = dict(hidden_layer_sizes=(256, 128), activation="relu", solver="adam", alpha=1e-4,
                  batch_size=512, learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=5, random_state=42)

_FEATURES = None
_TARGET = None


def _init(features, target):
    global _FEATURES, _TARGET
    _FEATURES, _TARGET = features, target


def _fit(task):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    seed, protocol, fold, train, test = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MLPRegressor(**MLP_CONFIG)
        model.fit(_FEATURES[train], _TARGET[train])
    return seed, protocol, fold, test, model.predict(_FEATURES[test]), int(model.n_iter_)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen", action="append", help="optional screen selector; repeat for several")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    package, output = args.package_dir.resolve(), args.output_dir.resolve()
    if output.is_relative_to(package):
        raise ValueError("Write predictions outside the frozen package")
    import sys
    sys.path.insert(0, str(package))
    from p4_rf_oof import SEEDS, load_inputs, partitions
    population, screens, scorer, builder = load_inputs(package, args.screen)
    output.mkdir(parents=True, exist_ok=True)
    for screen in screens:
        target_path = output / f"mlp_predictions_{screen.slug}.csv.gz"
        if target_path.exists():
            print(f"skip {screen.name}: predictions exist", flush=True)
            continue
        started = time.perf_counter()
        features = sp.csr_matrix(builder.build_features(screen.frame, screen.vocabulary).astype(np.float32))
        target = screen.frame.yield_fraction.to_numpy(dtype=float)
        parts = partitions(screen, list(SEEDS))
        tasks = [(p.seed, p.protocol, p.fold, p.train, p.test) for p in parts]
        frames, epochs = [], []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(features, target)) as pool:
            for seed, protocol, fold, test, prediction, n_iter in pool.map(_fit, tasks):
                frames.append(pd.DataFrame({"screen": screen.name, "row_uid": screen.frame.row_uid.iloc[test].to_numpy(),
                                            "protocol": protocol, "seed": seed, "fold": fold, "y_pred": prediction}))
                epochs.append(n_iter)
        table = pd.concat(frames, ignore_index=True)
        if len(table) != 10 * len(screen.frame) or not np.isfinite(table.y_pred).all():
            raise ValueError(f"{screen.name}: incomplete or nonfinite predictions")
        table.to_csv(target_path, index=False, compression={"method": "gzip", "mtime": 0})
        print(f"{screen.name}: {len(table)} predictions, epochs {min(epochs)}-{max(epochs)}, "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    pieces = sorted(p for p in output.glob("mlp_predictions_*.csv.gz") if p.name != "mlp_predictions_all.csv.gz")
    if len(pieces) == len(population.tasks):
        combined = pd.concat([pd.read_csv(p) for p in pieces], ignore_index=True)
        combined.to_csv(output / "mlp_predictions_all.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        print(f"combined {len(combined)} predictions", flush=True)


if __name__ == "__main__":
    main()
