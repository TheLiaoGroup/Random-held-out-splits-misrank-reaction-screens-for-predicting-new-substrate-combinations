"""A pretrained reaction representation on the frozen tasks: rxnfp embeddings plus condition identifiers, fixed MLP.

Three stages, so that the pretrained encoder runs in its own environment:

1. export-reactions (analysis environment): one reaction SMILES per substrate pair and product,
   "substrate 1.substrate 2>>product", for every screen -> reactions.csv.
2. embed (an environment with rxnfp 0.1.0, transformers and torch): the pretrained rxnfp BERT
   fingerprint (256 values) of each reaction, without fine-tuning -> embeddings.csv.gz.
3. fit (analysis environment): for every screen, task, split seed and frozen fold, the MLP specification
   of p4_mlp_oof.py on [embedding of the record's reaction, one-hot condition identifier]; out-of-fold
   predictions in the scoring-interface format -> rxnfp_predictions_all.csv.gz. Score with score_predictions.py.

Conditions are not part of the reaction SMILES because the recorded reagents are identifiers, not structures.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

EMBEDDING_COLUMNS = [f"e{i}" for i in range(256)]


def export_reactions(package: Path, output: Path) -> None:
    sys.path.insert(0, str(package))
    from score_predictions import load_population
    from p4_rf_oof_diagnostics import _truth_frames
    truth = _truth_frames(package, load_population(package))
    frames = []
    for name, frame in truth.items():
        frames.append(frame.assign(screen=name, reaction=frame.sub_1_smiles + "." + frame.sub_2_smiles + ">>"
                                   + frame.product_smiles)[["screen", "reaction"]].drop_duplicates())
    table = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "reactions.csv", index=False)
    print(f"{len(table)} screen-reaction rows, {table.reaction.nunique()} distinct reactions")


def embed(output: Path, batch: int = 64) -> None:
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
    table = pd.read_csv(output / "reactions.csv")
    reactions = sorted(table.reaction.unique())
    model, tokenizer = get_default_model_and_tokenizer()
    generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rows, started = [], time.perf_counter()
    for start in range(0, len(reactions), batch):
        chunk = reactions[start:start + batch]
        rows.extend(generator.convert_batch(chunk))
        print(f"{start + len(chunk)}/{len(reactions)} reactions, {time.perf_counter() - started:.0f} s", flush=True)
    emb = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=EMBEDDING_COLUMNS)
    emb.insert(0, "reaction", reactions)
    emb.to_csv(output / "embeddings.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})


_FEATURES = None
_TARGET = None


def _init(features, target):
    global _FEATURES, _TARGET
    _FEATURES, _TARGET = features, target


def _fit(task):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    from p4_mlp_oof import MLP_CONFIG
    seed, protocol, fold, train, test = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MLPRegressor(**MLP_CONFIG).fit(_FEATURES[train], _TARGET[train])
    return seed, protocol, fold, test, model.predict(_FEATURES[test]), int(model.n_iter_)


def fit(package: Path, output: Path, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(package))
    from p4_rf_oof import SEEDS, load_inputs, partitions
    population, screens, scorer, builder = load_inputs(package, None)
    emb = pd.read_csv(output / "embeddings.csv.gz").set_index("reaction")
    for screen in screens:
        target_path = output / f"rxnfp_predictions_{screen.slug}.csv.gz"
        if target_path.exists():
            print(f"skip {screen.name}: predictions exist", flush=True)
            continue
        started = time.perf_counter()
        full = builder.build_features(screen.frame, screen.vocabulary)
        conditions = full[:, 2048:full.shape[1] - 1024]  # substrate 1, substrate 2, condition one-hot, product
        reaction = (screen.frame.sub_1_smiles + "." + screen.frame.sub_2_smiles + ">>" + screen.frame.product_smiles)
        missing = set(reaction) - set(emb.index)
        if missing:
            raise ValueError(f"{screen.name}: {len(missing)} reactions without an embedding")
        features = np.hstack([emb.loc[reaction, EMBEDDING_COLUMNS].to_numpy(np.float32),
                              np.asarray(conditions, dtype=np.float32)])
        target = screen.frame.yield_fraction.to_numpy(float)
        tasks = [(p.seed, p.protocol, p.fold, p.train, p.test) for p in partitions(screen, list(SEEDS))]
        frames = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(features, target)) as pool:
            for seed, protocol, fold, test, prediction, _ in pool.map(_fit, tasks):
                frames.append(pd.DataFrame({"screen": screen.name, "row_uid": screen.frame.row_uid.iloc[test].to_numpy(),
                                            "protocol": protocol, "seed": seed, "fold": fold, "y_pred": prediction}))
        table = pd.concat(frames, ignore_index=True)
        if len(table) != 10 * len(screen.frame) or not np.isfinite(table.y_pred).all():
            raise ValueError(f"{screen.name}: incomplete or nonfinite predictions")
        table.to_csv(target_path, index=False, compression={"method": "gzip", "mtime": 0})
        print(f"{screen.name}: {len(table)} predictions, {time.perf_counter() - started:.0f} s", flush=True)
    pieces = sorted(p for p in output.glob("rxnfp_predictions_*.csv.gz") if p.name != "rxnfp_predictions_all.csv.gz")
    if len(pieces) == len(population.tasks):
        combined = pd.concat([pd.read_csv(p) for p in pieces], ignore_index=True)
        combined.to_csv(output / "rxnfp_predictions_all.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        print(f"combined {len(combined)} predictions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export-reactions", "embed", "fit"))
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True, help="working directory outside the package")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == "export-reactions":
        export_reactions(args.package_dir.resolve(), output)
    elif args.stage == "embed":
        embed(output)
    else:
        fit(args.package_dir.resolve(), output, args.workers)


if __name__ == "__main__":
    main()
