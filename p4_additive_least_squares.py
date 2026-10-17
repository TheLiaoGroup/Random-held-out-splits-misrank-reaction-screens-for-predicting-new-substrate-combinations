"""Least-squares controls for the additive identity reference of p4_additive_baseline.py.

No random forest is fitted. For every screen, task, split seed and fold, the additive model
yield = mu + a[substrate 1] + b[substrate 2] + c[condition] is fitted to the training folds by
ordinary least squares on indicator variables, with mu fixed at the training mean and the
minimum-norm solution for the effects (levels absent from the training folds contribute zero, as
in backfitting). Two comparisons are reported:

- held-out wells: the pooled R2 of the least-squares fit against the backfitted R2 released in
  additive_baseline_by_seed.csv;
- new combinations: the random forest's squared-error gain over the least-squares fit, the number
  of screens where it is positive and its Spearman correlation with the between-pair share.

Output, written to a separate directory: additive_least_squares_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from p4_additive_baseline import PROTOCOLS, SEEDS, slug


def min_norm_fit(codes, sizes, y, train, test):
    """Minimum-norm least-squares effects on indicator variables, with mu fixed at the training mean."""
    n = len(y)
    blocks = [sparse.csr_matrix((np.ones(n), (np.arange(n), c)), shape=(n, size)) for c, size in zip(codes, sizes)]
    design = sparse.hstack(blocks).tocsr()
    x_train = design[train]
    mu = float(y[train].mean())
    gram = (x_train.T @ x_train).toarray()
    beta = np.linalg.pinv(gram, rcond=1e-10, hermitian=True) @ (x_train.T @ (y[train] - mu))
    return mu + design[test] @ beta


def run(package: Path, results: Path):
    sys.path.insert(0, str(package))
    from score_predictions import load_population
    from p4_rf_oof_diagnostics import _truth_frames

    population = load_population(package)
    truth = _truth_frames(package, population)
    backfit = pd.read_csv(results / "diagnostics" / "additive_baseline_by_seed.csv")
    backfit = backfit.groupby(["screen", "protocol"]).mean(numeric_only=True)
    structure = pd.read_csv(results / "diagnostics" / "rf_yield_variance_structure.csv").set_index("screen")
    per_screen = {}
    for task in population.tasks:
        frame = truth[task.name].set_index("row_uid").loc[list(task.row_uids)].reset_index()
        codes = [pd.factorize(frame[col].astype(str))[0] for col in ("sub_1_smiles", "sub_2_smiles", "condition_id")]
        sizes = [int(c.max()) + 1 for c in codes]
        y_all = frame.yield_fraction.to_numpy(float)
        if not np.array_equal(y_all, task.y):
            raise ValueError(f"{task.name}: truth frame disagrees with frozen yields")
        sst = float(np.sum((y_all - y_all.mean()) ** 2))
        oof = pd.read_csv(results / "sources" / slug(task.name) / "oof_predictions.csv.gz",
                          usecols=["row_uid", "protocol", "seed", "fold", "y_true", "y_pred"])
        position = pd.Series(np.arange(len(frame)), index=frame.row_uid)
        scores = {protocol: {"r2": [], "skill": []} for protocol in PROTOCOLS}
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                part = oof.loc[oof.protocol.eq(protocol) & oof.seed.eq(seed)]
                if len(part) != len(frame) or part.row_uid.duplicated().any():
                    raise ValueError(f"{task.name}/{protocol}/{seed}: incomplete predictions")
                idx = position.loc[part.row_uid].to_numpy()
                fold = np.empty(len(frame), dtype=int)
                fold[idx] = part.fold.to_numpy(int)
                rf = np.empty(len(frame))
                rf[idx] = part.y_pred.to_numpy(float)
                if not np.allclose(y_all[idx], part.y_true.to_numpy(float), rtol=0, atol=1e-9):
                    raise ValueError(f"{task.name}: predictions misaligned with truth")
                pred = np.empty(len(frame))
                for f in range(5):
                    test = fold == f
                    pred[test] = min_norm_fit(codes, sizes, y_all, ~test, test)
                sse = float(np.sum((y_all - pred) ** 2))
                scores[protocol]["r2"].append(1.0 - sse / sst)
                scores[protocol]["skill"].append(1.0 - float(np.sum((y_all - rf) ** 2)) / sse)
        per_screen[task.name] = {
            protocol: {"r2_least_squares": float(np.mean(scores[protocol]["r2"])),
                       "r2_backfitting": float(backfit.loc[(task.name, protocol), "r2_additive"]),
                       "rf_skill_over_least_squares": float(np.mean(scores[protocol]["skill"]))}
            for protocol in PROTOCOLS}
        print(task.name, json.dumps(per_screen[task.name]), flush=True)
    rf_check = json.loads((results / "diagnostics" / "rf_pooled_model_results.json").read_text("utf-8"))
    mean_yield = {s["source_name"]: s["mean_yield_fraction"] for s in rf_check["sources"]}
    higher = sorted(name for name, value in mean_yield.items() if value >= 0.05)
    summary = {"per_screen": per_screen, "subsets": {}}
    for label, selected in (("all_18", sorted(per_screen)), ("mean_yield_at_least_5pct", higher)):
        row_diff = [abs(per_screen[n]["row"]["r2_least_squares"] - per_screen[n]["row"]["r2_backfitting"]) for n in selected]
        skill = np.array([per_screen[n]["group"]["rf_skill_over_least_squares"] for n in selected])
        summary["subsets"][label] = {
            "screens": len(selected),
            "max_abs_held_out_well_r2_difference_from_backfitting": float(max(row_diff)),
            "screens_where_rf_beats_least_squares_on_new_combinations": int(np.sum(skill > 0)),
            "rho_pair_share_vs_rf_skill_over_least_squares_new_combinations": float(spearmanr(
                structure.loc[selected, "pair_variance_share"], skill).statistic),
            "p_pair_share_vs_rf_skill_over_least_squares_new_combinations": float(spearmanr(
                structure.loc[selected, "pair_variance_share"], skill).pvalue),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.package_dir.resolve(), args.results_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "additive_least_squares_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["subsets"], indent=2))


if __name__ == "__main__":
    main()
