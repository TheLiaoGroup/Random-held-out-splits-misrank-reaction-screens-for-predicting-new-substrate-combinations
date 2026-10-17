"""Additive main-effect reference: substrate 1 + substrate 2 + condition, fitted in every fold.

No random forest is fitted. For each screen, task (row = held-out wells, group = new
combinations), split seed and fold, an additive model

    yield = mu + a[substrate 1] + b[substrate 2] + c[condition]

is fitted to the training folds by backfitting (iterated group means of partial
residuals, so effects are not double counted when the design is unbalanced). Levels
absent from the training folds contribute zero. Test predictions use the frozen fold
assignments stored with the released random-forest out-of-fold predictions, and errors
are pooled exactly as for the random forest. Outputs, written to a separate directory:

- additive_baseline_by_seed.csv: pooled R2, MAE and RMSE of the additive reference and
  of the random forest, and the random forest's squared-error skill over the additive
  reference, for every screen, task and seed.
- additive_baseline_summary.json: five-seed means per screen and cross-screen summaries.

On new combinations, a test well whose two substrates were both seen in training is also
classified by whether they fall in the same connected component of the training substrate graph
(substrate-1 and substrate-2 levels linked by the training pairs). Only within a component is the
additive prediction for a new pair independent of how offsets are shared between a and b; the
forest's gain over the additive model is reported separately on these wells.
"""

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
MAX_ITER = 50000
TOL = 1e-12


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)


def backfit(y, factors, sizes):
    """Fit y = mu + sum_k effect_k[factor_k] by backfitting; return mu, effects and cycles used.

    Iteration continues until no effect changes by more than TOL; failure to converge is an error.
    """
    mu = float(y.mean())
    effects = [np.zeros(n) for n in sizes]
    counts = [np.bincount(f, minlength=n) for f, n in zip(factors, sizes)]
    for cycle in range(1, MAX_ITER + 1):
        change = 0.0
        for k, (f, n) in enumerate(zip(factors, sizes)):
            partial = y - mu - sum(effects[j][factors[j]] for j in range(len(factors)) if j != k)
            sums = np.bincount(f, weights=partial, minlength=n)
            new = np.divide(sums, counts[k], out=np.zeros(n), where=counts[k] > 0)
            change = max(change, float(np.max(np.abs(new - effects[k]))))
            effects[k] = new
        if change < TOL:
            return mu, effects, cycle
    raise RuntimeError(f"Backfitting did not converge in {MAX_ITER} cycles (last change {change:.2e})")


def same_component(s1_train, s2_train, s1_test, s2_test, n1, n2):
    """For each test well, whether its two substrates share a connected component of the training graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(s1_train)), (s1_train, n1 + s2_train)), shape=(n1 + n2, n1 + n2))
    _, labels = connected_components(graph, directed=False)
    return labels[s1_test] == labels[n1 + s2_test]


def run(package: Path, results: Path):
    sys.path.insert(0, str(package))
    from score_predictions import load_population
    from p4_rf_oof_diagnostics import _truth_frames

    population = load_population(package)
    truth = _truth_frames(package, population)
    rows = []
    cycles = []
    both_seen = []
    for task in population.tasks:
        frame = truth[task.name].set_index("row_uid").loc[list(task.row_uids)].reset_index()
        codes = [pd.factorize(frame[col].astype(str))[0] for col in ("sub_1_smiles", "sub_2_smiles", "condition_id")]
        sizes = [int(c.max()) + 1 for c in codes]
        y_all = frame.yield_fraction.to_numpy(float)
        if not np.array_equal(y_all, task.y):
            raise ValueError(f"{task.name}: truth frame disagrees with frozen yields")
        sst = float(np.sum((y_all - y_all.mean()) ** 2))
        oof = pd.read_csv(results / "sources" / slug(task.name) / "oof_predictions.csv.gz",
                          usecols=["row_uid", "protocol", "seed", "fold", "y_true", "y_pred", "n_unseen"])
        position = pd.Series(np.arange(len(frame)), index=frame.row_uid)
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
                unseen = np.empty(len(frame), dtype=int)
                unseen[idx] = part.n_unseen.to_numpy(int)
                if not np.allclose(y_all[idx], part.y_true.to_numpy(float), rtol=0, atol=1e-9):
                    raise ValueError(f"{task.name}: predictions misaligned with truth")
                additive = np.empty(len(frame))
                within = np.zeros(len(frame), dtype=bool)
                for f in range(5):
                    test = fold == f
                    train = ~test
                    mu, effects, used = backfit(y_all[train], [c[train] for c in codes], sizes)
                    cycles.append({"screen": task.name, "protocol": protocol, "seed": seed, "fold": f,
                                   "cycles": used})
                    pred = np.full(int(test.sum()), mu)
                    for c, e in zip(codes, effects):
                        pred += e[c[test]]
                    additive[test] = pred
                    if protocol == "group":
                        within[test] = same_component(codes[0][train], codes[1][train], codes[0][test], codes[1][test],
                                                      sizes[0], sizes[1])
                if protocol == "group":
                    seen = unseen == 0
                    inside = seen & within
                    both_seen.append({"screen": task.name, "seed": seed, "rows": int(seen.sum()),
                                      "sse_additive": float(np.sum((y_all[seen] - additive[seen]) ** 2)),
                                      "sse_rf": float(np.sum((y_all[seen] - rf[seen]) ** 2)),
                                      "rows_within_component": int(inside.sum()),
                                      "sse_additive_within_component": float(np.sum((y_all[inside] - additive[inside]) ** 2)),
                                      "sse_rf_within_component": float(np.sum((y_all[inside] - rf[inside]) ** 2))})
                for label, pred in (("additive", additive), ("rf", rf)):
                    err = y_all - pred
                    rows.append({"screen": task.name, "protocol": protocol, "seed": seed, "model": label,
                                 "rows": len(frame), "sse": float(np.sum(err ** 2)),
                                 "sae": float(np.sum(np.abs(err))), "sst": sst,
                                 "r2": 1.0 - float(np.sum(err ** 2)) / sst,
                                 "mae_percentage_points": 100 * float(np.mean(np.abs(err))),
                                 "rmse_percentage_points": 100 * float(np.sqrt(np.mean(err ** 2)))})
    table = pd.DataFrame(rows)
    wide = table.pivot_table(index=["screen", "protocol", "seed"], columns="model",
                             values=["r2", "mae_percentage_points", "sse"]).reset_index()
    wide.columns = ["_".join(c for c in col if c) for col in wide.columns]
    wide["rf_skill_over_additive"] = 1.0 - wide.sse_rf / wide.sse_additive
    means = wide.groupby(["screen", "protocol"]).mean(numeric_only=True).drop(columns="seed")
    seen_table = pd.DataFrame(both_seen)
    seen_table["rf_skill_over_additive_both_seen"] = 1.0 - seen_table.sse_rf / seen_table.sse_additive
    seen_gain = seen_table.groupby("screen").rf_skill_over_additive_both_seen.mean()
    if (seen_table.rows_within_component == 0).any():
        raise ValueError("A screen and seed has no within-component both-seen wells")
    seen_table["rf_skill_over_additive_within_component"] = (
        1.0 - seen_table.sse_rf_within_component / seen_table.sse_additive_within_component)
    seen_table["cross_component_share_of_both_seen"] = 1.0 - seen_table.rows_within_component / seen_table.rows
    within_gain = seen_table.groupby("screen").rf_skill_over_additive_within_component.mean()
    cross_share = seen_table.groupby("screen").cross_component_share_of_both_seen.mean()
    rf_check = json.loads((results / "diagnostics" / "rf_pooled_model_results.json").read_text("utf-8"))
    for source in rf_check["sources"]:
        for protocol in PROTOCOLS:
            expected = np.mean([r["r2"] for r in source["pooled_by_seed"][protocol]])
            if not np.isclose(means.loc[(source["source_name"], protocol), "r2_rf"], expected, rtol=0, atol=1e-10):
                raise ValueError(f"{source['source_name']}: RF R2 recomputation disagrees with release")
    mean_yield = {s["source_name"]: s["mean_yield_fraction"] for s in rf_check["sources"]}
    structure = pd.read_csv(results / "diagnostics" / "rf_yield_variance_structure.csv").set_index("screen")
    higher = sorted(name for name, value in mean_yield.items() if value >= 0.05)
    cycle_table = pd.DataFrame(cycles)
    summary = {"per_screen": {}, "subsets": {},
               "convergence": {"tolerance": TOL, "max_cycles_allowed": MAX_ITER, "fits": int(len(cycle_table)),
                               "max_cycles_used": int(cycle_table.cycles.max()),
                               "median_cycles_used": float(cycle_table.cycles.median()),
                               "max_cycles_used_by_task": {k: int(v) for k, v in
                                                           cycle_table.groupby("protocol").cycles.max().items()}}}
    for name in sorted(mean_yield):
        summary["per_screen"][name] = {
            protocol: {key: float(means.loc[(name, protocol), key])
                       for key in ("r2_additive", "r2_rf", "mae_percentage_points_additive",
                                   "mae_percentage_points_rf", "rf_skill_over_additive")}
            for protocol in PROTOCOLS}
        summary["per_screen"][name]["group"]["rf_skill_over_additive_both_seen"] = float(seen_gain.loc[name])
        summary["per_screen"][name]["group"]["rf_skill_over_additive_within_component"] = float(within_gain.loc[name])
        summary["per_screen"][name]["group"]["cross_component_share_of_both_seen"] = float(cross_share.loc[name])
    for label, selected in (("all_18", sorted(mean_yield)), ("mean_yield_at_least_5pct", higher)):
        get = lambda key, protocol: np.array([summary["per_screen"][n][protocol][key] for n in selected])
        add_loss = get("r2_additive", "row") - get("r2_additive", "group")
        rf_loss = get("r2_rf", "row") - get("r2_rf", "group")
        summary["subsets"][label] = {
            "screens": len(selected),
            "median_additive_r2_loss": float(np.median(add_loss)),
            "median_rf_r2_loss": float(np.median(rf_loss)),
            "additive_r2_rank_rho_between_tasks": float(spearmanr(get("r2_additive", "row"), get("r2_additive", "group")).statistic),
            "screens_where_rf_beats_additive_on_new_combinations": int(np.sum(get("rf_skill_over_additive", "group") > 0)),
            "median_additive_fraction_of_rf_held_out_well_r2": float(np.median(get("r2_additive", "row") / get("r2_rf", "row"))),
            "median_rf_skill_over_additive_new_combinations": float(np.median(get("rf_skill_over_additive", "group"))),
            "rho_pair_share_vs_rf_skill_over_additive_new_combinations": float(spearmanr(
                structure.loc[selected, "pair_variance_share"], get("rf_skill_over_additive", "group")).statistic),
            "p_pair_share_vs_rf_skill_over_additive_new_combinations": float(spearmanr(
                structure.loc[selected, "pair_variance_share"], get("rf_skill_over_additive", "group")).pvalue),
            "rho_rf_held_out_well_r2_vs_rf_skill_over_additive_new_combinations": float(spearmanr(
                get("r2_rf", "row"), get("rf_skill_over_additive", "group")).statistic),
            "rho_additive_r2_loss_vs_rf_r2_loss": float(spearmanr(add_loss, rf_loss).statistic),
            "rho_pair_share_vs_rf_skill_over_additive_both_seen": float(spearmanr(
                structure.loc[selected, "pair_variance_share"], seen_gain.loc[selected]).statistic),
            "screens_where_rf_beats_additive_within_component": int(np.sum(within_gain.loc[selected] > 0)),
            "median_rf_skill_over_additive_within_component": float(within_gain.loc[selected].median()),
            "rho_pair_share_vs_rf_skill_over_additive_within_component": float(spearmanr(
                structure.loc[selected, "pair_variance_share"], within_gain.loc[selected]).statistic),
            "p_pair_share_vs_rf_skill_over_additive_within_component": float(spearmanr(
                structure.loc[selected, "pair_variance_share"], within_gain.loc[selected]).pvalue),
            "median_cross_component_share_of_both_seen": float(cross_share.loc[selected].median()),
            "range_cross_component_share_of_both_seen": [float(cross_share.loc[selected].min()),
                                                         float(cross_share.loc[selected].max())],
            "rho_substrate_levels_per_pair_vs_additive_fraction_of_rf_held_out_well_r2": float(spearmanr(
                structure.loc[selected, "substrate_levels_per_pair"],
                get("r2_additive", "row") / get("r2_rf", "row")).statistic),
        }
    return table, wide, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table, wide, summary = run(args.package_dir.resolve(), args.results_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.output_dir / "additive_baseline_by_seed.csv", index=False)
    (args.output_dir / "additive_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["subsets"], indent=2))


if __name__ == "__main__":
    main()
