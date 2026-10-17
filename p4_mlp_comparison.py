"""Compare the multilayer perceptron with the random forest on the frozen tasks, without fitting.

Inputs: the MLP prediction files written by p4_mlp_oof.py (scoring-interface format), the frozen
numerical populations and the released random-forest diagnostics. Errors are pooled exactly as for
the random forest (five folds per seed, R2 against the full-screen yield variance, five-seed means).
Outputs, in a separate directory: <label>_comparison_by_screen.csv and <label>_comparison_summary.json
(label "mlp" by default; "rxnfp" for the predictions of p4_rxnfp_oof.py).
The summary also relates the R2 lost by each of the three models (random forest, additive identity
model, MLP) to the between-pair variance share and to the substrate levels per pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEEDS = (42, 20261007, 20261008, 20261009, 20261010)


def rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="combined prediction CSV.gz")
    parser.add_argument("--label", default="mlp", help="column and file prefix for the compared model")
    parser.add_argument("--description", default="MLPRegressor (256, 128); see p4_mlp_oof.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    L = args.label
    package = args.package_dir.resolve()
    sys.path.insert(0, str(package))
    from score_predictions import load_population
    from p4_rf_oof_diagnostics import _truth_frames
    from p4_ranking_structure_diagnostics import pair_level_decomposition

    population = load_population(package)
    truth = _truth_frames(package, population)
    predictions = pd.read_csv(args.predictions)
    rf = pd.read_csv(package / "output/p4_unified_rf/diagnostics/rf_model_comparison.csv").set_index("screen")
    structure = pd.read_csv(package / "output/p4_unified_rf/diagnostics/rf_yield_variance_structure.csv").set_index("screen")
    additive = pd.read_csv(package / "output/p4_unified_rf/diagnostics/additive_baseline_by_seed.csv")
    additive = additive.groupby(["screen", "protocol"]).r2_additive.mean().unstack()
    records = []
    for task in population.tasks:
        frame = truth[task.name].set_index("row_uid")
        y_all = frame.yield_fraction
        sst = float(np.sum((y_all - y_all.mean()) ** 2))
        part_all = predictions.loc[predictions.screen.eq(task.name)]
        if len(part_all) != 10 * len(frame):
            raise ValueError(f"{task.name}: incomplete {L} predictions")
        record = {"screen": task.name, "mean_yield_percent": float(rf.loc[task.name, "mean_yield_percent"])}
        for protocol in ("row", "group"):
            r2s, maes, levels, withins = [], [], [], []
            for seed in SEEDS:
                part = part_all.loc[part_all.protocol.eq(protocol) & part_all.seed.eq(seed)]
                if len(part) != len(frame) or part.row_uid.duplicated().any():
                    raise ValueError(f"{task.name}/{protocol}/{seed}: prediction coverage")
                y = y_all.loc[part.row_uid].to_numpy(float)
                p = part.y_pred.to_numpy(float)
                r2s.append(1.0 - float(np.sum((y - p) ** 2)) / sst)
                maes.append(100 * float(np.mean(np.abs(y - p))))
                level, within = pair_level_decomposition(frame.pair_key.loc[part.row_uid], y, p)
                levels.append(level)
                withins.append(within)
            record[f"{L}_{protocol}_r2"] = float(np.mean(r2s))
            record[f"{L}_{protocol}_mae_percentage_points"] = float(np.mean(maes))
            record[f"{L}_{protocol}_pair_level_r2"] = float(np.mean(levels))
            record[f"{L}_{protocol}_within_pair_r2"] = float(np.mean(withins))
            record[f"rf_{protocol}_r2"] = float(rf.loc[task.name, f"rf_{protocol}_r2"])
        record[f"{L}_delta_r2"] = record[f"{L}_row_r2"] - record[f"{L}_group_r2"]
        record["rf_delta_r2"] = float(rf.loc[task.name, "rf_delta_r2"])
        record["pair_variance_share"] = float(structure.loc[task.name, "pair_variance_share"])
        record["substrate_levels_per_pair"] = float(structure.loc[task.name, "substrate_levels_per_pair"])
        record["additive_delta_r2"] = float(additive.loc[task.name, "row"] - additive.loc[task.name, "group"])
        records.append(record)
    table = pd.DataFrame(records)
    higher = table.loc[table.mean_yield_percent >= 5, "screen"].tolist()
    summary = {"model": args.description, "subsets": {}}
    for name, selected in (("all_18", table.screen.tolist()), ("mean_yield_at_least_5pct", higher)):
        part = table.set_index("screen").loc[selected]
        row_order = part[f"{L}_row_r2"].rank(ascending=False, method="min")
        group_order = part[f"{L}_group_r2"].rank(ascending=False, method="min")
        summary["subsets"][name] = {
            "screens": len(selected),
            f"{L}_between_task_r2_rank_rho": rho(part[f"{L}_row_r2"], part[f"{L}_group_r2"]),
            f"{L}_screens_with_lower_new_combination_r2": int((part[f"{L}_delta_r2"] > 0).sum()),
            f"median_{L}_delta_r2": float(part[f"{L}_delta_r2"].median()),
            f"rho_{L}_vs_rf_r2_loss": rho(part[f"{L}_delta_r2"], part.rf_delta_r2),
            f"rho_{L}_vs_rf_new_combination_r2": rho(part[f"{L}_group_r2"], part.rf_group_r2),
            f"rho_pair_share_vs_{L}_r2_loss": rho(part.pair_variance_share, part[f"{L}_delta_r2"]),
            f"median_{L}_row_pair_level_r2": float(part[f"{L}_row_pair_level_r2"].median()),
            f"median_{L}_group_pair_level_r2": float(part[f"{L}_group_pair_level_r2"].median()),
            f"median_{L}_row_within_pair_r2": float(part[f"{L}_row_within_pair_r2"].median()),
            f"median_{L}_group_within_pair_r2": float(part[f"{L}_group_within_pair_r2"].median()),
            **{f"{L}_paired_gain_over_rf_{protocol}_r2": {
                "median": float((part[f"{L}_{protocol}_r2"] - part[f"rf_{protocol}_r2"]).median()),
                "min": float((part[f"{L}_{protocol}_r2"] - part[f"rf_{protocol}_r2"]).min()),
                "min_screen": str((part[f"{L}_{protocol}_r2"] - part[f"rf_{protocol}_r2"]).idxmin()),
                "max": float((part[f"{L}_{protocol}_r2"] - part[f"rf_{protocol}_r2"]).max()),
                "max_screen": str((part[f"{L}_{protocol}_r2"] - part[f"rf_{protocol}_r2"]).idxmax()),
                "screens_above_rf": int(((part[f"{L}_{protocol}_r2"] - part[f"rf_{protocol}_r2"]) > 0).sum())}
               for protocol in ("row", "group")},
            f"{L}_screens_with_lower_new_combination_mae_than_rf": int(
                (part[f"{L}_group_mae_percentage_points"] < rf.loc[selected, "rf_group_mae"]).sum()),
            "design_rho_by_model": {
                model: {"pair_variance_share": rho(part.pair_variance_share, part[f"{model}_delta_r2"]),
                        "substrate_levels_per_pair": rho(part.substrate_levels_per_pair, part[f"{model}_delta_r2"])}
                for model in ("rf", "additive", L)},
            "ranks": {screen: {"held_out_wells": int(row_order[screen]), "new_combinations": int(group_order[screen])}
                      for screen in selected},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f"{L}_comparison_by_screen.csv", index=False)
    (args.output_dir / f"{L}_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ranks"} for k, v in summary["subsets"].items()}, indent=2))


if __name__ == "__main__":
    main()
