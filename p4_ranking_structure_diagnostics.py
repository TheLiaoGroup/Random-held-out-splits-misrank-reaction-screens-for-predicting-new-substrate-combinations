"""Relate the unified-RF screen rankings to where each screen's yield variation lies.

No model is fitted. Inputs are the frozen numerical populations and the released
unified-RF diagnostics. Outputs, written to a separate directory:

- rf_yield_variance_structure.csv: per screen, the share of total yield variance
  between substrate-pair means and between condition means, the RF scores, the
  share of wells predicted within 10 yield percentage points and the both-seen
  same-well MAE.
- rf_pairwise_reversals.csv: for the 11 screens with mean yield >= 5%, every screen
  pair, whether its mean R2 order reverses between protocols, and in how many of
  the five split seeds it reverses.
- rf_ranking_structure_summary.json: Spearman correlations and per-seed ranks.

The structure table also gives, per screen, the held-out-well R2 of quoting the mean yield of the
same pair's training wells, and the between-pair share after condition effects: the reduction in
squared error, as a share of the total, when pair effects are added to a model of condition means.
The pair + condition additive model is fitted to all numerical wells by backfitting; the reduction
depends only on its fitted values, so it does not depend on how effects are shared between pairs and
conditions when the pair-condition graph is disconnected.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEEDS = (42, 20261007, 20261008, 20261009, 20261010)
SUBSETS = ("all_18", "mean_yield_at_least_5pct")


def between_share(frame: pd.DataFrame, key: str) -> float:
    """Share of total sum of squares carried by the means of `key` groups."""
    y = frame.yield_fraction.to_numpy(float)
    sst = float(np.sum((y - y.mean()) ** 2))
    group_mean = frame.groupby(key).yield_fraction.transform("mean").to_numpy(float)
    return float(np.sum((group_mean - y.mean()) ** 2) / sst)


def condition_adjusted_pair_share(frame: pd.DataFrame) -> float:
    from p4_additive_baseline import backfit
    y = frame.yield_fraction.to_numpy(float)
    pair = pd.factorize(frame.pair_key)[0]
    condition = pd.factorize(frame.condition_id.astype(str))[0]
    mu, (pair_effect, condition_effect), _ = backfit(y, [pair, condition],
                                                     [int(pair.max()) + 1, int(condition.max()) + 1])
    sse_full = np.sum((y - mu - pair_effect[pair] - condition_effect[condition]) ** 2)
    condition_mean = np.bincount(condition, weights=y) / np.bincount(condition)
    sse_condition = np.sum((y - condition_mean[condition]) ** 2)
    return float((sse_condition - sse_full) / np.sum((y - y.mean()) ** 2))


def rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def plate_structure(frame: pd.DataFrame) -> dict:
    """Plate-to-plate variation within substrate pairs that were run on more than one plate.

    Each plate holds a single pair. For pairs on several plates, the spread of plate means
    around the pair mean combines plate effects with differences in condition composition,
    because the plates of one pair generally carry different conditions.
    """
    if frame.groupby("plate_group").pair_key.nunique().max() != 1:
        raise ValueError("A plate holds more than one substrate pair")
    plates_per_pair = frame.groupby("pair_key").plate_group.nunique()
    multi = plates_per_pair.index[plates_per_pair > 1]
    y = frame.yield_fraction.to_numpy(float)
    sst = float(np.sum((y - y.mean()) ** 2))
    pair_mean = frame.groupby("pair_key").yield_fraction.transform("mean").to_numpy(float)
    plate_mean = frame.groupby("plate_group").yield_fraction.transform("mean").to_numpy(float)
    record = {"plates": int(frame.plate_group.nunique()), "pairs_on_multiple_plates": int(len(multi)),
              "within_pair_between_plate_share": float(np.sum((plate_mean - pair_mean) ** 2) / sst)}
    if len(multi):
        part = frame.loc[frame.pair_key.isin(multi)]
        ym = part.yield_fraction.to_numpy(float)
        sstm = float(np.sum((ym - ym.mean()) ** 2))
        pm = part.groupby("pair_key").yield_fraction.transform("mean").to_numpy(float)
        plm = part.groupby("plate_group").yield_fraction.transform("mean").to_numpy(float)
        record.update({"multi_plate_wells": int(len(part)),
                       "multi_plate_between_pair_share": float(np.sum((pm - ym.mean()) ** 2) / sstm),
                       "multi_plate_within_pair_between_plate_share": float(np.sum((plm - pm) ** 2) / sstm)})
    else:
        record.update({"multi_plate_wells": 0, "multi_plate_between_pair_share": np.nan,
                       "multi_plate_within_pair_between_plate_share": np.nan})
    return record


def pairing_design(frame: pd.DataFrame) -> dict:
    """How sparsely the substrate pairs cover the substrate-by-substrate grid.

    substrate_levels_per_pair = (substrate-1 SMILES + substrate-2 SMILES - 1) / pairs: the number of
    additive substrate effects available per pair. Near 1, substrate identity nearly fixes each pair.
    """
    pairs = frame.drop_duplicates("pair_key")
    n1, n2 = pairs.sub_1_smiles.nunique(), pairs.sub_2_smiles.nunique()
    return {"substrate_levels_per_pair": float((n1 + n2 - 1) / len(pairs)),
            "grid_fill": float(len(pairs) / (n1 * n2)),
            "median_partners_substrate_1": float(pairs.groupby("sub_1_smiles").size().median()),
            "median_partners_substrate_2": float(pairs.groupby("sub_2_smiles").size().median())}


def pair_level_decomposition(pairs: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Split pooled out-of-fold accuracy into pair-level and within-pair parts.

    Pair-level R2 compares pair-mean predictions with pair-mean yields (every well weighted once);
    within-pair R2 compares deviations from the pair mean, i.e. how conditions rank within a pair.
    """
    frame = pd.DataFrame({"pair": pairs.to_numpy(), "y": y_true, "p": y_pred})
    grouped = frame.groupby("pair")
    y_bar = grouped.y.transform("mean").to_numpy()
    p_bar = grouped.p.transform("mean").to_numpy()
    between = 1.0 - np.sum((y_bar - p_bar) ** 2) / np.sum((y_bar - y_true.mean()) ** 2)
    within = 1.0 - np.sum(((y_true - y_bar) - (y_pred - p_bar)) ** 2) / np.sum((y_true - y_bar) ** 2)
    return float(between), float(within)


def sister_well_mean_r2(pairs: np.ndarray, y: np.ndarray, folds: np.ndarray, sst: float) -> float:
    """Held-out-well R2 of predicting each test well by the mean yield of its pair's training wells."""
    prediction = np.empty(len(y))
    for fold in np.unique(folds):
        test = folds == fold
        means = pd.Series(y[~test]).groupby(pairs[~test]).mean()
        prediction[test] = pd.Series(pairs[test]).map(means).fillna(y[~test].mean()).to_numpy()
    return float(1.0 - np.sum((y - prediction) ** 2) / sst)


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def bootstrap_rho(x, y, draws=10000, seed=20261011):
    """Percentile interval for Spearman rho when screens are resampled with replacement."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(idx)) < 3:
            continue
        value = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value):
            values.append(value)
    return {"draws_used": len(values), "low_2.5": float(np.percentile(values, 2.5)),
            "high_97.5": float(np.percentile(values, 97.5))}


def variance_structure(package: Path, results: Path) -> pd.DataFrame:
    sys.path.insert(0, str(package))
    from score_predictions import load_population
    from p4_rf_oof_diagnostics import _truth_frames

    population = load_population(package)
    truth = _truth_frames(package, population)
    scores = pd.read_csv(results / "diagnostics" / "rf_model_comparison.csv").set_index("screen")
    records = []
    for task in population.tasks:
        frame = truth[task.name]
        if not np.allclose(frame.yield_fraction.to_numpy(float), task.y, rtol=0, atol=0):
            raise ValueError(f"{task.name}: truth frame disagrees with frozen yields")
        if frame.pair_key.isna().any() or frame.condition_id.isna().any():
            raise ValueError(f"{task.name}: missing pair or condition identity")
        score = scores.loc[task.name]
        records.append({
            "screen": task.name,
            "numeric_records": len(frame),
            "pairs": int(frame.pair_key.nunique()),
            "condition_ids": int(frame.condition_id.nunique()),
            "mean_yield_percent": float(score.mean_yield_percent),
            "pair_variance_share": between_share(frame, "pair_key"),
            "condition_variance_share": between_share(frame, "condition_id"),
            "condition_adjusted_pair_variance_share": condition_adjusted_pair_share(frame),
            "zero_yield_well_share": float((frame.yield_fraction == 0).mean()),
            "wells_at_least_20pct_share": float((frame.yield_fraction >= 0.2).mean()),
            "rf_row_r2": float(score.rf_row_r2),
            "rf_group_r2": float(score.rf_group_r2),
            "rf_delta_r2": float(score.rf_delta_r2),
            **plate_structure(frame),
            **pairing_design(frame),
        })
    table = pd.DataFrame(records)
    if len(table) != 18 or table.numeric_records.sum() != 340178:
        raise ValueError("Unexpected frozen 18-screen population")
    return table


def seed_scores(results: Path) -> dict[str, dict[str, np.ndarray]]:
    pooled = json.loads((results / "diagnostics" / "rf_pooled_model_results.json").read_text("utf-8"))
    out = {}
    for source in pooled["sources"]:
        entry = {}
        for protocol in ("row", "group"):
            records = sorted(source["pooled_by_seed"][protocol], key=lambda r: SEEDS.index(r["seed"]))
            if [r["seed"] for r in records] != list(SEEDS):
                raise ValueError(f"{source['source_name']}: unexpected seeds")
            entry[protocol] = np.array([r["r2"] for r in records])
        out[source["source_name"]] = entry
    return out


def pairwise_reversals(screens: list[str], scores) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(screens, 2):
        row_diff = scores[a]["row"] - scores[b]["row"]
        group_diff = scores[a]["group"] - scores[b]["group"]
        mean_product = row_diff.mean() * group_diff.mean()
        if mean_product == 0 or np.any(row_diff * group_diff == 0):
            raise ValueError(f"Tied R2 order for {a} / {b}")
        rows.append({"screen_a": a, "screen_b": b,
                     "row_r2_difference": float(row_diff.mean()),
                     "group_r2_difference": float(group_diff.mean()),
                     "reversed_in_mean": bool(mean_product < 0),
                     "seeds_reversed": int(np.sum(row_diff * group_diff < 0))})
    return pd.DataFrame(rows)


def run(package: Path, results: Path):
    table = variance_structure(package, results)
    higher = table.loc[table.mean_yield_percent >= 5, "screen"].tolist()
    if len(higher) != 11:
        raise ValueError("The non-low-yield stratum must contain 11 screens")
    summary = {"variance_structure_rho": {}, "raw_mae_vs_median_mae_rho": {}}
    for name, selected in zip(SUBSETS, (table.screen.tolist(), higher)):
        part = table.set_index("screen").loc[selected]
        summary["variance_structure_rho"][name] = {
            f"{share}_vs_{score}": rho(part[share], part[score])
            for share in ("pair_variance_share", "condition_variance_share")
            for score in ("rf_delta_r2", "rf_group_r2", "rf_row_r2")
        }
        baseline = pd.read_csv(results / "diagnostics" / "rf_median_baseline_by_seed.csv")
        means = baseline.loc[baseline.screen.isin(selected)].groupby(
            ["screen", "protocol"]).mean(numeric_only=True)
        summary["raw_mae_vs_median_mae_rho"][name] = {
            protocol: rho(means.xs(protocol, level="protocol").model_mae_percentage_points,
                          means.xs(protocol, level="protocol").training_median_mae_percentage_points)
            for protocol in ("row", "group")
        }
    matched = pd.read_csv(results / "diagnostics" / "rf_matched_cohorts_by_seed.csv")
    seen = matched.loc[matched.group_exposure_stratum.eq("both_seen") & matched["rows"].gt(0)]
    seen = seen.groupby("screen")[["row_mae", "group_mae"]].mean()
    table["both_seen_row_mae_percentage_points"] = table.screen.map(seen.row_mae * 100)
    table["both_seen_group_mae_percentage_points"] = table.screen.map(seen.group_mae * 100)
    table["both_seen_relative_mae_increase"] = (table.both_seen_group_mae_percentage_points
                                                / table.both_seen_row_mae_percentage_points - 1)
    if table.both_seen_relative_mae_increase.isna().any():
        raise ValueError("Every screen must have a nonempty both-seen cohort")
    summary["pair_share_vs_both_seen_relative_mae_increase_rho"] = {}
    for name, selected in zip(SUBSETS, (table.screen.tolist(), higher)):
        part = table.set_index("screen").loc[selected]
        summary["pair_share_vs_both_seen_relative_mae_increase_rho"][name] = rho(
            part.pair_variance_share, part.both_seen_relative_mae_increase)
    for protocol in ("row", "group"):
        table[f"{protocol}_within_10pp_share"] = np.nan
        table[f"{protocol}_pair_level_r2"] = np.nan
        table[f"{protocol}_within_pair_r2"] = np.nan
    table["row_sister_well_mean_r2"] = np.nan
    sys.path.insert(0, str(package))
    from score_predictions import load_population
    from p4_rf_oof_diagnostics import _truth_frames
    truth = _truth_frames(package, load_population(package))
    frame_pairs = {name: frame.set_index("row_uid").pair_key for name, frame in truth.items()}
    for index, name in table.screen.items():
        slug = "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)
        oof = pd.read_csv(results / "sources" / slug / "oof_predictions.csv.gz",
                          usecols=["row_uid", "protocol", "seed", "fold", "y_true", "y_pred"])
        if len(oof) != table.at[index, "numeric_records"] * 10:
            raise ValueError(f"{name}: incomplete out-of-fold predictions")
        within = (oof.y_true - oof.y_pred).abs().le(0.10)
        pair_of = frame_pairs[name]
        for protocol in ("row", "group"):
            parts = [pair_level_decomposition(pair_of.loc[part.row_uid], part.y_true.to_numpy(float), part.y_pred.to_numpy(float))
                     for _, part in oof.loc[oof.protocol.eq(protocol)].groupby("seed")]
            table.at[index, f"{protocol}_pair_level_r2"] = float(np.mean([b for b, _ in parts]))
            table.at[index, f"{protocol}_within_pair_r2"] = float(np.mean([w for _, w in parts]))
        y_screen = truth[name].yield_fraction.to_numpy(float)
        sst = float(np.sum((y_screen - y_screen.mean()) ** 2))
        table.at[index, "row_sister_well_mean_r2"] = float(np.mean([
            sister_well_mean_r2(pair_of.loc[part.row_uid].to_numpy(), part.y_true.to_numpy(float), part.fold.to_numpy(), sst)
            for _, part in oof.loc[oof.protocol.eq("row")].groupby("seed")]))
        share = within.groupby([oof.protocol, oof.seed]).mean().groupby(level=0).mean()
        for protocol in ("row", "group"):
            table.at[index, f"{protocol}_within_10pp_share"] = float(share[protocol])
    scores = seed_scores(results)
    reversals = pairwise_reversals(higher, scores)
    if len(reversals) != 55:
        raise ValueError("Expected 55 screen pairs")
    mean_reversed = reversals.loc[reversals.reversed_in_mean]
    summary["pairwise_reversals_mean_yield_at_least_5pct"] = {
        "pairs": len(reversals),
        "reversed_in_mean": int(len(mean_reversed)),
        "reversed_in_all_seeds": int((mean_reversed.seeds_reversed == len(SEEDS)).sum()),
    }
    ranks = {}
    for protocol in ("row", "group"):
        ranks[protocol] = {}
        for index, seed in enumerate(SEEDS):
            order = sorted(higher, key=lambda s: -scores[s][protocol][index])
            ranks[protocol][str(seed)] = {s: order.index(s) + 1 for s in higher}
        mean_order = sorted(higher, key=lambda s: -scores[s][protocol].mean())
        ranks[protocol]["mean"] = {s: mean_order.index(s) + 1 for s in higher}
    summary["ranks_mean_yield_at_least_5pct"] = ranks
    indexed = table.set_index("screen")
    summary["rank_rho_r2_screen_bootstrap"] = {}
    for name, selected in zip(SUBSETS, (table.screen.tolist(), higher)):
        part = indexed.loc[selected]
        summary["rank_rho_r2_screen_bootstrap"][name] = {
            "rho": rho(part.rf_row_r2, part.rf_group_r2), **bootstrap_rho(part.rf_row_r2, part.rf_group_r2)}
    sweep = []
    for cutoff in range(0, 13):
        part = indexed.loc[indexed.mean_yield_percent >= cutoff]
        sweep.append({"mean_yield_cutoff_percent": cutoff, "screens": int(len(part)),
                      "rho_r2": rho(part.rf_row_r2, part.rf_group_r2) if len(part) >= 3 else None})
    summary["mean_yield_cutoff_sweep"] = sweep
    strata = pd.read_csv(results / "diagnostics" / "rf_substrate_strata_summary.csv")
    seen = strata.loc[strata.protocol.eq("group") & strata.stratum.eq("both_seen")].set_index("screen").row_share_mean
    table["new_combination_both_seen_share"] = table.screen.map(seen)
    indexed = table.set_index("screen")
    summary["two_design_properties"] = {}
    summary["pair_level_decomposition"] = {}
    for name, selected in zip(SUBSETS, (table.screen.tolist(), higher)):
        part = indexed.loc[selected]
        from scipy.stats import rankdata
        rank_sum = rankdata(part.pair_variance_share) + rankdata(part.substrate_levels_per_pair)
        summary["two_design_properties"][name] = {
            "partial_rho_pair_share_given_sparsity": partial_spearman(part.pair_variance_share, part.rf_delta_r2, part.substrate_levels_per_pair),
            "partial_rho_sparsity_given_pair_share": partial_spearman(part.substrate_levels_per_pair, part.rf_delta_r2, part.pair_variance_share),
            "rho_rank_sum_vs_rf_delta_r2": rho(rank_sum, part.rf_delta_r2),
            "rho_sparsity_vs_share_of_test_wells_with_an_unseen_substrate": rho(part.substrate_levels_per_pair, 1 - part.new_combination_both_seen_share),
        }
        summary["pair_level_decomposition"][name] = {
            "median_row_pair_level_r2": float(part.row_pair_level_r2.median()),
            "median_group_pair_level_r2": float(part.group_pair_level_r2.median()),
            "median_row_within_pair_r2": float(part.row_within_pair_r2.median()),
            "median_group_within_pair_r2": float(part.group_within_pair_r2.median()),
            "rho_group_pair_level_r2_vs_rf_delta_r2": rho(part.group_pair_level_r2, part.rf_delta_r2),
        }
    summary["sister_well_mean"] = {}
    for name, selected in zip(SUBSETS, (table.screen.tolist(), higher)):
        part = indexed.loc[selected]
        summary["sister_well_mean"][name] = {
            "median_row_r2": float(part.row_sister_well_mean_r2.median()),
            "screens_at_or_above_rf_row_r2": int((part.row_sister_well_mean_r2 >= part.rf_row_r2).sum()),
            "rho_vs_pair_variance_share": rho(part.row_sister_well_mean_r2, part.pair_variance_share),
            "max_abs_difference_from_pair_variance_share": float((part.row_sister_well_mean_r2 - part.pair_variance_share).abs().max()),
        }
    summary["condition_adjusted_pair_share"] = {}
    for name, selected in zip(SUBSETS, (table.screen.tolist(), higher)):
        part = indexed.loc[selected]
        summary["condition_adjusted_pair_share"][name] = {
            "rho_vs_pair_variance_share": rho(part.condition_adjusted_pair_variance_share, part.pair_variance_share),
            "rho_vs_rf_delta_r2": rho(part.condition_adjusted_pair_variance_share, part.rf_delta_r2),
            "p_vs_rf_delta_r2": float(spearmanr(part.condition_adjusted_pair_variance_share, part.rf_delta_r2).pvalue),
            "max_abs_difference_from_pair_variance_share": float(
                (part.condition_adjusted_pair_variance_share - part.pair_variance_share).abs().max()),
        }
    summary["pairing_design_rho"] = {}
    for name, selected in zip(SUBSETS, (table.screen.tolist(), higher)):
        part = indexed.loc[selected]
        summary["pairing_design_rho"][name] = {
            "substrate_levels_per_pair_vs_rf_delta_r2": rho(part.substrate_levels_per_pair, part.rf_delta_r2),
            "p_substrate_levels_per_pair_vs_rf_delta_r2": float(spearmanr(part.substrate_levels_per_pair, part.rf_delta_r2).pvalue),
            "substrate_levels_per_pair_vs_pair_variance_share": rho(part.substrate_levels_per_pair, part.pair_variance_share),
            "median_partners_range": [float(min(part.median_partners_substrate_1.min(), part.median_partners_substrate_2.min())),
                                      float(max(part.median_partners_substrate_1.max(), part.median_partners_substrate_2.max()))],
            "grid_fill_range": [float(part.grid_fill.min()), float(part.grid_fill.max())]}
    multi = table.loc[table.pairs_on_multiple_plates >= 10]
    summary["plate_structure"] = {
        "max_pairs_per_plate": 1,
        "pairs_on_multiple_plates_total": int(table.pairs_on_multiple_plates.sum()),
        "screens_with_at_least_10_multi_plate_pairs": multi.screen.tolist(),
        "max_within_pair_between_plate_share_of_screen_variance": float(table.within_pair_between_plate_share.max()),
        "multi_plate_pairs_within_pair_between_plate_share_range": [
            float(multi.multi_plate_within_pair_between_plate_share.min()),
            float(multi.multi_plate_within_pair_between_plate_share.max())],
        "multi_plate_pairs_between_pair_share_range": [
            float(multi.multi_plate_between_pair_share.min()), float(multi.multi_plate_between_pair_share.max())],
    }
    return table, reversals, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table, reversals, summary = run(args.package_dir.resolve(), args.results_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "rf_yield_variance_structure.csv", index=False)
    reversals.to_csv(args.output_dir / "rf_pairwise_reversals.csv", index=False)
    (args.output_dir / "rf_ranking_structure_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "ranks_mean_yield_at_least_5pct"},
                     indent=2))


if __name__ == "__main__":
    main()
