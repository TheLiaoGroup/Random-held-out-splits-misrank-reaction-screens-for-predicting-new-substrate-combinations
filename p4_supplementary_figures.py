"""Draw the supplementary figures (provisional numbers S1-S7) from the released results, without fitting.

Run from the package directory:  python p4_supplementary_figures.py --output-dir figures
S1 plate repeatability; S2 omitting an array and changing the yield cutoff; S3 order reversals between pairs of
arrays; S4 absolute-error metrics; S5 the forest's gain over the identity model on new pairs; S6 substrate exposure
of new-pair test wells and errors on wells whose two substrates were seen; S7 the full Grignard and
esterification yield tables. Style and colours follow
p4_chem_figures.py. Numbers are provisional until the supplementary information is ordered by first citation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p4_chem_figures as main_figures
from p4_chem_figures import (GRAY, GREEN, INK, LIGHT, MID, ORANGE, PURPLE, TEAL, WIDTH, despine, marker_style,
                             name, plt, rho, style)
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

SUBSET = "mean_yield_at_least_5pct"
STRATA = ("both_seen", "one_unseen", "both_unseen")
STRATUM_COLOURS = ("#4f5b62", "#9aa5ab", "#d5dadd")
STRATUM_LABELS = ("both seen", "one unseen", "both unseen")
SEEDS = (42, 20261007, 20261008, 20261009, 20261010)


def load(name_):
    path = main_figures.DIAG / name_
    return json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else pd.read_csv(path)


def figure_s1():
    scores = load("rf_model_comparison.csv").set_index("screen")
    bench = json.loads((main_figures.PACKAGE / "data/benchmarks/platform_validation_metadata.json")
                       .read_text(encoding="utf-8"))
    replicate = {entry["reaction_class"]: entry for entry in bench["benchmarks"]}
    mapping = {"Acylation": "acylation", "Suzuki": "suzuki_miyaura", "Buchwald-Hartwig": "buchwald_hartwig"}
    rmse = np.array([scores.loc[n, "rf_group_rmse"] for n in mapping])
    sd = np.array([replicate[c]["sample_sd_yield_percentage_points"] for c in mapping.values()])
    assert np.all(rmse > sd) and np.all(rmse < 100)
    ratio = rmse / sd
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.2)
    style.panel_title(ax, f"New-pair error is {ratio.min():.0f}–{ratio.max():.0f} times plate repeatability")
    x = np.arange(len(mapping))
    ax.bar(x - 0.15, rmse, 0.28, color=ORANGE)
    ax.bar(x + 0.15, sd, 0.28, color="#5b6770")
    for i in x:
        ax.text(i - 0.15, rmse[i] + 0.6, f"{rmse[i]:.1f}", ha="center", va="bottom", color=INK)
        ax.text(i + 0.15, sd[i] + 0.6, f"{sd[i]:.1f}", ha="center", va="bottom", color=INK)
        ax.text(i - 0.15, rmse[i] + 3.2, f"×{ratio[i]:.1f}", ha="center", va="bottom", color=INK, weight="bold")
    ax.text(2.45, 29, "RMSE on new pairs", ha="right", va="top", color=INK)
    ax.text(2.45, 26.5, "SD of a plate repeating\none reaction", ha="right", va="top", color=GRAY)
    ax.set_xticks(x, [name(n) for n in mapping])
    ax.set_ylim(0, 31)
    ax.set_ylabel("Yield error or SD, percentage points")
    despine(ax)
    main_figures.save(fig, "Figure_S1")


def figure_s2():
    diagnostics = load("rf_saved_score_diagnostics.json")
    ranking = load("rf_ranking_structure_summary.json")
    omissions = diagnostics["leave_one_screen_out_r2"]["screen_mean"]["by_omitted_screen"]
    reference = diagnostics["metric_rank_concordance"]["mean_yield_ge_0_05"]["screen_mean"]["r2"]
    fig = plt.figure(figsize=(WIDTH, 3.2))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.45, left=0.17, right=0.985, top=0.86,
                    bottom=0.16)
    ax = fig.add_subplot(grid[0, 0])
    style.panel_title(ax, f"A  No single array carries the 0.{round(100 * reference):02d}")
    for i, entry in enumerate(omissions):
        colour = PURPLE if entry["omitted_screen"] == "Grignard" else MID
        ax.scatter(entry["rho"], i, s=22, color=colour, zorder=3)
        ax.text(0.5, i, f"{entry['rho']:.2f}".replace("-", "−"), ha="right", va="center", color=INK)
    ax.axvline(reference, color=GRAY, ls=(0, (3, 2)), lw=0.7)
    ax.set_yticks(range(len(omissions)), [name(e["omitted_screen"]) for e in omissions])
    ax.set_ylim(len(omissions) - 0.5, -0.8)
    ax.set_xlim(-0.45, 0.52)
    ax.set_xlabel("Rank correlation without that array")
    despine(ax)
    ax = fig.add_subplot(grid[0, 1])
    style.panel_title(ax, "B  It stays near zero from a 5% to an 11% cutoff")
    sweep = [e for e in ranking["mean_yield_cutoff_sweep"] if e["rho_r2"] is not None]
    cut = np.array([e["mean_yield_cutoff_percent"] for e in sweep])
    values = np.array([e["rho_r2"] for e in sweep])
    ax.plot(cut, values, "o-", color=MID, ms=3.5, lw=1.2)
    previous = None
    for c, r, e in zip(cut, values, sweep):
        if e["screens"] != previous:
            ax.text(c, r + 0.04, str(e["screens"]), ha="center", va="bottom", color=GRAY)
        previous = e["screens"]
    ax.axhline(0, color=LIGHT, lw=0.6)
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-0.2, 0.7)
    ax.set_xlabel("Minimum array mean yield, %")
    ax.set_ylabel("Rank correlation between tests")
    despine(ax)
    main_figures.save(fig, "Figure_S2")


def figure_s3():
    ranking = load("rf_ranking_structure_summary.json")
    reversals = load("rf_pairwise_reversals.csv")
    row_rank = ranking["ranks_mean_yield_at_least_5pct"]["row"]["mean"]
    order = sorted(row_rank, key=row_rank.get)
    index = {n: i for i, n in enumerate(order)}
    summary = ranking["pairwise_reversals_mean_yield_at_least_5pct"]
    assert len(reversals) == summary["pairs"]
    colours = {"stable": "#3b454c", "partial": "#9aa5ab", "kept": "#e6eaec"}
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    fig.subplots_adjust(left=0.3, right=0.98, top=0.88, bottom=0.27)
    style.panel_title(ax, f"{summary['reversed_in_all_seeds']} of {summary['pairs']} pairs reverse in every seed")
    for record in reversals.itertuples():
        i, j = index[record.screen_a], index[record.screen_b]
        low, high = max(i, j), min(i, j)
        kind = ("stable" if int(record.seeds_reversed) == 5 else "partial") if str(record.reversed_in_mean) == "True" \
            else "kept"
        ax.add_patch(Rectangle((high, low), 0.94, 0.94, facecolor=colours[kind], edgecolor="none"))
    n = len(order)
    ax.set_xlim(-0.2, n - 0.8)
    ax.set_ylim(n + 0.1, 0.8)
    ax.set_xticks(np.arange(n - 1) + 0.47, [name(s) for s in order[:-1]], rotation=55, ha="right")
    ax.set_yticks(np.arange(1, n) + 0.47, [name(s) for s in order[1:]])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for k, (kind, text) in enumerate([("stable", "reversed in all 5 seeds"),
                                      ("partial", "reversed in 2–4 seeds"),
                                      ("kept", "order kept")]):
        ax.add_patch(Rectangle((5.2, 1.2 + 0.75 * k), 0.5, 0.5, facecolor=colours[kind], edgecolor="none"))
        ax.text(5.9, 1.45 + 0.75 * k, text, va="center", color=INK)
    ax.text(5.2, 4.0, f"{summary['reversed_in_mean']} of {summary['pairs']} reverse on average",
            va="center", color=INK)
    main_figures.save(fig, "Figure_S3")


def median_baseline(screens):
    rows = load("rf_median_baseline_by_seed.csv")
    out = {}
    for screen in screens:
        out[screen] = {}
        for protocol in ("row", "group"):
            sel = rows[(rows.screen == screen) & (rows.protocol == protocol)]
            assert len(sel) == 5, (screen, protocol)
            out[screen][protocol] = {"model": sel.model_mae_percentage_points.mean(),
                                     "median": sel.training_median_mae_percentage_points.mean(),
                                     "gain": sel.model_mae_improvement_over_training_median.mean()}
    return out


def figure_s4():
    table = main_figures.structure()
    high = table[table.high].screen.tolist()
    base = median_baseline(high)
    ranking = load("rf_ranking_structure_summary.json")
    ranks = load("rf_model_comparison.json")["unified_rf"]
    sharp = load("rf_baseline_weighting_summary.json")["subsets"]
    fig = plt.figure(figsize=(WIDTH, 5.6))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 0.9], wspace=0.45, hspace=0.55, left=0.08, right=0.985,
                    top=0.93, bottom=0.08)
    ax = fig.add_subplot(grid[0, 0])
    style.panel_title(ax, "A  The gain over a median collapses in some arrays")
    emphasis = {"Grignard": PURPLE, "Reductive-amination": INK, "Acylation": "#55606a"}
    for screen in high:
        left, right = 100 * base[screen]["row"]["gain"], 100 * base[screen]["group"]["gain"]
        colour = emphasis.get(screen, LIGHT)
        strong = screen in emphasis
        ax.plot((0, 1), (left, right), color=colour, lw=2.0 if strong else 0.9, zorder=3 if strong else 2,
                marker="o", ms=3.2 if strong else 2.2)
        if strong:
            shift = {"Grignard": 4.5, "Reductive-amination": -3.5}.get(screen, 0)
            ax.text(1.06, right + shift, f"{name(screen)} {left:.0f}% → {right:.0f}%", va="center", color=INK)
    ax.set_xticks((0, 1), ("Held-out wells", "New pairs"))
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(-2, 70)
    ax.set_ylabel("MAE gain over training median, %")
    despine(ax)
    ax.spines["bottom"].set_bounds(0, 1)
    ax = fig.add_subplot(grid[0, 1])
    style.panel_title(ax, "B  Raw error follows the yield spread")
    model = np.array([base[s]["group"]["model"] for s in high])
    reference = np.array([base[s]["group"]["median"] for s in high])
    ax.plot((5, 30), (5, 30), color=GRAY, ls=(0, (3, 2)), lw=0.7)
    for screen, x, y in zip(high, reference, model):
        ax.scatter(x, y, s=26, color=emphasis.get(screen, MID), edgecolor="white", linewidths=0.5, zorder=3)
    for screen, (dx, dy, ha) in {"Grignard": (-6, 4, "right"), "Reductive-amination": (6, -12, "left"),
                                 "Acylation": (6, -4, "left")}.items():
        i = high.index(screen)
        ax.annotate(name(screen), (reference[i], model[i]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    color=INK)
    ax.text(6, 29, f"rank correlation {ranking['raw_mae_vs_median_mae_rho'][SUBSET]['group']:.2f}\n"
                   "points on the line gain nothing", ha="left", va="top", color=INK)
    ax.set_xlim(5, 30)
    ax.set_ylim(5, 30)
    ax.set_xlabel("Training-median MAE on new pairs, pp")
    ax.set_ylabel("Model MAE on new pairs, pp")
    despine(ax)
    ax = fig.add_subplot(grid[1, :])
    style.panel_title(ax, "C  How much of the order survives depends on the metric")
    labels = ("R²", "RMSE", "MAE", "MAE gain over median")
    groups = {f"all {len(table)} arrays": [ranks["all_screens"]["metric_rho"]["r2"],
                                         ranks["all_screens"]["metric_rho"]["rmse"],
                                         sharp["all_18"]["baseline_model_mae_percentage_points_rho"],
                                         sharp["all_18"]["baseline_model_mae_improvement_over_training_median_rho"]],
              f"{len(high)} arrays averaging ≥5% yield": [ranks[SUBSET]["metric_rho"]["r2"],
                                                         ranks[SUBSET]["metric_rho"]["rmse"],
                                                         sharp[SUBSET]["baseline_model_mae_percentage_points_rho"],
                                                         sharp[SUBSET][
                                                             "baseline_model_mae_improvement_over_training_median_rho"]]}
    x = np.arange(len(labels))
    for k, ((group, values), colour) in enumerate(zip(groups.items(), (LIGHT, "#55606a"))):
        offset = -0.1 if k == 0 else 0.1
        ax.bar(x + offset, values, 0.18, color=colour)
        for xi, v in zip(x + offset, values):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", color=INK)
        ax.add_patch(Rectangle((-0.42, 1.0 - 0.12 * k), 0.07, 0.06, color=colour, transform=ax.transData))
        ax.text(-0.3, 1.03 - 0.12 * k, group, va="center", color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rank correlation between tests")
    despine(ax)
    main_figures.save(fig, "Figure_S4")


def figure_s5():
    table = main_figures.structure()
    additive = load("additive_baseline_summary.json")
    per = additive["per_screen"]
    table["gain"] = table.screen.map(lambda s: 100 * per[s]["group"]["rf_skill_over_additive"])
    table["pair_pct"] = 100 * table.pair_variance_share
    high = table[table.high]
    value = additive["subsets"][SUBSET]["rho_pair_share_vs_rf_skill_over_additive_new_combinations"]
    assert abs(value - rho(high.pair_variance_share, high.gain)) < 1e-9
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.85, bottom=0.17)
    style.panel_title(ax, "Structure adds least where the pair sets the yield")
    for row in table.itertuples():
        ax.scatter(row.pair_pct, row.gain, **marker_style(row))
    for screen, (dx, dy, ha) in {"Grignard": (-6, 6, "right"), "Esterification": (6, 0, "left"),
                                 "Acylation": (6, 0, "left")}.items():
        row = table.set_index("screen").loc[screen]
        ax.annotate(name(screen), (row.pair_pct, row.gain), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", color=INK)
    ax.axhline(0, color=LIGHT, lw=0.6)
    ax.text(0.97, 0.02, f"rank correlation {value:.2f} (11 arrays)".replace("-", "−"), transform=ax.transAxes,
            ha="right", va="bottom", color=INK)
    ax.set_xlim(18, 75)
    ax.set_ylim(-15, 35)
    ax.set_xlabel("Yield variance between pairs, %")
    ax.set_ylabel("Forest's gain over identity\nmodel on new pairs, %")
    despine(ax)
    main_figures.save(fig, "Figure_S5")


def figure_s6():
    table = main_figures.structure()
    order = table.sort_values("pair_variance_share", ascending=False).screen.tolist()
    low = set(table.loc[~table.high, "screen"])
    strata = load("rf_substrate_strata_summary.csv")
    strata = strata[strata.protocol == "group"]
    assert len(strata) == 3 * len(order)
    cohorts = load("rf_matched_cohorts_by_seed.csv")
    assert len(cohorts) == len(order) * len(SEEDS) * len(STRATA)
    seen = cohorts[(cohorts.group_exposure_stratum == "both_seen") & (cohorts["rows"] > 0)]
    fig = plt.figure(figsize=(WIDTH, 6.4))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.08, left=0.2, right=0.985, top=0.9,
                    bottom=0.1)
    y = np.arange(len(order))
    ax = fig.add_subplot(grid[0, 0])
    style.panel_title(ax, "A  Most new-pair wells pair two seen substrates")
    left = np.zeros(len(order))
    for stratum, colour, text in zip(STRATA, STRATUM_COLOURS, STRATUM_LABELS):
        shares = strata[strata.stratum == stratum].set_index("screen").loc[order, "row_share_mean"].to_numpy() * 100
        ax.barh(y, shares, left=left, color=colour, height=0.45)
        left += shares
    for k, (colour, text) in enumerate(zip(STRATUM_COLOURS, STRATUM_LABELS)):
        ax.add_patch(Rectangle((2 + 33 * k, -1.55), 4, 0.5, color=colour, clip_on=False))
        ax.text(7.5 + 33 * k, -1.3, text, va="center", color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("New-pair test wells, %")
    ax.set_yticks(y, [name(s) for s in order])
    for tick, screen in zip(ax.get_yticklabels(), order):
        if screen in low:
            tick.set_color(GRAY)
            tick.set_style("italic")
    ax.set_ylim(len(order) - 0.5, -2.0)
    despine(ax, keep=("bottom",))
    ax = fig.add_subplot(grid[0, 1])
    style.panel_title(ax, "B  On those wells error still rises")
    for i, screen in enumerate(order):
        sel = seen[seen.screen == screen]
        for kind, colour, dy in (("row", TEAL, -0.12), ("group", ORANGE, 0.12)):
            values = sel[f"{kind}_mae"].to_numpy() * 100
            if len(values):
                ax.plot([values.min(), values.max()], [i + dy] * 2, color=colour, lw=0.8)
                ax.scatter(values.mean(), i + dy, s=14, color=colour, zorder=3)
    ax.text(30, -1.0, "teal, held-out wells"+chr(10)+"orange, new pairs (line: seed range)", ha="right", va="center",
            color=INK)
    ax.set_yticks(y, [""] * len(order))
    ax.set_ylim(len(order) - 0.5, -2.0)
    ax.set_xlim(0, 30)
    ax.set_xlabel("MAE on the same wells, pp")
    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    main_figures.save(fig, "Figure_S6")


def figure_s7():
    """Full Grignard and esterification yield tables, from which the Figure 4 blocks are drawn."""
    data = main_figures.records(["Grignard", "Esterification"])
    grignard, g_flags, _ = main_figures.matrix(data[data.screen == "Grignard"])
    ester, e_flags, _ = main_figures.matrix(data[data.screen == "Esterification"])
    assert grignard.shape == (136, 47) and ester.shape == (80, 95)
    table = main_figures.structure().set_index("screen")

    fig = plt.figure(figsize=(WIDTH, 4.6))
    grid = GridSpec(1, 2, figure=fig, wspace=0.28, left=0.07, right=0.985, top=0.9, bottom=0.2)
    for k, (title, frame, flags, screen) in enumerate((
            ("A  Grignard addition", grignard, g_flags, "Grignard"),
            ("B  Esterification", ester, e_flags, "Esterification"))):
        ax = fig.add_subplot(grid[0, k])
        style.panel_title(ax, title)
        main_figures.heatmap(ax, frame, flags)
        row = table.loc[screen]
        ax.set_ylabel(f"{frame.shape[0]} substrate pairs, by pair mean", labelpad=2)
        ax.set_xlabel(f"{frame.shape[1]} conditions, by condition mean" + chr(10)
                      + f"between pairs {100 * row.pair_variance_share:.1f}%; "
                        f"between conditions {100 * row.condition_variance_share:.1f}%", labelpad=3)
    cax = fig.add_axes([0.07, 0.05, 0.2, 0.025])
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=main_figures.CMAP, norm=plt.Normalize(0, 100)), cax=cax,
                       orientation="horizontal")
    bar.set_ticks([0, 50, 100])
    bar.outline.set_linewidth(0.5)
    cax.tick_params(labelsize=8, length=2, pad=1)
    cax.text(1.05, 0.5, "yield, %", transform=cax.transAxes, ha="left", va="center", color=INK)
    key = fig.add_axes([0.45, 0.03, 0.5, 0.06])
    key.axis("off")
    for x, (fc, ec, hatch, label) in zip((0.0, 0.5), ((main_figures.MISSING, main_figures.MISSING_HATCH, "//////",
                                                      "no numerical yield"), ("white", GRAY, None, "no record"))):
        key.add_patch(Rectangle((x, 0.2), 0.04, 0.6, fc=fc, ec=ec, lw=0.5 if hatch is None else 0, hatch=hatch,
                                transform=key.transAxes, clip_on=False))
        key.text(x + 0.06, 0.5, label, va="center", transform=key.transAxes, color=INK)
    main_figures.save(fig, "Figure_S7")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package-dir", type=Path, default=main_figures.PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    main_figures.PACKAGE = args.package_dir.resolve()
    main_figures.OUTPUT = main_figures.PACKAGE / "output"
    main_figures.DIAG = main_figures.OUTPUT / "p4_unified_rf" / "diagnostics"
    main_figures.OUT = (args.output_dir or main_figures.PACKAGE / "figures").resolve()
    main_figures.OUT.mkdir(parents=True, exist_ok=True)
    for draw in (figure_s1, figure_s2, figure_s3, figure_s4, figure_s5, figure_s6, figure_s7):
        draw()
    print(f"Saved supplementary Figures S1-S7 in {main_figures.OUT}")


if __name__ == "__main__":
    main()
