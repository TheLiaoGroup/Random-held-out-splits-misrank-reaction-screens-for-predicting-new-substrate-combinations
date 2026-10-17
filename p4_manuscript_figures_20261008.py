"""Draw P4 main Figures 1-3 and 5 and Extended Data Figures 1-3 from saved results without fitting.

Figure 3 is the additive identity reference and Figure 5 the error-scale comparison.

Figure 4 is drawn by p4_rf_figure.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

from p4_figure_style import GRAY, INK, PNG_DPI, RC, finalize, panel_title, use_style


ROOT = Path(__file__).resolve().parent
UNIFIED_RESULTS = ROOT / "output" / "p4_unified_rf" / "diagnostics"
OUTPUT = ROOT / "figures"
BENCHMARKS = ROOT / "data/benchmarks/platform_validation_metadata.json"
ROW_COLOR = "#377e86"
GROUP_COLOR = "#d08a3c"
RISE_COLOR = "#5a4f9c"
LIGHT = "#bcc4c8"
PALE = "#e7ecef"
POINT = "#4f5b62"
SUBSET = "mean_yield_at_least_5pct"
ALIASES = {"C-O competitive reactions": "C-O competitive",
           "Reductive-amination": "Reductive amination",
           "Allylic-substitution": "Allylic substitution"}
HIGHLIGHT = {"Grignard": GROUP_COLOR, "Buchwald-Hartwig": RISE_COLOR, "Acylation": INK}


def label(name):
    return ALIASES.get(name, name)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#879197")
    ax.tick_params(colors=INK, length=3)
    ax.set_axisbelow(True)


def save(fig, name):
    finalize(fig)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=PNG_DPI, facecolor="white")
    fig.savefig(OUTPUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def background_panel(ax):
    """Prior evaluation studies at their stated scope, contrasted with this comparison."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "a  Earlier tests and this one")
    ax.text(0.02, 0.95, "Study", weight="bold", color=GRAY, va="center")
    ax.text(0.30, 0.95, "Withheld from training; scope; endpoint", weight="bold",
            color=GRAY, va="center")
    ax.plot((0.02, 0.99), (0.90, 0.90), color=LIGHT, linewidth=0.6)
    studies = (
        ("Sandfort 2020", "a molecular component; yield"),
        ("Ismail 2026", "components or pairs; four HTE datasets; yield"),
        ("Götz 2025", "reactant dimensions; one three-component\nreaction; reaction outcome"),
        ("Neves 2026", "data sources; Buchwald-Hartwig; success class"),
    )
    y = 0.82
    for name, scope in studies:
        ax.text(0.02, y, name, color=INK, va="top")
        ax.text(0.30, y, scope, color=INK, va="top", linespacing=1.15)
        y -= 0.20 if "\n" in scope else 0.125
    ax.add_patch(Rectangle((0.0, y - 0.135), 1.04, 0.16, facecolor=PALE, edgecolor="none"))
    ax.text(0.02, y, "This work", weight="bold", color=INK, va="top")
    ax.text(0.30, y, "whole pairs; 18 screens in 17 families;\nscreen order compared between tasks", weight="bold", color=INK, va="top", linespacing=1.15)


def task_matrix(ax, x0, y0, grouped):
    """Rows are substrate pairs, columns are conditions; cells are explanatory."""
    rows, columns, cell = 4, 7, 0.052
    for row in range(rows):
        for column in range(columns):
            if grouped:
                test = row == 1
            else:
                test = (row * 3 + column * 2) % 5 == 0
            face = (GROUP_COLOR if grouped else ROW_COLOR) if test else PALE
            ax.add_patch(Rectangle((x0 + column * cell, y0 + (rows - 1 - row) * cell),
                                   cell - 0.006, cell - 0.006, facecolor=face, edgecolor="none"))
    ax.text(x0 - 0.012, y0 + rows * cell / 2, "substrate\npairs", rotation=90, ha="right",
            va="center", color=GRAY, linespacing=1.0)
    ax.text(x0 + columns * cell / 2, y0 - 0.02, "conditions", ha="center", va="top", color=GRAY)


def question_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "b  Two tasks on the same records")
    task_matrix(ax, 0.09, 0.50, grouped=False)
    task_matrix(ax, 0.58, 0.50, grouped=True)
    ax.text(0.27, 0.93, "Held-out wells", ha="center", weight="bold", color=INK)
    ax.text(0.27, 0.86, "other wells of the test pair\nremain in training", ha="center", va="top", color=INK, linespacing=1.15)
    ax.text(0.76, 0.93, "New combinations", ha="center", weight="bold",
            color=INK)
    ax.text(0.76, 0.86, "the whole pair and its plates\nare withheld", ha="center", va="top", color=INK, linespacing=1.15)
    ax.text(0.5, 0.28,
            "340,178 yields in 18 screens and 17 reaction families\n"
            "One random-forest specification; five split seeds\n"
            "Each plate holds a single substrate pair",
            ha="center", va="top", color=INK, linespacing=1.45)


def rank_slope(ax, scores, ranks, rho11, rho18):
    higher = [s for s in scores if s["mean_yield_percent"] >= 5]
    row_rank = ranks["row"]["mean"]
    group_rank = ranks["group"]["mean"]
    if len(higher) != 11 or set(row_rank) != {s["source_name"] for s in higher}:
        raise ValueError("Unexpected 11-screen ranking population")
    for score in higher:
        name = score["source_name"]
        color = HIGHLIGHT.get(name, LIGHT)
        emphasized = name in HIGHLIGHT
        left, right = row_rank[name], group_rank[name]
        ax.plot((0, 1), (left, right), color=color, linewidth=2.2 if emphasized else 0.9,
                zorder=3 if emphasized else 2)
        ax.scatter((0, 1), (left, right), color=color, s=22 if emphasized else 12, zorder=4)
        for x, rank, align in ((-0.07, left, "right"), (1.07, right, "left")):
            ax.text(x, rank, f"{rank}  {label(name)}" if align == "left" else f"{label(name)}  {rank}",
                    ha=align, va="center",
                    color=INK, weight="bold" if emphasized else "normal")
    ax.text(0, 0.15, "Held-out\nwells", ha="center", va="bottom", color=INK,
            weight="bold", linespacing=1.0)
    ax.text(1, 0.15, "New\ncombinations", ha="center", va="bottom",
            color=INK, weight="bold", linespacing=1.0)
    ax.text(0.5, 12.05, f"$R$² rank correlation {rho11:.2f} (11 screens)",
            ha="center", va="top", color=INK)
    ax.set_xlim(-1.05, 2.05)
    ax.set_ylim(12.3, -0.9)
    ax.axis("off")
    panel_title(ax, "c  The order of screens does not survive")


def variance_scatter(ax, structure, rho11):
    for record in structure:
        x = 100 * record["pair_variance_share"]
        y = record["rf_delta_r2"]
        name = record["screen"]
        if record["mean_yield_percent"] < 5:
            ax.scatter(x, y, s=34, facecolors="white", edgecolors=GRAY, linewidth=1.1, zorder=3)
            continue
        color = HIGHLIGHT.get(name, POINT)
        ax.scatter(x, y, s=48 if name in HIGHLIGHT else 36, color=color, edgecolors="white",
                   linewidth=0.6, zorder=4)
    notes = {"Grignard": (-6, 6, "right"), "Buchwald-Hartwig": (-5, -10, "right"),
             "Acylation": (6, 4, "left"), "Esterification": (6, 4, "left"),
             "Suzuki Bpin": (6, 3, "left"), "Heck": (6, -9, "left")}
    lookup = {record["screen"]: record for record in structure}
    for name, (dx, dy, align) in notes.items():
        record = lookup[name]
        ax.annotate(label(name), (100 * record["pair_variance_share"], record["rf_delta_r2"]),
                    xytext=(dx, dy), textcoords="offset points", ha=align,
                    color=INK, weight="bold" if name in HIGHLIGHT else "normal")
    ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho11:.2f}\n(11 screens with mean yield $\\geq$5%)",
            transform=ax.transAxes, va="top", color=INK, linespacing=1.35)
    ax.set_xlim(12, 75)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Yield variance between substrate pairs (%)")
    ax.set_ylabel("$R$² lost on new combinations")
    style_axes(ax)
    handles = [Line2D([], [], marker="o", linestyle="none", color=POINT, markersize=5,
                      label="Mean yield $\\geq$5%"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                      markeredgecolor=GRAY, markersize=5, label="Mean yield <5%")]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    panel_title(ax, "d  Pair-level variance tracks the $R$² lost")


def figure1(rf_scores, ranking, structure, rf_ranks):
    fig = plt.figure(figsize=(7.0, 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.70, 1.18), width_ratios=(1, 1),
                            left=0.05, right=0.975, top=0.955, bottom=0.08,
                            hspace=0.20, wspace=0.30)
    background_panel(fig.add_subplot(grid[0, 0]))
    question_panel(fig.add_subplot(grid[0, 1]))
    rho11 = rf_ranks["unified_rf"][SUBSET]["metric_rho"]["r2"]
    rho18 = rf_ranks["unified_rf"]["all_screens"]["metric_rho"]["r2"]
    rank_slope(fig.add_subplot(grid[1, 0]), rf_scores, ranking["ranks_mean_yield_at_least_5pct"],
               rho11, rho18)
    variance_scatter(fig.add_subplot(grid[1, 1]), structure,
                     ranking["variance_structure_rho"][SUBSET]["pair_variance_share_vs_rf_delta_r2"])
    save(fig, "Figure_1")


def screen_scatter(ax, rf_scores, rf_ranks):
    for score in rf_scores:
        name = score["source_name"]
        x, y = float(score["rf_row_r2"]), float(score["rf_group_r2"])
        if score["mean_yield_percent"] < 5:
            ax.scatter(x, y, s=46, facecolors="white", edgecolors=GRAY, linewidth=1.3, zorder=3)
        else:
            ax.scatter(x, y, s=60 if name in HIGHLIGHT else 46, color=HIGHLIGHT.get(name, POINT),
                       edgecolors="white", linewidth=0.6, zorder=4)
    offsets = {"Acylation": (-8, 8, "right"), "Grignard": (8, -3, "left"),
               "Buchwald-Hartwig": (-8, 6, "right")}
    lookup = {score["source_name"]: score for score in rf_scores}
    for name, (dx, dy, align) in offsets.items():
        score = lookup[name]
        ax.annotate(label(name), (float(score["rf_row_r2"]), float(score["rf_group_r2"])),
                    xytext=(dx, dy), textcoords="offset points", ha=align,
                    color=INK, weight="bold")
    limits = (-0.30, 0.85)
    ax.plot(limits, limits, color=GRAY, linewidth=1, linestyle=(0, (4, 4)))
    ax.axhline(0, color="#c9ced1", linewidth=0.8)
    delta = np.array([float(score["rf_delta_r2"]) for score in rf_scores])
    rho11 = rf_ranks["unified_rf"][SUBSET]["metric_rho"]["r2"]
    rho18 = rf_ranks["unified_rf"]["all_screens"]["metric_rho"]["r2"]
    ax.text(0.03, 0.97,
            f"{int((delta > 0).sum())} of 18 below the equality line\n"
            f"Median $R$² lost: {np.median(delta):.3f}\n"
            f"Rank correlation (11 screens): {rho11:.2f}",
            transform=ax.transAxes, color=INK, va="top", linespacing=1.4)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("Held-out-well $R$²")
    ax.set_ylabel("New-combination $R$²")
    style_axes(ax)
    handles = [Line2D([], [], marker="o", linestyle="none", color=POINT, markersize=6,
                      label="Mean yield $\\geq$5%"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                      markeredgecolor=GRAY, markersize=6, label="Mean yield <5%")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2, frameon=False)
    panel_title(ax, "a  Every screen loses, by very different amounts")


def reversal_matrix(ax, reversals, order, summary):
    index = {name: position for position, name in enumerate(order)}
    n = len(order)
    colors = {"stable": "#3b454c", "partial": "#9aa5ab", "kept": "#e6eaec"}
    for record in reversals:
        i, j = index[record["screen_a"]], index[record["screen_b"]]
        low, high = max(i, j), min(i, j)
        if record["reversed_in_mean"] == "True":
            kind = "stable" if int(record["seeds_reversed"]) == 5 else "partial"
        else:
            kind = "kept"
        ax.add_patch(Rectangle((high, low), 0.94, 0.94, facecolor=colors[kind], edgecolor="none"))
    ax.set_xlim(-0.2, n - 1 + 0.2)
    ax.set_ylim(n + 0.1, 0.8)
    ax.set_xticks(np.arange(n - 1) + 0.47, [label(name) for name in order[:-1]], rotation=55,
                  ha="right")
    ax.set_yticks(np.arange(1, n) + 0.47, [label(name) for name in order[1:]])
    ax.tick_params(length=0, colors=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors["stable"], label="Reversed in all five seeds"),
               Rectangle((0, 0), 1, 1, facecolor=colors["partial"], label="Reversed on average (2-4 seeds)"),
               Rectangle((0, 0), 1, 1, facecolor=colors["kept"], label="Order kept on average")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, handlelength=1.2)
    ax.text(0.98, 0.70,
            f"{summary['reversed_in_mean']} of {summary['pairs']} screen pairs reverse;\n"
            f"{summary['reversed_in_all_seeds']} reverse in every seed",
            transform=ax.transAxes, ha="right", va="top", color=INK, linespacing=1.35)
    panel_title(ax, "b  Most reversals recur in every seed")


def figure2(rf_scores, rf_ranks, reversals, ranking):
    fig, (ax, matrix_ax) = plt.subplots(1, 2, figsize=(7.0, 3.7),
                                        gridspec_kw={"width_ratios": (1, 1.08)},
                                        layout="constrained")
    screen_scatter(ax, rf_scores, rf_ranks)
    row_rank = ranking["ranks_mean_yield_at_least_5pct"]["row"]["mean"]
    order = sorted(row_rank, key=row_rank.get)
    reversal_matrix(matrix_ax, reversals, order, ranking["pairwise_reversals_mean_yield_at_least_5pct"])
    save(fig, "Figure_2")


def figure5_error_scales(baseline_means, rf_ranks, sharp, ranking):
    """Raw MAE rankings follow yield spread; the model's gain over a median reorders."""
    fig = plt.figure(figsize=(7.0, 5.9))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 0.9), width_ratios=(1, 1),
                            left=0.08, right=0.985, top=0.95, bottom=0.08,
                            hspace=0.42, wspace=0.30)
    higher = sorted(baseline_means)
    if len(higher) != 11:
        raise ValueError("Expected 11 screens in the median-baseline panel")

    ax = fig.add_subplot(grid[0, 0])
    emphasis = {"Grignard": GROUP_COLOR, "Reductive-amination": RISE_COLOR, "Acylation": INK}
    label_y = {}
    for name in higher:
        left = 100 * baseline_means[name]["row"]["improvement"]
        right = 100 * baseline_means[name]["group"]["improvement"]
        color = emphasis.get(name, LIGHT)
        strong = name in emphasis
        ax.plot((0, 1), (left, right), color=color, linewidth=2.0 if strong else 0.9,
                zorder=3 if strong else 2)
        ax.scatter((0, 1), (left, right), color=color, s=18 if strong else 10, zorder=4)
        label_y[name] = right
    # Spread right-hand labels so they do not overlap.
    ordered = sorted(higher, key=lambda n: -label_y[n])
    positions = [label_y[n] for n in ordered]
    gap = 3.4
    for i in range(1, len(positions)):
        positions[i] = min(positions[i], positions[i - 1] - gap)
    positions[-1] = max(positions[-1], 1.0)
    for i in range(len(positions) - 2, -1, -1):
        positions[i] = max(positions[i], positions[i + 1] + gap)
    for name, y in zip(ordered, positions):
        ax.plot((1.01, 1.05), (label_y[name], y), color=LIGHT, linewidth=0.5)
        ax.text(1.06, y, f"{label(name)} {label_y[name]:.0f}%", va="center",
                color=INK, weight="bold" if name in emphasis else "normal")
    for name in emphasis:
        ax.text(-0.06, 100 * baseline_means[name]["row"]["improvement"],
                f"{100 * baseline_means[name]['row']['improvement']:.0f}%", ha="right",
                va="center", color=INK, weight="bold")
    ax.set_xlim(-0.30, 1.95)
    ax.set_ylim(-2, 70)
    ax.set_xticks((0, 1), ("Held-out\nwells", "New\ncombinations"))
    ax.set_ylabel("MAE gain over training median (%)")
    ax.spines[["top", "right"]].set_visible(False)
    style_axes(ax)
    panel_title(ax, "a  The model's gain collapses in some screens")

    ax = fig.add_subplot(grid[0, 1])
    model = np.array([baseline_means[n]["group"]["model_mae"] for n in higher])
    reference = np.array([baseline_means[n]["group"]["median_mae"] for n in higher])
    ax.plot((5, 30), (5, 30), color=GRAY, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(26.5, 28.2, "no gain", color=GRAY, rotation=36)
    for name, x, y in zip(higher, reference, model):
        color = emphasis.get(name, POINT)
        ax.scatter(x, y, s=40, color=color, edgecolors="white", linewidth=0.6, zorder=3)
    for name, (dx, dy, align) in {"Grignard": (-6, 4, "right"), "Reductive-amination": (6, -2, "left"),
                                   "Acylation": (6, -4, "left"), "Suzuki Bpin": (-6, 4, "right")}.items():
        position = higher.index(name)
        ax.annotate(label(name), (reference[position], model[position]), xytext=(dx, dy),
                    textcoords="offset points", ha=align,
                    color=INK, weight="bold" if name in emphasis else "normal")
    rho_group = ranking["raw_mae_vs_median_mae_rho"][SUBSET]["group"]
    ax.text(0.04, 0.96, f"Rank correlation of model MAE\nwith median MAE: {rho_group:.2f}",
            transform=ax.transAxes, va="top", color=INK, linespacing=1.35)
    ax.set_xlim(5, 30)
    ax.set_ylim(5, 30)
    ax.set_xlabel("Training-median MAE (pp)")
    ax.set_ylabel("Model MAE on new combinations (pp)")
    style_axes(ax)
    panel_title(ax, "b  Raw MAE tracks yield spread")

    ax = fig.add_subplot(grid[1, :])
    labels = ("$R$²", "RMSE", "MAE", "MAE gain over median")
    all18 = sharp["subsets"]["all_18"]
    sub11 = sharp["subsets"][SUBSET]
    values = {
        "All 18 screens": [rf_ranks["unified_rf"]["all_screens"]["metric_rho"]["r2"],
                           rf_ranks["unified_rf"]["all_screens"]["metric_rho"]["rmse"],
                           all18["baseline_model_mae_percentage_points_rho"],
                           all18["baseline_model_mae_improvement_over_training_median_rho"]],
        "11 screens, mean yield $\\geq$5%": [rf_ranks["unified_rf"][SUBSET]["metric_rho"]["r2"],
                                            rf_ranks["unified_rf"][SUBSET]["metric_rho"]["rmse"],
                                            sub11["baseline_model_mae_percentage_points_rho"],
                                            sub11["baseline_model_mae_improvement_over_training_median_rho"]],
    }
    x = np.arange(len(labels))
    for offset, (name, vals), color in zip((-0.11, 0.11), values.items(), (LIGHT, GROUP_COLOR)):
        bars = ax.bar(x + offset, vals, 0.20, color=color, label=name)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}",
                    ha="center", color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rank correlation between tasks")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    style_axes(ax)
    panel_title(ax, "c  How much of the screen order survives depends on the metric")
    save(fig, "Figure_5")


def extended_data_figure1(rf_scores, benchmarks):
    """New-combination RMSE against the SD of a plate repeating one reaction."""
    mapping = {"Acylation": "acylation", "Suzuki": "suzuki_miyaura", "Buchwald-Hartwig": "buchwald_hartwig"}
    names = list(mapping)
    lookup = {score["source_name"]: score for score in rf_scores}
    replicate = {entry["reaction_class"]: entry for entry in benchmarks["benchmarks"]}
    rmse = np.array([float(lookup[name]["rf_group_rmse"]) for name in names])
    sd = np.array([replicate[mapping[name]]["sample_sd_yield_percentage_points"] for name in names])
    ratios = rmse / sd
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.5, 3.4), layout="constrained")
    width = 0.26
    ax.bar(x - width / 2, rmse, width, color=GROUP_COLOR, label="RMSE on new combinations")
    ax.bar(x + width / 2, sd, width, color="#5b6770", label="SD of a plate repeating one reaction")
    for position in range(len(names)):
        ax.text(position - width / 2, rmse[position] + 0.4, f"{rmse[position]:.1f}", ha="center", color=INK)
        ax.text(position + width / 2, sd[position] + 0.4, f"{sd[position]:.2f}", ha="center", color=INK)
        ax.text(position - width / 2, rmse[position] + 2.6, f"{ratios[position]:.1f}\u00d7", ha="center", weight="bold", color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 31)
    ax.set_ylabel("Yield error or SD (percentage points)")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    panel_title(ax, "New-combination error is three to five\ntimes the repeatability of a plate")
    save(fig, "Extended_Data_Figure_1")


def extended_data_figure2(diagnostics, ranking):
    """Dependence of the rank correlation on which screens are compared."""
    fig, (ax, sweep_ax) = plt.subplots(1, 2, figsize=(7.0, 3.4), layout="constrained",
                                       gridspec_kw={"width_ratios": (1.05, 1)})
    omissions = diagnostics["leave_one_screen_out_r2"]["screen_mean"]["by_omitted_screen"]
    reference = diagnostics["metric_rank_concordance"]["mean_yield_ge_0_05"]["screen_mean"]["r2"]
    y = np.arange(len(omissions))
    ax.axvline(reference, color=GRAY, linewidth=1, linestyle=(0, (4, 4)))
    for position, entry in enumerate(omissions):
        color = GROUP_COLOR if entry["omitted_screen"] == "Grignard" else POINT
        ax.scatter(entry["rho"], position, color=color, s=26, zorder=3)
        ax.text(0.47, position, f"{entry['rho']:.2f}".replace("-", "−"), va="center", ha="right", color=INK)
    ax.set_yticks(y, [label(entry["omitted_screen"]) for entry in omissions])
    ax.set_ylim(len(omissions) - 0.5, -1.1)
    ax.set_xlim(-0.45, 0.50)
    ax.set_xlabel("$R$² rank correlation between tasks")
    ax.text(reference + 0.02, -0.75, f"all 11: {reference:.2f}", color=GRAY)
    style_axes(ax)
    panel_title(ax, "a  Omitting any one screen")
    sweep = [entry for entry in ranking["mean_yield_cutoff_sweep"] if entry["rho_r2"] is not None]
    cut = np.array([entry["mean_yield_cutoff_percent"] for entry in sweep])
    rho = np.array([entry["rho_r2"] for entry in sweep])
    sweep_ax.plot(cut, rho, "o-", color=POINT, markersize=4, linewidth=1.3)
    previous = None
    for c, r, entry in zip(cut, rho, sweep):
        if entry["screens"] != previous:  # label only where the number of retained screens changes
            dx = 0.35 if c in (0, 4, 5) else 0.0
            sweep_ax.text(c + dx, r + (0.03 if c < 5 else -0.055), str(entry["screens"]), ha="left" if dx else "center", color=GRAY)
        previous = entry["screens"]
    sweep_ax.axhline(0, color="#c9ced1", linewidth=0.8)
    sweep_ax.set_xlim(-0.6, 12.6)
    sweep_ax.set_ylim(-0.2, 0.7)
    sweep_ax.set_xlabel("Minimum screen mean yield (%)")
    sweep_ax.set_ylabel("$R$² rank correlation between tasks")
    style_axes(sweep_ax)
    panel_title(sweep_ax, "b  Changing the mean-yield cutoff")
    save(fig, "Extended_Data_Figure_2")


def extended_data_figure3(diagnostics):
    """Complete retrospective top-k selection curve."""
    selection = diagnostics["retrospective_selection"]
    curve = selection["screen_mean"]["by_k"]
    ranges = selection["seed_ranges_by_k"]
    k = np.array([entry["k"] for entry in curve])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), layout="constrained")
    ax, overlap_ax = axes
    keys = (
        ("group_r2_mean_row_selected", ROW_COLOR, "Chosen by held-out-well $R$²"),
        ("group_r2_mean_group_selected", GROUP_COLOR, "Chosen by new-combination $R$²"),
    )
    for key, color, name in keys:
        values = [entry[key] for entry in curve]
        ax.fill_between(k, [entry[key]["min"] for entry in ranges], [entry[key]["max"] for entry in ranges],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(k, values, "o-", color=color, label=name, markersize=3.5, linewidth=1.4)
    ax.set_xlim(0.6, 11.4)
    ax.set_ylim(0.20, 0.74)
    ax.set_ylabel("Mean new-combination $R$²")
    ax.set_xlabel("Screens chosen ($k$)")
    ax.legend(loc="upper right", frameon=False)
    panel_title(ax, "a  Held-out-well choices do worse on new pairs")
    overlap = [entry["overlap_count"] for entry in curve]
    overlap_ax.fill_between(k, [entry["overlap_count"]["min"] for entry in ranges],
                            [entry["overlap_count"]["max"] for entry in ranges],
                            color=ROW_COLOR, alpha=0.12, linewidth=0)
    overlap_ax.plot(k, k, color=GRAY, linewidth=1, linestyle=(0, (4, 4)), label="Identical sets")
    overlap_ax.plot(k, overlap, "o-", color=ROW_COLOR, markersize=3.5, linewidth=1.4, label="Shared screens")
    overlap_ax.set_xlim(0.6, 11.4)
    overlap_ax.set_ylim(0, 11.8)
    overlap_ax.set_xlabel("Screens chosen ($k$)")
    overlap_ax.set_ylabel("Shared screens")
    overlap_ax.legend(loc="upper left", frameon=False)
    panel_title(overlap_ax, "b  One shared screen up to $k$ = 4")
    for axis in axes:
        style_axes(axis)
    save(fig, "Extended_Data_Figure_3")


def figure3_identity_reference(additive, structure, add_summary, decomposition_medians):
    """Additive substrate-plus-condition reference against the random forest."""
    fig, grid_axes = plt.subplots(2, 2, figsize=(7.0, 6.0), layout="constrained")
    (ax, skill_ax), (level_ax, note_ax) = grid_axes
    shares = {record["screen"]: record for record in structure}
    for name, item in additive.items():
        low = shares[name]["mean_yield_percent"] < 5
        x, y = item["row"]["r2_additive"], item["row"]["r2_rf"]
        if low:
            ax.scatter(x, y, s=30, facecolors="white", edgecolors=GRAY, linewidth=1.1, zorder=3)
        else:
            ax.scatter(x, y, s=34, color=HIGHLIGHT.get(name, POINT), edgecolors="white", linewidth=0.5, zorder=4)
    for name, (dx, dy, ha) in {"Grignard": (-6, 3, "right"), "Suzuki Bpin": (6, -3, "left"),
                               "Acylation": (6, -2, "left")}.items():
        item = additive[name]["row"]
        ax.annotate(label(name), (item["r2_additive"], item["r2_rf"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, color=INK)
    ax.plot((0.2, 0.8), (0.2, 0.8), color=GRAY, linewidth=1, linestyle=(0, (4, 4)))
    fraction = add_summary["subsets"]["mean_yield_at_least_5pct"]["median_additive_fraction_of_rf_held_out_well_r2"]
    ax.text(0.97, 0.04, f"Median additive / forest: {100 * fraction:.0f}%\n(11 screens with mean yield $\\geq$5%)",
            transform=ax.transAxes, ha="right", va="bottom", color=INK, linespacing=1.35)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel("Additive identity model, held-out-well $R$²")
    ax.set_ylabel("Random forest, held-out-well $R$²")
    style_axes(ax)
    panel_title(ax, "a  Identity explains most held-out-well $R$²")
    for name, item in additive.items():
        record = shares[name]
        x, y = 100 * record["pair_variance_share"], 100 * item["group"]["rf_skill_over_additive"]
        if record["mean_yield_percent"] < 5:
            skill_ax.scatter(x, y, s=30, facecolors="white", edgecolors=GRAY, linewidth=1.1, zorder=3)
        else:
            skill_ax.scatter(x, y, s=34, color=HIGHLIGHT.get(name, POINT), edgecolors="white", linewidth=0.5, zorder=4)
    for name, (dx, dy, ha) in {"Grignard": (-6, 5, "right"), "Acylation": (6, 0, "left"),
                               "Esterification": (4, -10, "left"), "Buchwald-Hartwig": (0, -13, "center")}.items():
        record = shares[name]
        skill_ax.annotate(label(name), (100 * record["pair_variance_share"], 100 * additive[name]["group"]["rf_skill_over_additive"]),
                          xytext=(dx, dy), textcoords="offset points", ha=ha,
                          color=INK)
    rho11 = add_summary["subsets"]["mean_yield_at_least_5pct"]["rho_pair_share_vs_rf_skill_over_additive_new_combinations"]
    skill_ax.text(0.97, 0.10, f"Spearman $\\rho$ = {rho11:.2f} (11 screens)".replace("-", "−"), transform=skill_ax.transAxes,
                  ha="right", va="bottom", color=INK, linespacing=1.35)
    skill_ax.axhline(0, color="#c9ced1", linewidth=0.8)
    skill_ax.set_xlim(18, 75)
    skill_ax.set_xlabel("Yield variance between substrate pairs (%)")
    skill_ax.set_ylabel("Forest gain over additive model,\nnew combinations (%)")
    style_axes(skill_ax)
    panel_title(skill_ax, "b  Gain falls with pair share")
    for record in structure:
        if record["mean_yield_percent"] < 5:
            continue
        name = record["screen"]
        color = HIGHLIGHT.get(name, POINT)
        level_ax.scatter(float(record["row_pair_level_r2"]), float(record["group_pair_level_r2"]), s=30, marker="o",
                         color=color, edgecolors="white", linewidth=0.5, zorder=4)
        level_ax.scatter(float(record["row_within_pair_r2"]), float(record["group_within_pair_r2"]), s=30, marker="^",
                         facecolors="white", edgecolors=color, linewidth=1.0, zorder=4)
    level_ax.plot((-0.1, 1.0), (-0.1, 1.0), color=GRAY, linewidth=1, linestyle=(0, (4, 4)))
    grignard = next(r for r in structure if r["screen"] == "Grignard")
    level_ax.annotate("Grignard", (float(grignard["row_pair_level_r2"]), float(grignard["group_pair_level_r2"])),
                      xytext=(-6, 2), textcoords="offset points", ha="right", color=INK)
    level_ax.set_xlim(-0.1, 1.0)
    level_ax.set_ylim(-0.1, 1.0)
    level_ax.set_xlabel("Held-out-well $R$²")
    level_ax.set_ylabel("New-combination $R$²")
    style_axes(level_ax)
    level_ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=POINT, markersize=5,
                                    label="Pair level (mean yield of each pair)"),
                             Line2D([], [], marker="^", linestyle="none", markerfacecolor="white",
                                    markeredgecolor=POINT, markersize=5, label="Within pair (across conditions)")],
                    loc="upper left", frameon=False)
    panel_title(level_ax, "c  The loss is at the pair level")
    note_ax.axis("off")
    medians, network, pretrained = decomposition_medians
    panel_title(note_ax, "d  Median $R$² across the 11 screens")
    rows = [("", "Held-out" + chr(10) + "wells", "New" + chr(10) + "combinations")]
    for model, values, prefix in (("Forest", medians, "median_"), ("Network", network, "median_mlp_"),
                                  ("Pretrained", pretrained, "median_rxnfp_")):
        rows.append((f"{model}, pair level", f"{values[prefix + 'row_pair_level_r2']:.2f}", f"{values[prefix + 'group_pair_level_r2']:.2f}"))
        rows.append((f"{model}, within pair", f"{values[prefix + 'row_within_pair_r2']:.2f}", f"{values[prefix + 'group_within_pair_r2']:.2f}"))
    for i, (label_text, left, right) in enumerate(rows):
        y = 0.95 - (0.13 if i else 0) - 0.085 * max(i - 1, 0)
        weight = "bold" if i == 0 else "normal"
        note_ax.text(0.02, y, label_text, color=INK, va="top", weight=weight)
        note_ax.text(0.55, y, left, color=INK, va="top", ha="center", weight=weight, linespacing=1.0)
        note_ax.text(0.82, y, right, color=INK, va="top", ha="center", weight=weight, linespacing=1.0)
    note_ax.text(0.02, 0.17, "On a new pair all three models lose most of\ntheir pair-level accuracy and far less\n"
                 "of their within-pair accuracy.", color=INK, va="top", linespacing=1.4)
    save(fig, "Figure_3")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def baseline_summary(path, screens):
    """Five-seed mean model MAE, median MAE and relative gain per screen and protocol."""
    rows = read_csv(path)
    out = {}
    for name in screens:
        out[name] = {}
        for protocol in ("row", "group"):
            selected = [r for r in rows if r["screen"] == name and r["protocol"] == protocol]
            if len(selected) != 5:
                raise ValueError(f"{name}/{protocol}: expected five seeds")
            out[name][protocol] = {
                "model_mae": np.mean([float(r["model_mae_percentage_points"]) for r in selected]),
                "median_mae": np.mean([float(r["training_median_mae_percentage_points"]) for r in selected]),
                "improvement": np.mean([float(r["model_mae_improvement_over_training_median"])
                                        for r in selected]),
            }
    return out


def main():
    import argparse
    global UNIFIED_RESULTS, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, default=UNIFIED_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    UNIFIED_RESULTS, OUTPUT = args.diagnostics_dir.resolve(), args.output_dir.resolve()
    use_style()
    benchmarks = json.loads(BENCHMARKS.read_text(encoding="utf-8"))
    rf_diagnostics = json.loads((UNIFIED_RESULTS / "rf_saved_score_diagnostics.json").read_text(encoding="utf-8"))
    rf_results = json.loads((UNIFIED_RESULTS / "rf_pooled_model_results.json").read_text(encoding="utf-8"))
    rf_ranks = json.loads((UNIFIED_RESULTS / "rf_model_comparison.json").read_text(encoding="utf-8"))
    rf_summary = json.loads((UNIFIED_RESULTS / "rf_baseline_weighting_summary.json").read_text(encoding="utf-8"))
    ranking = json.loads((UNIFIED_RESULTS / "rf_ranking_structure_summary.json").read_text(encoding="utf-8"))
    add_summary = json.loads((UNIFIED_RESULTS / "additive_baseline_summary.json").read_text(encoding="utf-8"))
    if set(add_summary["per_screen"]) != {s["screen"] for s in read_csv(UNIFIED_RESULTS / "rf_model_comparison.csv")}:
        raise ValueError("Additive-baseline summary disagrees with the RF population")
    rf_scores = read_csv(UNIFIED_RESULTS / "rf_model_comparison.csv")
    structure = read_csv(UNIFIED_RESULTS / "rf_yield_variance_structure.csv")
    reversals = read_csv(UNIFIED_RESULTS / "rf_pairwise_reversals.csv")
    for score in rf_scores:
        score["source_name"] = score.pop("screen")
        score["mean_yield_percent"] = float(score["mean_yield_percent"])
    for record in structure:
        for key in ("mean_yield_percent", "pair_variance_share", "condition_variance_share",
                    "rf_row_r2", "rf_group_r2", "rf_delta_r2", "row_pair_level_r2", "group_pair_level_r2",
                    "row_within_pair_r2", "group_within_pair_r2"):
            record[key] = float(record[key])
    if rf_diagnostics["provenance"]["input_sha256"] != hashlib.sha256(
            (UNIFIED_RESULTS / "rf_pooled_model_results.json").read_bytes()).hexdigest():
        raise ValueError("Saved-score diagnostics do not match the current pooled results")
    if (rf_results.get("status") != "completed" or len(rf_results.get("sources", [])) != 18
            or len(rf_scores) != 18 or sum(int(score["numeric_records"]) for score in rf_scores) != 340178):
        raise ValueError("Unexpected frozen unified-RF result population")
    if ({record["screen"] for record in structure} != {s["source_name"] for s in rf_scores}
            or len(reversals) != 55):
        raise ValueError("Ranking-structure diagnostics disagree with the RF population")
    for record in structure:
        score = next(s for s in rf_scores if s["source_name"] == record["screen"])
        if not np.isclose(record["rf_delta_r2"], float(score["rf_delta_r2"]), rtol=0, atol=1e-12):
            raise ValueError("Variance-structure scores disagree with the RF comparison")
    higher = [s["source_name"] for s in rf_scores if s["mean_yield_percent"] >= 5]
    baseline_means = baseline_summary(UNIFIED_RESULTS / "rf_median_baseline_by_seed.csv", higher)
    with plt.rc_context(RC):
        figure1(rf_scores, ranking, structure, rf_ranks)
        figure2(rf_scores, rf_ranks, reversals, ranking)
        figure5_error_scales(baseline_means, rf_ranks, rf_summary, ranking)
        extended_data_figure1(rf_scores, benchmarks)
        extended_data_figure2(rf_diagnostics, ranking)
        extended_data_figure3(rf_diagnostics)
        mlp = json.loads((ROOT / "output" / "p4_mlp" / "mlp_comparison_summary.json").read_text(encoding="utf-8"))
        rxnfp = json.loads((ROOT / "output" / "p4_rxnfp" / "rxnfp_comparison_summary.json").read_text(encoding="utf-8"))
        figure3_identity_reference(add_summary["per_screen"], structure, add_summary,
                                   (ranking["pair_level_decomposition"][SUBSET], mlp["subsets"][SUBSET],
                                    rxnfp["subsets"][SUBSET]))
    print(f"Saved Figures 1-3 and 5 and Extended Data Figures 1-3 as PNG and PDF in {OUTPUT}")


if __name__ == "__main__":
    main()
