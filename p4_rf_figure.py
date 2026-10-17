"""Draw Figure 4 (unified random forest) from verified saved diagnostics.

No prediction, fitting or change to the original figures is performed.
Defaults target the released RF diagnostics and its separate figure directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from p4_figure_style import GRAY, INK, PNG_DPI, RC, finalize, panel_title, use_style
import pandas as pd


ROW_COLOR = "#377e86"
GROUP_COLOR = "#d08a3c"
STRATA = ("both_seen", "one_unseen", "both_unseen")
STRATUM_COLORS = ("#4f5b62", "#9aa5ab", "#d5dadd")
STRATUM_LABELS = ("Both seen", "One unseen", "Both unseen")
METRICS = ("r2", "mae", "rmse")
SEEDS = (42, 20261007, 20261008, 20261009, 20261010)
SUBSET = "mean_yield_at_least_5pct"
ALIASES = {
    "C-O competitive reactions": "C-O competitive",
    "Allylic-substitution": "Allylic substitution",
    "Reductive-amination": "Reductive amination",
}


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#879197")
    ax.tick_params(colors=INK, length=3)
    ax.set_axisbelow(True)


def matched_cohort_summary(frame, screens):
    required = {"screen", "seed", "group_exposure_stratum", "rows", "row_mae", "group_mae"}
    if (not required.issubset(frame.columns) or len(set(screens)) != len(screens)
            or len(frame) != len(screens) * len(SEEDS) * len(STRATA)
            or set(frame.screen) != set(screens) or set(frame.seed) != set(SEEDS)
            or set(frame.group_exposure_stratum) != set(STRATA)
            or frame.duplicated(["screen", "seed", "group_exposure_stratum"]).any()):
        raise ValueError("Matched cohorts must cover each screen, seed and group-defined stratum once")
    counts = frame["rows"].to_numpy(float)
    if (not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts))):
        raise ValueError("Invalid matched-cohort row counts")
    active = frame.loc[frame["rows"] > 0, ["row_mae", "group_mae"]].to_numpy(float)
    if not np.isfinite(active).all() or np.any(active < 0):
        raise ValueError("Nonempty matched cohorts require finite nonnegative errors")
    if not frame.loc[frame["rows"] == 0, ["row_mae", "group_mae"]].isna().all().all():
        raise ValueError("Empty matched cohorts must have absent errors")
    records = []
    for screen in screens:
        selected = frame.loc[frame.screen.eq(screen)
                             & frame.group_exposure_stratum.eq("both_seen") & frame["rows"].gt(0)]
        record = {"screen": screen, "count": len(selected)}
        for kind in ("row", "group"):
            for statistic in ("mean", "min", "max"):
                record[f"{kind}_{statistic}"] = getattr(selected[f"{kind}_mae"], statistic)()
        records.append(record)
    return pd.DataFrame(records)


def load_diagnostics(directory):
    scores = pd.read_csv(directory / "rf_model_comparison.csv")
    ranks = json.loads((directory / "rf_model_comparison.json").read_text("utf-8"))
    strata = pd.read_csv(directory / "rf_substrate_strata_summary.csv")
    required = {"screen", "numeric_records", "mean_yield_percent", "rf_delta_r2"}
    required.update(f"{model}_{kind}_{metric}" for model in ("original", "rf")
                    for kind in ("row", "group") for metric in METRICS)
    if (not required.issubset(scores.columns) or len(scores) != 18
            or scores.screen.duplicated().any() or scores.numeric_records.sum() != 340178):
        raise ValueError("Unexpected frozen 18-screen RF comparison population")
    if not np.isfinite(scores[list(required - {"screen"})].to_numpy(float)).all():
        raise ValueError("Nonfinite RF comparison metric")
    if not np.allclose(scores.rf_row_r2 - scores.rf_group_r2, scores.rf_delta_r2,
                       rtol=1e-12, atol=1e-12):
        raise ValueError("RF protocol differences disagree with saved scores")
    higher = scores.loc[scores.mean_yield_percent >= 5]
    if len(higher) != 11:
        raise ValueError("The frozen non-low-yield population must contain 11 screens")
    for model, label in (("original", "original"), ("unified_rf", "rf")):
        for subset, frame in (("all_screens", scores), (SUBSET, higher)):
            item = ranks[model][subset]
            if (item["screens"] != len(frame)
                    or set(item["r2_order"]["row"]) != set(frame.screen)
                    or set(item["r2_order"]["group"]) != set(frame.screen)):
                raise ValueError("Saved ranking population differs from comparison table")
            for metric in METRICS:
                row = frame[f"{label}_row_{metric}"].rank().to_numpy()
                group = frame[f"{label}_group_{metric}"].rank().to_numpy()
                actual = (float(np.corrcoef(row, group)[0, 1])
                          if np.ptp(row) and np.ptp(group) else None)
                expected = item["metric_rho"][metric]
                if ((actual is None) != (expected is None)
                        or (actual is not None and not np.isclose(actual, expected,
                                                                 rtol=1e-12, atol=1e-12))):
                    raise ValueError("Saved rank correlation differs from comparison table")
    required_strata = {"screen", "protocol", "stratum", "row_share_mean", "mae_mean",
                       "mae_count", "mae_min", "mae_max"}
    if not required_strata.issubset(strata.columns):
        raise ValueError("Missing RF exposure-stratum summary fields")
    grouped = strata.loc[strata.protocol.eq("group")].copy()
    if (len(grouped) != 54 or grouped.duplicated(["screen", "stratum"]).any()
            or set(grouped.screen) != set(scores.screen)
            or set(grouped.stratum) != set(STRATA)):
        raise ValueError("Grouped exposure summaries must cover all 18 screens and three strata")
    shares = grouped.row_share_mean.to_numpy()
    if (not np.isfinite(shares).all() or np.any((shares < 0) | (shares > 1))
            or not np.allclose(grouped.groupby("screen").row_share_mean.sum(), 1,
                               rtol=1e-12, atol=1e-12)):
        raise ValueError("Grouped exposure shares do not recover full-screen coverage")
    active = grouped.loc[grouped.mae_count > 0]
    if (not np.isfinite(active[["mae_min", "mae_mean", "mae_max"]].to_numpy()).all()
            or (active.mae_min > active.mae_mean).any()
            or (active.mae_mean > active.mae_max).any()):
        raise ValueError("Invalid nonempty RF stratum errors or seed ranges")
    if not grouped.loc[grouped.mae_count.eq(0), "mae_mean"].isna().all():
        raise ValueError("Empty strata must have absent errors")
    matched = pd.read_csv(directory / "rf_matched_cohorts_by_seed.csv")
    cohorts = matched_cohort_summary(matched, scores.screen.tolist())
    return scores, ranks, grouped, cohorts


def rf_scatter(ax, scores, ranks):
    higher = scores.mean_yield_percent >= 5
    ax.scatter(scores.loc[~higher, "rf_row_r2"], scores.loc[~higher, "rf_group_r2"],
               s=42, facecolors="white", edgecolors=GRAY, linewidth=1.2, zorder=3)
    ax.scatter(scores.loc[higher, "rf_row_r2"], scores.loc[higher, "rf_group_r2"],
               s=43, color=ROW_COLOR, edgecolors="white", linewidth=0.6, zorder=3)
    values = scores[["rf_row_r2", "rf_group_r2"]].to_numpy()
    lower, upper = float(values.min()), float(values.max())
    padding = max((upper - lower) * 0.10, 0.06)
    limits = (min(lower - padding, -0.06), max(upper + padding, 0.06))
    ax.plot(limits, limits, color=GRAY, linestyle=(0, (4, 4)), linewidth=1)
    ax.axhline(0, color="#c9ced1", linewidth=0.7)
    for name, color, offset in (("Acylation", INK, (-8, -18)),
                                 ("Grignard", GROUP_COLOR, (8, 8))):
        score = scores.loc[scores.screen.eq(name)].iloc[0]
        ax.scatter(score.rf_row_r2, score.rf_group_r2, s=60, color=color,
                   edgecolors="white", linewidth=0.7, zorder=4)
        ax.annotate(name, (score.rf_row_r2, score.rf_group_r2), xytext=offset,
                    textcoords="offset points", color=INK, weight="bold",
                    ha="right" if offset[0] < 0 else "left")
    item = ranks["unified_rf"][SUBSET]
    rho = item["metric_rho"]["r2"]
    rho_text = f"{rho:.3f}" if rho is not None else "undefined"
    text = (f"{int((scores.rf_delta_r2 > 0).sum())}/18: row score > grouped score\n"
            f"Median difference: {scores.rf_delta_r2.median():.3f}\n"
            f"11-screen rank rho: {rho_text}")
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top",
            color=INK, linespacing=1.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
    ax.set(xlim=limits, ylim=limits, xlabel="RF held-out-well pooled $R$²",
           ylabel="RF combination-held-out pooled $R$²")
    handles = [Line2D([], [], marker="o", linestyle="none", color=ROW_COLOR,
                      label="Mean yield >=5%"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                      markeredgecolor=GRAY, color=GRAY, label="Mean yield <5%")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.73),
              frameon=False)


def rank_panel(ax, ranks):
    x = np.arange(len(METRICS))
    handles = []
    for offset, model, color, label in ((-0.08, "original", GRAY, "Original RF/XGB"),
                                         (0.08, "unified_rf", GROUP_COLOR, "Unified RF")):
        values = ranks[model][SUBSET]["metric_rho"]
        y = np.asarray([values[metric] if values[metric] is not None else np.nan
                        for metric in METRICS])
        ax.scatter(x + offset, y, s=50, color=color, zorder=3)
        handles.append(Line2D([], [], marker="o", linestyle="none", color=color, label=label))
        for position, value in zip(x + offset, y):
            if np.isfinite(value):
                ax.annotate(f"{value:.3f}", (position, value), xytext=(-5 if offset < 0 else 5, 9),
                            textcoords="offset points",
                            ha="right" if offset < 0 else "left", color=INK)
    for position, metric in enumerate(METRICS):
        pair = [ranks[model][SUBSET]["metric_rho"][metric] for model in ("original", "unified_rf")]
        if all(value is not None for value in pair):
            ax.plot([position - 0.08, position + 0.08], pair, color="#b8c0c4", linewidth=1)
    ax.axhline(0, color="#c9ced1", linewidth=0.8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-1.10, 1.17)
    ax.set_xticks(x, ("$R$²", "MAE", "RMSE"))
    ax.set_yticks((-1, -0.5, 0, 0.5, 1))
    ax.set_ylabel("Between-protocol Spearman rho")
    ax.set_xlabel("Ranking metric (same 11 screens)")
    ax.legend(handles=handles, loc="lower left", frameon=False)


def composition_panel(ax, screens, strata, low=()):
    y = np.arange(len(screens))
    left = np.zeros(len(screens))
    for label, color in zip(STRATA, STRATUM_COLORS):
        shares = strata.loc[strata.stratum.eq(label)].set_index("screen").loc[screens, "row_share_mean"].to_numpy() * 100
        ax.barh(y, shares, left=left, color=color, height=0.64)
        left += shares
    ax.set_xlim(0, 100)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xlabel("New-combination test wells (%)")
    screen_axis(ax, screens, low)


def matched_error_panel(ax, screens, cohorts, low=()):
    lookup = {name: index for index, name in enumerate(screens)}
    rows = cohorts.loc[cohorts["count"] > 0]
    centers = np.asarray([lookup[name] for name in rows.screen])
    for position, row in zip(centers, rows.itertuples(index=False)):
        ax.plot([row.row_mean * 100, row.group_mean * 100], [position - 0.10, position + 0.10],
                color="#b8c0c4", linewidth=0.9, zorder=2)
    for offset, kind, color in ((-0.10, "row", ROW_COLOR), (0.10, "group", GROUP_COLOR)):
        y = centers + offset
        mean, lower, upper = [rows[f"{kind}_{stat}"].to_numpy() * 100 for stat in ("mean", "min", "max")]
        ax.errorbar(mean, y, xerr=np.maximum(np.vstack((mean - lower, upper - mean)), 0),
                    fmt="o", color=color, markersize=4, linewidth=0.9,
                    capsize=2, zorder=3,
                    label="Held-out wells" if kind == "row" else "New combinations")
    maximum = float(rows[["row_max", "group_max"]].max().max()) * 100
    ax.set_xlim(0, max(maximum * 1.08, 1))
    ax.set_xlabel("MAE on the same wells (percentage points)")
    screen_axis(ax, screens, low)
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.82), frameon=False)


def screen_axis(ax, screens, low=()):
    ax.set_yticks(np.arange(len(screens)), [ALIASES.get(name, name) for name in screens])
    for tick, name in zip(ax.get_yticklabels(), screens):
        if name in low:
            tick.set_color(GRAY)
            tick.set_style("italic")
    ax.set_ylim(len(screens) - 0.5, -0.5)


def draw_figure(scores, ranks, strata, cohorts, output, order=None):
    use_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.6),
                             gridspec_kw={"height_ratios": (1.20, 1.0)}, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.95), h_pad=0.18, w_pad=0.05)
    screens = list(order) if order is not None else scores.screen.tolist()
    low = set(scores.loc[scores.mean_yield_percent < 5, "screen"])
    composition_panel(axes[0], screens, strata, low)
    matched_error_panel(axes[1], screens, cohorts, low)
    for ax, title in zip(axes, ("a  Most test wells pair two substrates seen in training",
                                "b  Seen substrates, new pair: error rises most at high pair share")):
        style_axes(ax)
        panel_title(ax, title)
        for tick in ax.get_yticklabels():
            if tick.get_text() in {ALIASES.get(name, name) for name in low}:
                tick.set_color(GRAY)
                tick.set_style("italic")
    handles = [Line2D([], [], marker="s", linestyle="none", color=color, label=label,
                      markersize=7) for color, label in zip(STRATUM_COLORS, STRATUM_LABELS)]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                   frameon=False, ncol=3)
    fig.text(0.5, 0.009,
             "Rows: highest between-pair yield-variance share first; grey italic labels, mean yield below 5%.",
             ha="center", color=INK)
    output.mkdir(parents=True, exist_ok=True)
    name = "Figure_4"
    finalize(fig)
    fig.savefig(output / f"{name}.png", dpi=PNG_DPI, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--diagnostics-dir", type=Path,
                        default=root / "output/p4_unified_rf/diagnostics")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "figures")
    args = parser.parse_args()
    diagnostics, output = args.diagnostics_dir.resolve(), args.output_dir.resolve()
    if diagnostics.is_relative_to(output):
        raise ValueError("The figure directory must not contain the diagnostic inputs")
    scores, ranks, strata, cohorts = load_diagnostics(diagnostics)
    structure = pd.read_csv(diagnostics / "rf_yield_variance_structure.csv")
    if set(structure.screen) != set(scores.screen):
        raise ValueError("Variance-structure screens disagree with the RF comparison")
    order = structure.sort_values("pair_variance_share", ascending=False).screen.tolist()
    draw_figure(scores, ranks, strata, cohorts, output, order)
    print(f"Saved Figure_4.png and .pdf in {output}")


if __name__ == "__main__":
    main()
