"""Draw the Chem main Figures 1-5 and the graphical abstract from the released results, without fitting.

Run from the package directory:  python p4_chem_figures.py --output-dir figures
Figure 1 is written as a layered, editable SVG (groups named by content) and converted to PDF and PNG; Figures 2-5
and the graphical abstract are drawn with matplotlib. Fonts and panel labels follow p4_figure_style.py at 9 pt and 8 pt.

Colour semantics, used in every figure: teal = held-out wells; orange = new substrate pairs; purple = Grignard
addition; green = esterification; grey = other arrays; filled / open marker = array mean yield >= 5% / < 5%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from rdkit import Chem  # noqa: E402

import p4_figure_style as style  # noqa: E402
import p4_structure_drawing as chem  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE / "output"
DIAG = OUTPUT / "p4_unified_rf" / "diagnostics"
OUT = PACKAGE / "figures"





style.LETTER_SIZE, style.TITLE_SIZE, style.LABEL_SIZE, style.TEXT_SIZE = 9, 9, 8, 8
style.TITLE_INDENT = 11
style.use_style()

INK, GRAY = style.INK, style.GRAY
TEAL, ORANGE = "#2E7F86", "#D08A3C"
PURPLE, GREEN = "#5B4B9A", "#3F8A55"
LIGHT = "#c3c8cd"
MID = "#8d959b"
WIDTH = 7.0  # inches, final print width
ANALYSED = ("quantified", "zero_by_rule")
LOW_CUTOFF = 5.0  # percent mean yield

NAMES = {
    "Reductive-amination": "Reductive amination", "Buchwald-Hartwig": "Buchwald–Hartwig",
    "Buchwald-Hartwig CMC": "Buchwald–Hartwig CMC", "Allylic-substitution": "Allylic substitution",
    "C-O competitive reactions": "C–O competitive", "Pd-carbonylation": "Pd carbonylation",
}
HIGHLIGHT = {"Grignard": PURPLE, "Esterification": GREEN}


def name(screen: str) -> str:
    return NAMES.get(screen, screen)


def structure() -> pd.DataFrame:
    """Per-array table: yields, variance shares, design quantities and the forest's scores (scope b)."""
    table = pd.read_csv(DIAG / "rf_yield_variance_structure.csv")
    assert len(table) == 19, len(table)
    table["high"] = table.mean_yield_percent >= LOW_CUTOFF
    assert table.high.sum() == 11
    return table


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records(screens) -> pd.DataFrame:
    data = pd.read_csv(PACKAGE / "data" / "p4_records.csv.gz",
                       usecols=["screen", "measurement_status", "pair_key", "condition_id", "sub_1_smiles",
                                "sub_2_smiles", "yield_fraction"])
    data = data[data.screen.isin(screens)]
    assert set(data.screen) == set(screens), set(screens) - set(data.screen)
    return data


def marker_style(row, base=MID, size=26):
    """Filled for arrays with mean yield >= 5 %, open otherwise; highlighted arrays keep their colour."""
    colour = HIGHLIGHT.get(row.screen, base)
    if row.high:
        return dict(s=size, color=colour, edgecolor="white", linewidths=0.5, zorder=3)
    return dict(s=size, facecolor="white", edgecolor=colour, linewidths=1.0, zorder=3)


def despine(ax, keep=("left", "bottom")):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(side in keep)
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(color=GRAY, width=0.6, length=2.5)


def save(fig, stem: str):
    style.finalize(fig)
    fig.savefig(OUT / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    if chem.compose_pdf(fig, OUT / f"{stem}.pdf"):
        import pymupdf
        with pymupdf.open(OUT / f"{stem}.pdf") as document:
            document[0].get_pixmap(dpi=style.PNG_DPI).save(OUT / f"{stem}.png")
    else:
        fig.savefig(OUT / f"{stem}.png", dpi=style.PNG_DPI)
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf")


def rho(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


SVG_GREY, SVG_LIGHT, CELL = "#7a8389", "#c3c8cd", "#d9dde1"
W, H = 510, 334


def text(x, y, s, size=8, weight="normal", anchor="start", fill=INK, style=""):
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}"{st}>{escape(s)}</text>')


def lines(x, y, rows, size=8, gap=9.6, **kw):
    return "".join(text(x, y + i * gap, r, size=size, **kw) for i, r in enumerate(rows))


def rect(x, y, w, h, fill="none", stroke="none", sw=0.6, dash="", rx=0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke=SVG_GREY, sw=0.6, cap="round"):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="{cap}"/>')


def arrow(x1, y1, x2, y2, c1=None):
    """Quadratic curve with an explicit arrowhead along the end tangent."""
    cx, cy = c1 if c1 else ((x1 + x2) / 2, y1)
    dx, dy = x2 - cx, y2 - cy
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / norm, dy / norm
    length, half = 5.0, 2.2
    bx, by = x2 - ux * length, y2 - uy * length
    head = (f'<path d="M{x2:.1f},{y2:.1f} L{bx - uy * half:.1f},{by + ux * half:.1f} '
            f'L{bx + uy * half:.1f},{by - ux * half:.1f}z" fill="{SVG_GREY}"/>')
    return (f'<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {bx:.1f},{by:.1f}" fill="none" stroke="{SVG_GREY}" '
            f'stroke-width="0.9"/>' + head)


def group(name, body):
    return f'<g id="{name}">{body}</g>\n'


def figure1():
    """Layered SVG: one real screening table (Grignard) whose highlighted row splits into the two tests."""
    table = structure()
    high = table[table.high].copy()
    high["row_rank"] = high.rf_row_r2.rank(ascending=False).astype(int)
    high["group_rank"] = high.rf_group_r2.rank(ascending=False).astype(int)
    ranks = high.set_index("screen")
    rho11 = high[["rf_row_r2", "rf_group_r2"]].corr(method="spearman").iloc[0, 1]
    med = high[["row_pair_level_r2", "group_pair_level_r2", "row_within_pair_r2", "group_within_pair_r2"]].median()
    r_share = high[["pair_variance_share", "rf_delta_r2"]].corr(method="spearman").iloc[0, 1]
    r_levels = high[["substrate_levels_per_pair", "rf_delta_r2"]].corr(method="spearman").iloc[0, 1]
    records_n = int(table.numeric_records.sum())
    assert records_n == 378070 and len(table) == 19
    assert (ranks.loc["Grignard", "row_rank"], ranks.loc["Grignard", "group_rank"]) == (2, 11)
    assert (ranks.loc["Buchwald-Hartwig", "row_rank"], ranks.loc["Buchwald-Hartwig", "group_rank"]) == (11, 4)
    g_row = ranks.loc["Grignard"]
    grignard, g_flags, _ = matrix(records(["Grignard"]))
    n_rows, n_cols = grignard.shape
    assert (n_rows, n_cols) == (136, 47)

    def colour(value, flagged):
        if flagged:
            return MISSING
        if not np.isfinite(value):
            return "white"
        return matplotlib.colors.to_hex(CMAP(value))

    parts = [group("question", text(6, 14, "Is this reaction worth modelling for substrate pairs not yet run?", size=9,
                                    weight="bold")),
             group("scale", text(W - 6, 14, f"19 arrays · 17 reaction families · {records_n:,} yields",
                                 anchor="end", fill=SVG_GREY))]

    # visual centre: the real Grignard table, one row highlighted
    hx, hy, hw, hh = 8, 50, 118, 176
    cw, ch = hw / n_cols, hh / n_rows
    pick = 24  # a productive pair whose row is shown in both branches
    body = text(hx, hy - 16, "One screening table", weight="bold")
    body += text(hx, hy - 6, f"Grignard addition: {n_rows} pairs × {n_cols} conditions", fill=SVG_GREY)
    for r in range(n_rows):
        for c in range(n_cols):
            body += rect(hx + c * cw, hy + r * ch, cw + 0.05, ch + 0.05,
                         fill=colour(grignard.values[r, c], g_flags[r, c]))
    body += rect(hx - 0.8, hy - 0.8, hw + 1.6, hh + 1.6, stroke=PURPLE, sw=1.2)
    body += rect(hx - 0.4, hy + pick * ch - 0.6, hw + 0.8, ch + 1.2, stroke=INK, sw=0.9)
    body += text(hx, hy + hh + 10, "rows: pairs · columns: conditions", fill=SVG_GREY)
    parts.append(group("table", body))

    # the table read before modelling, written where it applies
    body = text(hx, hy + hh + 26, "Read before modelling", weight="bold")
    body += text(hx, hy + hh + 37, f"between-pair share {100 * g_row.pair_variance_share:.0f}%")
    body += text(hx, hy + hh + 47, f"pairing sparsity {g_row.substrate_levels_per_pair:.2f}")
    body += text(hx, hy + hh + 59, f"both track the loss: {r_share:.2f}, {r_levels:.2f}", fill=SVG_GREY)
    body += text(hx, hy + hh + 69, "(rank correlations, 11 arrays)", fill=SVG_GREY)
    parts.append(group("before-modelling", body))

    # two branches built from the highlighted row
    bx, sw_cell = 150, 3.1
    strip_w = n_cols * sw_cell
    up_y, lo_y = 72, 168
    values, flags = grignard.values[pick], g_flags[pick]
    held = {4, 12, 21, 30, 39}
    row_y = hy + (pick + 0.5) * ch
    body = arrow(hx + hw + 3, row_y, bx - 5, up_y + 5, c1=(bx - 18, up_y + 5))
    body += arrow(hx + hw + 3, row_y, bx - 5, lo_y + 5, c1=(bx - 18, lo_y + 5))
    body += text(bx, up_y - 6, "Held-out wells", weight="bold")
    for c in range(n_cols):
        if c in held:
            body += rect(bx + c * sw_cell + 0.4, up_y + 0.4, sw_cell - 0.8, 9.2, fill="white", stroke=TEAL, sw=0.9)
        else:
            body += rect(bx + c * sw_cell, up_y, sw_cell + 0.05, 10, fill=colour(values[c], flags[c]))
    body += lines(bx, up_y + 22, ["a few wells are hidden; the pair's other", "wells stay in training: the model recalls"])
    body += text(bx, up_y + 44, "random splits: Ahneman 2018, Schwaller 2021", fill=SVG_GREY)
    body += text(bx, lo_y - 6, "New pairs", weight="bold")
    body += rect(bx + 0.4, lo_y + 0.4, strip_w - 0.8, 9.2, fill="white", stroke=ORANGE, sw=1.2)
    body += text(bx + strip_w / 2, lo_y + 8.2, "?", anchor="middle", weight="bold")
    body += lines(bx, lo_y + 22, ["the whole pair and its plates are withheld:", "the model must predict the pair"])
    body += lines(bx, lo_y + 44, ["withheld components or pairs:", "Sandfort 2020, Ismail 2026,",
                                  "Götz 2025, Neves 2026"], fill=SVG_GREY)
    parts.append(group("branches", body))

    # what the forest loses on new pairs, beside the branch where it happens
    ly = lo_y + 86
    body = rect(bx, ly - 7, 16, 7, fill="#55606a")
    body += text(bx + 21, ly, f"pair mean R² {med.row_pair_level_r2:.2f} → {med.group_pair_level_r2:.2f}",
                 weight="bold")
    for k, dy in enumerate((3, -2, 4, -3, 2)):
        body += line(bx + 2 + 3 * k, ly + 8, bx + 2 + 3 * k, ly + 8 - dy, stroke=SVG_GREY, sw=1.0, cap="butt")
    body += text(bx + 21, ly + 11, f"within pair {med.row_within_pair_r2:.2f} → {med.group_within_pair_r2:.2f}")
    body += text(bx + 21, ly + 21, "median R², 11 arrays", fill=SVG_GREY)
    parts.append(group("lost", body))

    # rank ladder fed by the two branches
    lx0, lx1, ly0, step = 392, 450, 88, 13
    bottom = ly0 + 10 * step
    body = ""
    for x, label, hidden in ((lx0, "held-out", True), (lx1, "new pairs", False)):
        for c in range(6):
            if hidden:
                held_cell = c in (1, 4)
                body += rect(x - 15 + 5 * c + 0.3, ly0 - 31.7, 4.4, 5.4, fill="white" if held_cell else CELL,
                             stroke=TEAL if held_cell else "none", sw=0.8 if held_cell else 0.6)
        if not hidden:
            body += rect(x - 15, ly0 - 32, 29.6, 6, fill=ORANGE)
        body += text(x, ly0 - 14, label, anchor="middle", fill=SVG_GREY)
    order = sorted(ranks.index, key=lambda s: (s in ("Grignard", "Buchwald-Hartwig"), s))
    for screen in order:
        r, g = ranks.loc[screen, "row_rank"], ranks.loc[screen, "group_rank"]
        stroke, sw = SVG_LIGHT, 0.9
        if screen == "Grignard":
            stroke, sw = PURPLE, 2.6
        elif screen == "Buchwald-Hartwig":
            stroke, sw = INK, 1.8
        body += line(lx0, ly0 + (r - 1) * step, lx1, ly0 + (g - 1) * step, stroke=stroke, sw=sw)
    for k in range(11):
        for x in (lx0, lx1):
            body += f'<circle cx="{x}" cy="{ly0 + k * step:.1f}" r="2.1" fill="{SVG_GREY}"/>'
    body += text(lx0 - 6, ly0 + step + 3, "Grignard 2nd", anchor="end", weight="bold")
    body += text(lx0 - 6, ly0 + 10 * step + 3, "Buchwald–Hartwig 11th", anchor="end", weight="bold")
    body += text(lx1 + 6, ly0 + 3 * step + 3, "4th", weight="bold")
    body += text(lx1 + 6, ly0 + 10 * step + 3, "11th", weight="bold")
    body += text((lx0 + lx1) / 2, bottom + 28, f"rank correlation {rho11:.2f}", anchor="middle", weight="bold")
    body += text((lx0 + lx1) / 2, bottom + 38, "11 arrays averaging ≥5% yield", anchor="middle", fill=SVG_GREY)
    parts.append(group("ladder", body))

    # the gap: no earlier work compared the order of reactions
    body = rect(lx0 - 76, 24, 178, 28, stroke=INK, sw=0.9, dash="3 2")
    body += text(lx0 - 70, 35, "Does the order of reactions survive?", weight="bold")
    body += text(lx0 - 70, 46, "not compared before: this work", fill=SVG_GREY)
    parts.append(group("gap", body))

    parts.append(group("grignard-chemistry", text(bx + 150, H - 52, "Grignard addition", weight="bold")
                       + chem.svg_group("grignard_addition", bx + 150, H - 47)))

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{H * 25.4 / 72:.1f}mm" '
           f'viewBox="0 0 {W} {H}">\n'
           f'<rect width="{W}" height="{H}" fill="white"/>\n' + "".join(parts) + "</svg>\n")
    (OUT / "Figure_1.svg").write_text(svg, encoding="utf-8")
    print("wrote", OUT / "Figure_1.svg")


def motif_row(ax, x, y, hidden, cells=6, width=0.6, height=0.42):
    """The motif: one substrate pair across conditions. Teal-outlined cells are held out; an orange row is withheld."""
    step = width / cells
    left = x - width / 2
    if hidden == "all":
        ax.add_patch(Rectangle((left, y - height / 2), width - 0.15 * step, height, facecolor=ORANGE, lw=0,
                               clip_on=False))
        return
    for c in range(cells):
        held = c in hidden
        ax.add_patch(Rectangle((left + c * step, y - height / 2), 0.85 * step, height, clip_on=False,
                               facecolor="white" if held else "#d9dde1", edgecolor=TEAL if held else "none",
                               lw=1.2 if held else 0))


def figure2():
    table = structure()
    high = table[table.high].copy()
    high["row_rank"] = high.rf_row_r2.rank(ascending=False).astype(int)
    high["group_rank"] = high.rf_group_r2.rank(ascending=False).astype(int)
    ranks = high.set_index("screen")
    assert (ranks.loc["Grignard", "row_rank"], ranks.loc["Grignard", "group_rank"]) == (2, 11)
    assert (ranks.loc["Buchwald-Hartwig", "row_rank"], ranks.loc["Buchwald-Hartwig", "group_rank"]) == (11, 4)
    structure_summary = load_json(DIAG / "rf_ranking_structure_summary.json")
    reversals = structure_summary["pairwise_reversals_mean_yield_at_least_5pct"]
    rho11 = rho(high.rf_row_r2, high.rf_group_r2)
    by_seed = pd.read_csv(DIAG / "additive_baseline_by_seed.csv")
    rf_seed = by_seed.pivot_table(index=["screen", "seed"], columns="protocol", values="r2_rf")
    assert len(rf_seed) == 95
    losses_all = int((rf_seed["row"] > rf_seed["group"]).sum())
    selection = load_json(DIAG / "rf_saved_score_diagnostics.json")["retrospective_selection"]["screen_mean"]["by_k"]
    random_pick = high.rf_group_r2.mean()

    fig = plt.figure(figsize=(WIDTH, 4.35))
    grid = GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1], height_ratios=[1, 1], wspace=0.42, hspace=0.62,
                    left=0.01, right=0.985, top=0.9, bottom=0.1)

    # a: rank slope chart (hero)
    ax = fig.add_subplot(grid[:, 0])
    style.panel_title(ax, "a  Grignard falls from 2nd to 11th")
    order = [s for s in high.sort_values("row_rank").screen]
    for screen in order:
        r, g = ranks.loc[screen, "row_rank"], ranks.loc[screen, "group_rank"]
        colour, width, z = LIGHT, 1.0, 1
        if screen == "Grignard":
            colour, width, z = PURPLE, 2.6, 3
        elif screen == "Buchwald-Hartwig":
            colour, width, z = INK, 2.0, 2
        ax.plot([0, 1], [r, g], color=colour, lw=width, zorder=z, solid_capstyle="round")
        ax.scatter([0, 1], [r, g], s=14, color=colour if z > 1 else MID, zorder=z + 1)
        bold = "bold" if screen in ("Grignard", "Buchwald-Hartwig") else "normal"
        ax.text(-0.07, r, f"{name(screen)}  {r}", ha="right", va="center", color=INK, weight=bold)
        ax.text(1.07, g, f"{g}  {name(screen)}", ha="left", va="center", color=INK, weight=bold)
    ax.text(0, 0.1, "Held-out\nwells", ha="center", va="bottom", color=INK, weight="bold")
    ax.text(1, 0.1, "New\npairs", ha="center", va="bottom", color=INK, weight="bold")
    motif_row(ax, 0, -1.75, hidden=(1, 4))
    motif_row(ax, 1, -1.75, hidden="all")
    ax.text(0.5, 12.25, f"Rank correlation {rho11:.2f} (11 arrays)\n"
                        f"{reversals['reversed_in_mean']} of {reversals['pairs']} array pairs change order, "
                        f"{reversals['reversed_in_all_seeds']} in all five seeds",
            ha="center", va="top", color=INK)
    ax.set_xlim(-1.75, 2.05)
    ax.set_ylim(13.4, -2.25)
    ax.axis("off")

    # b: held-out vs new R2, all 19 arrays
    ax = fig.add_subplot(grid[0, 1])
    style.panel_title(ax, "b  Every array loses accuracy on new pairs")
    lim = (-0.3, 0.85)
    ax.plot(lim, lim, ls=(0, (3, 2)), color=GRAY, lw=0.7, zorder=1)
    for row in table.itertuples():
        base = INK if row.screen == "Buchwald-Hartwig" else MID
        ax.scatter(row.rf_row_r2, row.rf_group_r2, **marker_style(row, base=base))
    placed = {"Grignard": ((0.681, -0.065), "center", False), "Acylation": ((0.748, 0.59), "center", False),
              "Heck": ((0.605, 0.499), "left", False), "Esterification": ((0.30, 0.565), "right", True),
              "Buchwald-Hartwig": ((0.20, 0.455), "right", True)}
    for row in table.itertuples():
        if row.screen in placed:
            (tx, ty), ha, leader = placed[row.screen]
            ax.annotate(name(row.screen), (row.rf_row_r2, row.rf_group_r2), xytext=(tx, ty), textcoords="data",
                        ha=ha, va="center", color=INK,
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkA=1, shrinkB=3) if leader else None)
    ax.text(-0.27, 0.8, f"{losses_all} of {len(rf_seed)} array × seed comparisons\nfall below the line; "
                        f"median loss {table.rf_delta_r2.median():.2f}", ha="left", va="top", color=INK)
    ax.text(0.83, -0.27, "open: arrays averaging < 5% yield", ha="right", va="bottom", color=GRAY)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("R² on held-out wells")
    ax.set_ylabel("R² on new pairs")
    despine(ax)

    # c: picking arrays by either score
    ax = fig.add_subplot(grid[1, 1])
    style.panel_title(ax, "c  Picking by held-out wells is no better than random")
    ks = [r["k"] for r in selection]
    by_row = [r["group_r2_mean_row_selected"] for r in selection]
    by_group = [r["group_r2_mean_group_selected"] for r in selection]
    ax.axhline(random_pick, color=GRAY, ls=(0, (3, 2)), lw=0.7)
    ax.plot(ks, by_group, color=ORANGE, lw=1.6, marker="o", ms=3.2)
    ax.plot(ks, by_row, color=TEAL, lw=1.6, marker="o", ms=3.2)
    ax.text(5.3, 0.475, "picked by new-pair R²\n(best possible)", color=INK, ha="left", va="bottom")
    ax.text(4.4, 0.222, "picked by held-out-well R²", color=INK, ha="left", va="bottom")
    ax.text(11, random_pick - 0.014, f"random pick, {random_pick:.2f}", color=INK, ha="right", va="top")
    ax.annotate(f"{by_row[2]:.2f}", (3, by_row[2]), xytext=(0, -7), textcoords="offset points", ha="center",
                va="top", color=INK)
    ax.annotate(f"{by_group[2]:.2f}", (3, by_group[2]), xytext=(5, 5), textcoords="offset points", ha="left",
                va="bottom", color=INK)
    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(0.2, 0.7)
    ax.set_xticks(range(1, 12, 2))
    ax.set_xlabel("Number of arrays picked")
    ax.set_ylabel("Mean R² on new pairs")
    despine(ax)
    save(fig, "Figure_2")


PAIR_DY, WITHIN_DY = -0.17, 0.17


def dumbbell(ax, y, row_pair, group_pair, row_within, group_within, size=1.0):
    ax.plot([row_pair, group_pair], [y + PAIR_DY] * 2, color="#55606a", lw=1.6 * size, zorder=1,
            solid_capstyle="round")
    ax.scatter([row_pair], [y + PAIR_DY], s=24 * size, color=TEAL, zorder=3, edgecolor="white", linewidths=0.4)
    ax.scatter([group_pair], [y + PAIR_DY], s=24 * size, color=ORANGE, zorder=3, edgecolor="white", linewidths=0.4)
    ax.plot([row_within, group_within], [y + WITHIN_DY] * 2, color=LIGHT, lw=1.0 * size, zorder=1)
    ax.scatter([row_within], [y + WITHIN_DY], s=20 * size, marker="^", facecolor="white", edgecolor=TEAL,
               linewidths=1.0, zorder=3)
    ax.scatter([group_within], [y + WITHIN_DY], s=20 * size, marker="^", facecolor="white", edgecolor=ORANGE,
               linewidths=1.0, zorder=3)


def figure3():
    table = structure()
    high = table[table.high].sort_values("pair_variance_share", ascending=False)
    external = load_json(OUTPUT / "p4_external" / "external_hte_summary.json")
    additive = load_json(DIAG / "additive_baseline_summary.json")["per_screen"]
    models = {"Random forest": (high, "")}
    for label, stem, prefix in [("Neural network", "p4_mlp/mlp_comparison_by_screen.csv", "mlp_"),
                                ("Pretrained reaction\nrepresentation", "p4_rxnfp/rxnfp_comparison_by_screen.csv",
                                 "rxnfp_")]:
        frame = pd.read_csv(OUTPUT / stem)
        frame = frame[frame.mean_yield_percent >= 5.0]
        assert len(frame) == 11
        models[label] = (frame, prefix)

    fig = plt.figure(figsize=(WIDTH, 4.6))
    grid = GridSpec(2, 2, figure=fig, width_ratios=[0.95, 1], height_ratios=[0.8, 1.2], wspace=0.42, hspace=0.62,
                    left=0.215, right=0.985, top=0.9, bottom=0.1)

    # a: per-array dumbbells (hero)
    ax = fig.add_subplot(grid[:, 0])
    style.panel_title(ax, "a  New pairs lose mainly the pair mean")
    rows = list(high.itertuples())
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows):
        dumbbell(ax, y, row.row_pair_level_r2, row.group_pair_level_r2, row.row_within_pair_r2,
                 row.group_within_pair_r2)
        ax.text(-0.3, y, f"{name(row.screen)}  {100 * row.pair_variance_share:.0f}%", ha="right", va="center",
                color=INK, weight="bold" if row.screen == "Grignard" else "normal")
    ext_y = len(rows) + 0.9
    for i, (label, key) in enumerate([("C–N coupling, ref. 11", "Buchwald-Hartwig (Ahneman et al.)"),
                                      ("Suzuki coupling, ref. 2", "Suzuki-Miyaura (Perera et al.)")]):
        e = external[key]
        y = ext_y + i
        dumbbell(ax, y, e["row_pair_level_r2"], e["group_pair_level_r2"], e["row_within_pair_r2"],
                 e["group_within_pair_r2"])
        ax.text(-0.3, y, f"{label}  {100 * e['pair_variance_share']:.0f}%", ha="right", va="center", color=INK)
    ax.axhline(len(rows) + 0.2, color=LIGHT, lw=0.6)
    ax.text(-0.3, len(rows) + 0.42, "Other laboratories", ha="right", va="center", color=GRAY, style="italic")
    ax.text(-0.3, -1.05, "Array  share of yield\nvariance between pairs", ha="right", va="bottom", color=GRAY)
    # key with the motif: a pair's row is its mean (dark bar, circles) plus deviations within it (ticks, triangles)
    ax.add_patch(Rectangle((0.2, -1.06), 0.15, 0.2, facecolor="#55606a", lw=0, clip_on=False))
    ax.text(0.4, -0.96, "pair mean (circles)", ha="left", va="center", color=INK)
    for k, dy in enumerate((0.1, -0.08, 0.12, -0.1, 0.06)):
        ax.plot([0.215 + 0.03 * k] * 2, [-0.52, -0.52 - dy], color=GRAY, lw=1.0, solid_capstyle="butt",
                clip_on=False)
    ax.plot([0.2, 0.35], [-0.52, -0.52], color=LIGHT, lw=0.4, clip_on=False)
    ax.text(0.4, -0.52, "within pair (triangles)", ha="left", va="center", color=INK)
    ax.set_xlim(-0.3, 1.02)
    ax.set_ylim(ext_y + 1.6, -1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("R² (teal: held-out wells; orange: new pairs)")
    ax.set_yticks([])
    despine(ax, keep=("bottom",))

    # b: three models, medians
    ax = fig.add_subplot(grid[0, 1])
    style.panel_title(ax, "b  Three models all lose the pair mean")
    for y, (label, (frame, prefix)) in enumerate(models.items()):
        m = [frame[f"{prefix}{c}"].median() for c in
             ("row_pair_level_r2", "group_pair_level_r2", "row_within_pair_r2", "group_within_pair_r2")]
        dumbbell(ax, y * 1.15, *m, size=1.1)
        ax.text(-0.14, y * 1.15, label, ha="right", va="center", color=INK)
        ax.text(m[1] - 0.05, y * 1.15 + PAIR_DY, f"{m[1]:.2f}", ha="right", va="center", color=INK)
        ax.text(m[0] + 0.05, y * 1.15 + PAIR_DY, f"{m[0]:.2f}", ha="left", va="center", color=INK)
    ax.set_xlim(-0.86, 1.2)
    ax.set_ylim(2.75, -0.55)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("Median R², 11 arrays", x=0.65)
    ax.set_yticks([])
    despine(ax, keep=("bottom",))
    ax.spines["bottom"].set_bounds(0, 1.0)

    # c: chemistry-free references on held-out wells
    ax = fig.add_subplot(grid[1, 1])
    style.panel_title(ax, "c  No chemistry needed on held-out wells")
    lim = (0.15, 0.8)
    ax.plot(lim, lim, ls=(0, (3, 2)), color=GRAY, lw=0.7, zorder=1)
    add = high.screen.map(lambda s: additive[s]["row"]["r2_additive"])
    ax.scatter(high.rf_row_r2, add, s=24, marker="s", color=TEAL, edgecolor="white", linewidths=0.4, zorder=3)
    ax.scatter(high.rf_row_r2, high.row_sister_well_mean_r2, s=26, marker="D", facecolor="white", edgecolor=TEAL,
               linewidths=1.0, zorder=3)
    ratio_add = (add / high.rf_row_r2).median()
    ratio_sister = (high.row_sister_well_mean_r2 / high.rf_row_r2).median()
    ax.scatter([0.18], [0.785], s=24, marker="s", color=TEAL, edgecolor="white", linewidths=0.4)
    ax.text(0.2, 0.785, f"identities only: {100 * ratio_add:.0f}%", ha="left", va="center", color=INK)
    ax.scatter([0.18], [0.735], s=26, marker="D", facecolor="white", edgecolor=TEAL, linewidths=1.0)
    ax.text(0.2, 0.735, f"pair's other wells: {100 * ratio_sister:.0f}%", ha="left", va="center",
            color=INK)
    ax.text(0.2, 0.682, "medians, as % of the forest", ha="left", va="center", color=GRAY)
    g = high.set_index("screen").loc["Grignard"]
    ax.annotate("Grignard", (g.rf_row_r2, g.row_sister_well_mean_r2), xytext=(0.72, 0.62), textcoords="data",
                ha="center", va="center", color=INK, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkB=3))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Random forest R², held-out wells")
    ax.set_ylabel("Reference R², held-out wells")
    despine(ax)
    save(fig, "Figure_3")


YIELD_CMAP = LinearSegmentedColormap.from_list("yield", ["#f5eedc", "#a9c4d4", "#4f7fa3", "#1d3557"])
CMAP = YIELD_CMAP
MISSING, MISSING_HATCH = "#e4e7ea", "#8d959b"
AMINO = Chem.MolFromSmarts("[NX3;!$(N=*);!$(N-[C,S]=O);!$([N+])]-c")
PROTIC = Chem.MolFromSmarts("[#7,#8;!H0]")
# Selected Grignard pairs (reagent, carbonyl partner): the four highest pair means, then four amino/N-H partners
# whose organomagnesium reagent gives product with other partners. Asset names refer to chemical_assets/.
EXAMPLES = [
    ("Fc1ccc([Mg]Br)cc1", "O=Cc1cccnc1Cl", "chloronicotinaldehyde", "4-FC₆H₄MgBr"),
    ("Cc1ccccc1[Mg]Br", "O=C(c1ccccn1)c1ccccn1", "dipyridyl_ketone", "2-MeC₆H₄MgBr"),
    ("Cc1ccc([Mg]Br)cc1", "O=Cc1cc(Cl)ccn1", "chloropicolinaldehyde", "4-MeC₆H₄MgBr"),
    ("Cc1ccc([Mg]Cl)cc1", "O=Cc1cccc2cccnc12", "quinoline_8_carbaldehyde", "4-MeC₆H₄MgCl"),
    ("CN(C)c1ccc([Mg]Br)cc1", "CNc1ccc(C(C)=O)cc1", "methylamino_acetophenone", "4-Me₂NC₆H₄MgBr"),
    ("Cc1ccc([Mg]Br)cc1", "CC(=O)Cc1c[nH]c2ccccc12", "indolyl_acetone", "4-MeC₆H₄MgBr"),
    ("COc1cccc([Mg]Br)c1", "Cc1cc(=O)n(-c2ccccc2)[nH]1", "phenyl_methyl_pyrazolone", "3-MeOC₆H₄MgBr"),
    ("COc1ccccc1[Mg]Br", "CC(=O)c1ccc(N2CCCCC2)cc1", "piperidinyl_acetophenone", "2-MeOC₆H₄MgBr"),
]


def reagent_label(fig, centre_x, baseline_y, formula):
    """Lower ordinary Arial digits at 8 pt; system Arial lacks subscript glyphs."""
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    segments = []
    for character in "+ " + formula:
        lowered = character in "₀₁₂₃₄₅₆₇₈₉"
        value = character.translate(subscripts) if lowered else character
        if segments and segments[-1][1] == lowered:
            segments[-1] = (segments[-1][0] + value, lowered)
        else:
            segments.append((value, lowered))
    from matplotlib.font_manager import FontProperties
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family="Arial", size=8)
    widths = [renderer.get_text_width_height_descent(value, font, False)[0] / fig.dpi
              for value, _ in segments]
    cursor = centre_x - sum(widths) / 2
    for (value, lowered), width in zip(segments, widths):
        fig.text(cursor / fig.get_figwidth(),
                 (baseline_y - (1.8 / 72 if lowered else 0)) / fig.get_figheight(),
                 value, ha="left", va="baseline", color=INK, fontsize=8)
        cursor += width


def matrix(frame):
    """Pairs x conditions: yield for analysed wells, -1 for missing, NaN for never run; rows/columns sorted by mean."""
    analysed = frame[frame.measurement_status.isin(ANALYSED)]
    pairs = analysed.pair_key.unique()
    frame = frame[frame.pair_key.isin(pairs)]
    row_order = analysed.groupby("pair_key").yield_fraction.mean().sort_values(ascending=False).index
    col_order = analysed.groupby("condition_id").yield_fraction.mean().sort_values(ascending=False).index
    values = analysed.pivot_table(index="pair_key", columns="condition_id", values="yield_fraction", aggfunc="mean")
    missing = frame[~frame.measurement_status.isin(ANALYSED)].groupby(["pair_key", "condition_id"]).size()
    grid = values.reindex(index=row_order, columns=col_order)
    flags = np.zeros(grid.shape, dtype=bool)
    for (pair, condition) in missing.index:
        if condition in grid.columns and np.isnan(grid.at[pair, condition]):
            flags[grid.index.get_loc(pair), grid.columns.get_loc(condition)] = True
    return grid, flags, analysed


def heatmap(ax, grid, flags, hatch=True):
    """Yield cells on the sequential scale; hatched grey for a record without a numerical yield; white for none."""
    ax.set_facecolor("white")
    ax.imshow(grid.values * 100, cmap=CMAP, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    for r, c in zip(*np.nonzero(flags)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=MISSING, lw=0,
                               edgecolor=MISSING_HATCH if hatch else MISSING, hatch="//////" if hatch else None))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRAY)
        side.set_linewidth(0.6)


def figure4():
    plt.rcParams["hatch.linewidth"] = 0.3
    table = structure().set_index("screen")
    data = records(["Grignard", "Esterification"])
    grignard, g_flags, g_analysed = matrix(data[data.screen == "Grignard"])
    ester, e_flags, _ = matrix(data[data.screen == "Esterification"])
    assert grignard.shape == (136, 47) and ester.shape == (80, 95), (grignard.shape, ester.shape)
    measured, missing = int(np.isfinite(grignard.values).sum()), int(g_flags.sum())
    assert (measured, missing, grignard.size - measured - missing) == (5361, 116, 915)

    pair_info = g_analysed.groupby("pair_key").agg(reagent=("sub_1_smiles", "first"), carbonyl=("sub_2_smiles", "first"))
    reagent_max = g_analysed.groupby("sub_1_smiles").yield_fraction.max()
    dead_reagents = set(reagent_max[reagent_max == 0].index)
    assert len(dead_reagents) == 4
    flag_carbonyl = {s: (lambda m: m.HasSubstructMatch(AMINO) or m.HasSubstructMatch(PROTIC))(Chem.MolFromSmiles(s))
                     for s in pair_info.carbonyl.unique()}
    rows = pair_info.reindex(grignard.index)
    amino_rows = rows.carbonyl.map(flag_carbonyl).values
    dead_rows = rows.reagent.isin(dead_reagents).values
    assert amino_rows.sum() == 18 and len(set(rows.carbonyl[amino_rows])) == 12
    pair_mean = g_analysed.groupby("pair_key").yield_fraction.mean()
    examples = []
    for reagent, carbonyl, asset, formula in EXAMPLES:
        match = rows.index[(rows.reagent == reagent) & (rows.carbonyl == carbonyl)]
        assert len(match) == 1 and reagent not in dead_reagents, (reagent, carbonyl)
        examples.append((grignard.index.get_loc(match[0]), 100 * pair_mean[match[0]], asset, formula, carbonyl))
    examples.sort(key=lambda e: e[0])  # number the call-outs from the top row down
    top4 = sorted(pair_mean.values, reverse=True)[:4]
    assert [round(e[1], 1) for e in examples[:4]] == [round(100 * v, 1) for v in top4]
    assert all(flag_carbonyl[e[4]] for e in examples[4:]) and not any(flag_carbonyl[e[4]] for e in examples[:4])

    fig = plt.figure(figsize=(WIDTH, 9.8))
    grid = GridSpec(3, 2, figure=fig, height_ratios=[2.4, 2.4, 1.05], width_ratios=[1, 1], hspace=0.27,
                    wspace=0.42, left=0.075, right=0.985, top=0.955, bottom=0.06)

    # a: two heatmaps with their schemes; the frame axis carries the panel title
    frame = fig.add_subplot(grid[0, :])
    style.panel_title(frame, "a  Grignard yields follow the substrate pair; esterification yields follow the conditions")
    frame.axis("off")
    fig.canvas.draw()
    box = frame.get_position()
    x0, y0, w, h = box.x0, box.y0, box.width, box.height
    hm_bottom, hm_height = y0 + 0.17 * h, 0.58 * h
    g_w, e_x, e_w = 0.40 * w, 0.585 * w, 0.415 * w
    g_ax = fig.add_axes([x0, hm_bottom, g_w, hm_height])
    e_ax = fig.add_axes([x0 + e_x, hm_bottom, e_w, hm_height])
    heatmap(g_ax, grignard, g_flags)
    heatmap(e_ax, ester, e_flags)
    for ax, rx, shape in [(g_ax, table.loc["Grignard"], grignard.shape), (e_ax, table.loc["Esterification"], ester.shape)]:
        ax.set_ylabel(f"{shape[0]} substrate pairs", labelpad=2)
        ax.set_xlabel(f"{shape[1]} conditions", labelpad=2)
        ax.text(0.5, -0.075, f"between pairs {100 * rx.pair_variance_share:.1f}%;  "
                           f"between conditions {100 * rx.condition_variance_share:.1f}%",
                ha="center", va="top", transform=ax.transAxes, color=INK)
    scheme_y = (y0 + 0.85 * h) * fig.get_figheight()
    for asset, label, centre_x in (
            ("grignard_addition", "Grignard addition", (x0 + g_w / 2) * WIDTH),
            ("esterification", "Esterification", (x0 + e_x + e_w / 2) * WIDTH)):
        chem.place(fig, asset, centre_in=(centre_x, scheme_y))
        _, scheme_height = chem.extent(asset)
        fig.text(centre_x / WIDTH, (scheme_y + scheme_height / 2 + 0.07) / fig.get_figheight(),
                 label, ha="center", va="bottom", weight="bold", color=INK)

    # strips beside the Grignard rows, then numbered call-outs to panel b
    strip_w = 0.011 * w
    for k, (mask, colour) in enumerate([(amino_rows, PURPLE), (dead_rows, INK)]):
        sax = fig.add_axes([x0 + g_w + 0.006 * w + k * (strip_w + 0.004 * w), hm_bottom, strip_w, hm_height])
        sax.imshow(np.where(mask, 1.0, np.nan)[:, None], cmap=ListedColormap([colour]), aspect="auto",
                   interpolation="nearest", vmin=0, vmax=1)
        sax.set_xlim(-0.5, 0.5)
        sax.set_xticks([])
        sax.set_yticks([])
        for side in sax.spines.values():
            side.set_color(LIGHT)
            side.set_linewidth(0.5)
    n = len(grignard)
    call = fig.add_axes([x0 + g_w + 0.04 * w, hm_bottom, 0.07 * w, hm_height])
    call.set_xlim(0, 1)
    call.set_ylim(n - 0.5, -0.5)
    call.axis("off")
    label_rows = list(np.linspace(4, 34, 4)) + list(np.linspace(n - 31, n - 1, 4))
    for k, ((row, *_), target) in enumerate(zip(examples, label_rows), start=1):
        call.plot([0.0, 0.3, 0.62], [row, row, target], color=GRAY, lw=0.5, clip_on=False)
        call.text(0.72, target, str(k), ha="left", va="center", color=INK, weight="bold")

    # keys
    key_ax = fig.add_axes([x0, y0 - 0.03 * h, w, 0.07 * h])
    key_ax.axis("off")
    keys = [(PURPLE, PURPLE, None, "amino or N–H on carbonyl"), (INK, INK, None, "reagent never gave product"),
            (MISSING, MISSING_HATCH, "//////", "no numerical yield"), ("white", GRAY, None, "no record")]
    for k, (fc, ec, hatch, label) in enumerate(keys):
        kx = (0.0, 0.245, 0.49, 0.665)[k]
        key_ax.add_patch(Rectangle((kx, 0.25), 0.014, 0.5, fc=fc, ec=ec, lw=0.5 if hatch is None else 0,
                                   hatch=hatch, transform=key_ax.transAxes, clip_on=False))
        key_ax.text(kx + 0.022, 0.5, label, va="center", transform=key_ax.transAxes, color=INK)
    cax = fig.add_axes([x0 + 0.81 * w, y0 - 0.02 * h, 0.15 * w, 0.035 * h])
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 100)), cax=cax,
                       orientation="horizontal")
    bar.set_ticks([0, 50, 100])
    bar.outline.set_linewidth(0.5)
    bar.outline.set_edgecolor(GRAY)
    cax.tick_params(labelsize=8, length=2, pad=1)
    cax.text(0.5, 1.25, "yield, %", transform=cax.transAxes, ha="center", va="bottom", color=INK, fontsize=8)

    # b: the eight selected carbonyl partners, two rows of four
    gallery = fig.add_subplot(grid[1, :])
    style.panel_title(gallery, "b  Selected carbonyl partners: amino or N–H groups give little product")
    gallery.axis("off")
    fig.canvas.draw()
    gb = gallery.get_position()
    fig_w, fig_h = WIDTH, fig.get_figheight()
    row_names = ("highest\npair means", "amino\nor N–H\npartners")
    col_w = gb.width * fig_w / 4
    left_in = gb.x0 * fig_w
    row_h = gb.height * fig_h / 2
    # structures are centred in a box as tall as the tallest one in their row, so labels share one baseline
    box_h = [max(chem.extent(e[2])[1] for e in examples[4 * r:4 * r + 4]) for r in range(2)]
    for k, (row, mean, asset, formula, _) in enumerate(examples):
        r, c = divmod(k, 4)
        cx = left_in + (c + 0.5) * col_w
        top_in = (gb.y0 + gb.height) * fig_h - r * row_h
        ex, ey = chem.extent(asset)
        if ex > col_w - 0.08 or box_h[r] + 0.49 > row_h:
            raise ValueError(f"Native structure needs more gallery space: {asset}")
        cy = top_in - 0.12 - box_h[r] / 2
        chem.place(fig, asset, centre_in=(cx, cy))
        fig.text((cx - col_w / 2 + 0.02) / fig_w, (top_in - 0.05) / fig_h, str(k + 1), ha="left", va="top",
                 weight="bold", color=INK)
        base = top_in - 0.12 - box_h[r]
        reagent_label(fig, cx, base - 0.15, formula)
        fig.text(cx / fig_w, (base - 0.30) / fig_h, f"pair mean {mean:.0f}%", ha="center",
                 va="baseline", color=INK, weight="bold")
        if c == 0:
            fig.text(0.020, (top_in - 0.45 * row_h) / fig_h, row_names[r], ha="left", va="center", color=GRAY)

    # c: R2 under the two tests
    high = structure()
    high = high[high.high]
    ax = fig.add_subplot(grid[2, 0])
    style.panel_title(ax, "c  Grignard keeps almost none of its accuracy")
    for row in high.itertuples():
        colour = {"Grignard": PURPLE, "Esterification": GREEN}.get(row.screen, LIGHT)
        lw = 2.2 if colour != LIGHT else 0.9
        ax.plot([0, 1], [row.rf_row_r2, row.rf_group_r2], color=colour, lw=lw, zorder=3 if lw > 1 else 1,
                marker="o", ms=3.5 if lw > 1 else 2.5)
        if row.screen in ("Grignard", "Esterification"):
            ax.text(1.06, row.rf_group_r2, f"{name(row.screen)}  {row.rf_row_r2:.2f} → {row.rf_group_r2:.2f}",
                    ha="left", va="center", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Held-out wells", "New pairs"])
    ax.set_xlim(-0.15, 2.2)
    ax.set_ylim(-0.05, 0.8)
    ax.set_ylabel("R²")
    despine(ax)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.text(1.06, 0.72, "grey: the other\nnine arrays", ha="left", va="top", color=GRAY)

    # d: both substrates seen, error still rises with the pair share
    ax = fig.add_subplot(grid[2, 1])
    style.panel_title(ax, "d  Seen substrates do not rescue Grignard")
    rho_seen = load_json(DIAG / "rf_ranking_structure_summary.json")[
        "pair_share_vs_both_seen_relative_mae_increase_rho"]["mean_yield_at_least_5pct"]
    for row in high.itertuples():
        colour = {"Grignard": PURPLE, "Esterification": GREEN}.get(row.screen, MID)
        ax.scatter(100 * row.pair_variance_share, 100 * row.both_seen_relative_mae_increase, s=26, color=colour,
                   edgecolor="white", linewidths=0.5, zorder=3)
    g, e = high.set_index("screen").loc["Grignard"], high.set_index("screen").loc["Esterification"]
    ax.annotate(f"Grignard, error ×{1 + g.both_seen_relative_mae_increase:.1f}",
                (100 * g.pair_variance_share, 100 * g.both_seen_relative_mae_increase), xytext=(0, -8),
                textcoords="offset points", ha="right", va="top", color=INK)
    ax.annotate("Esterification", (100 * e.pair_variance_share, 100 * e.both_seen_relative_mae_increase),
                xytext=(6, -5), textcoords="offset points", ha="left", va="top", color=INK)
    ax.text(22, 118, f"wells whose two substrates were\nboth in training; rank correlation {rho_seen:.2f}",
            ha="left", va="top", color=INK)
    ax.set_xlim(20, 75)
    ax.set_ylim(0, 120)
    ax.set_xlabel("Yield variance between pairs, %")
    ax.set_ylabel("Error increase, %")
    despine(ax)
    save(fig, "Figure_4")


LEADER = dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkB=3)


def points(ax, frame, x):
    for row in frame.itertuples():
        ax.scatter(getattr(row, x), row.rf_delta_r2, **marker_style(row))


def label(ax, frame, x, screen, offset, ha):
    row = frame.set_index("screen").loc[screen]
    ax.annotate(name(screen), (row[x], row.rf_delta_r2), xytext=offset, textcoords="offset points", ha=ha,
                va="center", color=INK)


def figure5():
    table = structure()
    high = table[table.high].copy()
    summary = load_json(DIAG / "rf_ranking_structure_summary.json")
    high["combined"] = high.pair_variance_share.rank() + high.substrate_levels_per_pair.rank()
    r_comb = rho(high.combined, high.rf_delta_r2)
    expected = summary["two_design_properties"]["mean_yield_at_least_5pct"]["rho_rank_sum_vs_rf_delta_r2"]
    assert abs(r_comb - expected) < 1e-9, (r_comb, expected)
    assert len(high) == 11

    fig = plt.figure(figsize=(WIDTH, 3.6))
    ax = fig.add_subplot(GridSpec(1, 1, figure=fig, left=0.085, right=0.615, bottom=0.14, top=0.86)[0, 0])
    style.panel_title(ax, "Two table properties anticipate the accuracy lost on new pairs")
    area = lambda loss: 1000 * loss  # noqa: E731  marker area in points^2, proportional to the loss
    for row in high.sort_values("rf_delta_r2", ascending=False).itertuples():
        colour = HIGHLIGHT.get(row.screen, MID)
        ax.scatter(100 * row.pair_variance_share, row.substrate_levels_per_pair, s=area(row.rf_delta_r2),
                   color=colour, alpha=0.85 if row.screen in HIGHLIGHT else 0.55, edgecolor="white",
                   linewidths=0.6, zorder=3)
    offsets = {"Grignard": (0, 15, "center"), "Esterification": (10, 10, "center"),
               "Suzuki Bpin": (0, 14, "center"), "Sonogashira": (-9, 6, "right"), "Ullmann": (8, -6, "left"),
               "Kumada": (9, 0, "left"), "Heck": (7, 0, "left"), "Acylation": (7, 0, "left"),
               "Suzuki": (7, -2, "left"), "Buchwald-Hartwig": (0, -11, "center"),
               "Reductive-amination": (0, 12, "center")}
    for row in high.itertuples():
        dx, dy, ha = offsets[row.screen]
        bold = "bold" if row.screen in HIGHLIGHT else "normal"
        ax.annotate(name(row.screen), (100 * row.pair_variance_share, row.substrate_levels_per_pair),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=INK, weight=bold)
    ax.text(0.02, 0.03, f"rank correlation of the summed ranks with the loss: {r_comb:.2f} (11 arrays)",
            transform=ax.transAxes, ha="left", va="bottom", color=INK)
    ax.set_xlim(18, 78)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Between-pair share of yield variance, %")
    ax.set_ylabel("Pairing sparsity")
    despine(ax)

    # right column: how to read both quantities from the yield table, and the size key
    x0 = 0.665
    fig.text(x0, 0.80, "From the yield table, before modelling", weight="bold", color=INK, ha="left", va="baseline",
             fontsize=8)
    icon = 0.0135  # icon cell, figure-width units

    def cells(left, bottom, shades):
        h = icon * WIDTH / 3.6
        for r, row in enumerate(shades):
            for c, shade in enumerate(row):
                if shade is None:
                    fig.patches.append(Rectangle((left + c * icon, bottom + (len(shades) - 1 - r) * h), icon * 0.88,
                                                 h * 0.88, transform=fig.transFigure, facecolor="white",
                                                 edgecolor=LIGHT, lw=0.4))
                else:
                    fig.patches.append(Rectangle((left + c * icon, bottom + (len(shades) - 1 - r) * h), icon * 0.88,
                                                 h * 0.88, transform=fig.transFigure, facecolor=shade, lw=0))

    # between-pair share: rows (pairs) differ in their mean, columns barely matter
    dark, mid, pale = "#55606a", "#a4acb2", "#dde1e4"
    cells(x0, 0.64, [[dark] * 5, [mid] * 5, [pale] * 5, [mid] * 5])
    fig.text(x0 + 6.2 * icon, 0.735, "Between-pair share", ha="left", va="baseline", color=INK, fontsize=8)
    fig.text(x0 + 6.2 * icon, 0.665, r"$\Sigma_p\, n_p(\bar{y}_p-\bar{y})^2 \,/\, \Sigma_i\, (y_i-\bar{y})^2$",
             ha="left", va="baseline", color=INK, fontsize=8)
    fig.text(x0, 0.585, "how much of the yield follows the pair", ha="left", va="baseline", color=GRAY, fontsize=8)
    # pairing sparsity: substrate 1 x substrate 2 grid, filled cells are measured pairs
    f = dark
    cells(x0, 0.395, [[f, None, None, None, f], [None, f, None, None, None], [None, None, f, f, None],
                      [None, None, None, None, f]])
    fig.text(x0 + 6.2 * icon, 0.49, "Pairing sparsity", ha="left", va="baseline", color=INK, fontsize=8)
    fig.text(x0 + 6.2 * icon, 0.42, r"$(n_1+n_2-1)\,/\,n_{\mathrm{pairs}}$", ha="left", va="baseline",
             color=INK, fontsize=8)
    fig.text(x0, 0.34, "substrate levels per measured pair;", ha="left", va="baseline", color=GRAY, fontsize=8)
    fig.text(x0, 0.29, "high when substrates have few partners", ha="left", va="baseline", color=GRAY, fontsize=8)
    # size key
    fig.text(x0, 0.19, "R² lost on new pairs", ha="left", va="baseline", color=INK, fontsize=8)
    key = fig.add_axes([x0, 0.02, 0.3, 0.14])
    key.axis("off")
    key.set_xlim(0, 3)
    key.set_ylim(0, 1)
    for k, loss in enumerate((0.1, 0.3, 0.6)):
        key.scatter(0.25 + k * 0.95, 0.55, s=area(loss), color=MID, alpha=0.55, edgecolor="white", linewidths=0.6,
                    clip_on=False)
        key.text(0.25 + k * 0.95 + 0.32, 0.55, f"{loss:.1f}", ha="left", va="center", color=INK)
    save(fig, "Figure_5")


def toc():
    table = structure().set_index("screen")
    data = records(["Grignard", "Esterification"])
    grignard, g_flags, _ = matrix(data[data.screen == "Grignard"])
    ester, e_flags, _ = matrix(data[data.screen == "Esterification"])
    fig = plt.figure(figsize=(3.5, 3.0))
    fig.text(0.5, 0.955, "Read the yield table before modelling", ha="center", va="top", fontsize=9,
             weight="bold", color=INK)
    for x0, grid, flags, screen, colour, verdict in [
            (0.06, grignard, g_flags, "Grignard", PURPLE, "set by the pair:\nnew pairs not predicted"),
            (0.54, ester, e_flags, "Esterification", GREEN, "more set by conditions:\nmost accuracy kept")]:
        ax = fig.add_axes([x0, 0.36, 0.4, 0.47])
        heatmap(ax, grid, flags)
        for side in ax.spines.values():
            side.set_color(colour)
            side.set_linewidth(1.6)
        row = table.loc[screen]
        name = "Grignard addition" if screen == "Grignard" else "Esterification"
        fig.text(x0 + 0.2, 0.855, name, ha="center", va="bottom", fontsize=8, weight="bold", color=INK)
        fig.text(x0 + 0.2, 0.33, f"{100 * row.pair_variance_share:.0f}% of yield variance\nbetween pairs",
                 ha="center", va="top", fontsize=8, color=INK)
        fig.text(x0 + 0.2, 0.2, f"R² {row.rf_row_r2:.2f} → {row.rf_group_r2:.2f}", ha="center", va="top",
                 fontsize=8, weight="bold", color=INK)
        fig.text(x0 + 0.2, 0.12, verdict, ha="center", va="top", fontsize=8, color=GRAY)
    style.check_text_bounds(fig)
    fig.savefig(OUT / "TOC.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "TOC.png", dpi=300)
    print("wrote", OUT / "TOC.pdf")

def svg_to_pdf_png(svg_path: Path):
    """Convert the Figure 1 SVG to PDF with Arial embedded, and render a PNG from the PDF."""
    import pymupdf
    from reportlab.graphics import renderPDF
    from svglib import fonts as svg_fonts
    from svglib.svglib import svg2rlg

    faces = (("normal", "normal", "arial.ttf", "Arial"), ("bold", "normal", "arialbd.ttf", "Arial-Bold"),
             ("normal", "italic", "ariali.ttf", "Arial-Italic"))
    for weight, slant, file, registered in faces:
        candidates = [Path("C:/Windows/Fonts") / file, Path("/usr/share/fonts/truetype/msttcorefonts") / file,
                      Path("/Library/Fonts") / file]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Arial font file {file} is required to convert Figure 1")
        name, exact = svg_fonts.register_font("Arial", font_path=str(path), weight=weight, style=slant,
                                              rlgFontName=registered)
        if name != registered:
            raise RuntimeError(f"Could not register {file} for Figure 1")
    drawing = svg2rlg(str(svg_path))
    # ReportLab initializes both its canvas and rendering state from the drawing.
    drawing.initialFontName = "Arial"
    drawing.initialFontSize = 8
    pdf_path = svg_path.with_suffix(".pdf")
    renderPDF.drawToFile(drawing, str(pdf_path), invariant=1)
    pymupdf.open(pdf_path)[0].get_pixmap(dpi=style.PNG_DPI).save(svg_path.with_suffix(".png"))


def main():
    global PACKAGE, OUTPUT, DIAG, OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package-dir", type=Path, default=PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    PACKAGE = args.package_dir.resolve()
    OUTPUT = PACKAGE / "output"
    DIAG = OUTPUT / "p4_unified_rf" / "diagnostics"
    OUT = (args.output_dir or PACKAGE / "figures").resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    svg_to_pdf_png(OUT / "Figure_1.svg")
    for draw in (figure2, figure3, figure4, figure5, toc):
        draw()
    print(f"Saved Figures 1-5 and the graphical abstract in {OUT}")


if __name__ == "__main__":
    main()
