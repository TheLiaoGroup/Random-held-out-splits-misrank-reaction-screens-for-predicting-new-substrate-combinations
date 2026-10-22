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
    style.finalize(fig, figure_name=stem)
    if stem == "Figure_2":
        for label in fig.texts:
            if label.get_text() == "C":
                x, y = label.get_position()
                label.set_position((x, y + 10.8 / (72 * fig.get_figheight())))
        style.check_text_bounds(fig)
    if stem == "Figure_3":
        for label in fig.texts:
            if label.get_text() in ("C", "D"):
                x, y = label.get_position()
                label.set_position((x, y + 9.6 / (72 * fig.get_figheight())))
        style.check_text_bounds(fig)
    if stem == "Figure_4":
        for ax in fig.axes:
            for label in ax.texts:
                if label.get_gid() == "pale-yield-number":
                    label.set_color("#4d565c")
    if stem == "TOC":
        scores = [text for ax in fig.axes for text in ax.texts if text.get_gid() == "toc-score"]
        if len(scores) != 2:
            raise ValueError("The graphical abstract must have exactly two score labels")
        for score in scores:
            score.set_fontsize(16)
        style.check_text_bounds(fig)
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
W, H = 510, 214


def text(x, y, s, size=8, weight="normal", anchor="start", fill=INK, style=""):
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}"{st}>{escape(s)}</text>')


def svg_reagent_label(x, y, pieces):
    """Use positioned Arial digits because the system font lacks Unicode subscripts."""
    from matplotlib.font_manager import FontProperties, findfont
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_name = "ChemArialMetrics"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, findfont(FontProperties(family="Arial"), fallback_to_default=False)))
    widths = [pdfmetrics.stringWidth(token, font_name, 8) for token, _ in pieces]
    cursor = x - sum(widths) / 2
    spans = []
    for (token, subscript), width in zip(pieces, widths):
        offset = 1.8 if subscript else 0
        spans.append(f'<tspan x="{cursor:.4f}" y="{y + offset:.1f}">{escape(token)}</tspan>')
        cursor += width
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="8" '
            f'text-anchor="start" fill="{INK}">' + "".join(spans) + '</text>')


def lines(x, y, rows, size=8, gap=9.6, **kw):
    return "".join(text(x, y + i * gap, r, size=size, **kw) for i, r in enumerate(rows))


def rect(x, y, w, h, fill="none", stroke="none", sw=0.6, dash="", rx=0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}" rx="{rx}" fill="{fill}" '
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


OVERVIEW_PAIRS = (
    "Grignard-pair-00040", "Grignard-pair-00019", "Grignard-pair-00042", "Grignard-pair-00045",
    "Grignard-pair-00064", "Grignard-pair-00094", "Grignard-pair-00138", "Grignard-pair-00143",
)
OVERVIEW_CONDITIONS = ("C38", "C14", "C46", "C25", "C27", "C48", "C47", "C24", "C12", "C6", "C31", "C8")
OVERVIEW_CMAP = LinearSegmentedColormap.from_list("overview", ["#f1f5f9", "#c6d5e6", "#5d86b0", "#1d3557"])


def overview_data():
    """The approved 8-pair, 12-condition illustration, with full-pair means kept distinct from block means."""
    grid, flags, analysed = matrix(records(["Grignard"]))
    assert grid.shape == (136, 47) and len(analysed) == 5361
    row_ids = grid.index.get_indexer(OVERVIEW_PAIRS)
    col_ids = grid.columns.get_indexer(OVERVIEW_CONDITIONS)
    assert (row_ids >= 0).all() and (col_ids >= 0).all()
    values = grid.values[np.ix_(row_ids, col_ids)]
    missing = flags[np.ix_(row_ids, col_ids)]
    means = analysed.groupby("pair_key").yield_fraction.mean().reindex(OVERVIEW_PAIRS)
    assert abs(means.iloc[0] * 100 - 75.0293617021) < 1e-6 and means.iloc[-1] == 0
    assert np.isfinite(values).sum() == 95 and missing.sum() == 0
    assert not np.isfinite(values[3, 5])  # Pair 00045, condition C48 has no source record.
    return values, missing, means


def overview_colour(value, missing=False):
    if missing:
        return "#e4e7ea"
    return matplotlib.colors.to_hex(OVERVIEW_CMAP(value)) if np.isfinite(value) else "#ffffff"


def overview_svg_cell(x, y, width, height, value, missing=False):
    body = rect(x, y, width, height, fill=overview_colour(value, missing),
                stroke=SVG_LIGHT if not np.isfinite(value) else "none", sw=0.5)
    if missing:
        for offset in np.arange(-height, width, 3):
            start, end = max(0, -offset), min(height, width - offset)
            if end > start:
                body += line(x + offset + start, y + height - start,
                             x + offset + end, y + height - end, stroke=MID, sw=0.5, cap="butt")
    return body


def figure1():
    """The approved V5 layout, populated from the measured cells and unrounded array scores."""
    table = structure()
    high = table[table.high].copy()
    high["row_rank"] = high.rf_row_r2.rank(ascending=False).astype(int)
    high["group_rank"] = high.rf_group_r2.rank(ascending=False).astype(int)
    ranks = high.set_index("screen")
    rho11 = high[["rf_row_r2", "rf_group_r2"]].corr(method="spearman").iloc[0, 1]
    assert int(table.numeric_records.sum()) == 378070 and len(table) == 19
    assert (ranks.loc["Grignard", "row_rank"], ranks.loc["Grignard", "group_rank"]) == (2, 11)
    assert (ranks.loc["Buchwald-Hartwig", "row_rank"], ranks.loc["Buchwald-Hartwig", "group_rank"]) == (11, 4)
    grignard = ranks.loc["Grignard"]
    values, missing, means = overview_data()

    def title(x, y, letter, words):
        return text(x, y, letter, size=9, weight="bold") + text(x + 11, y, words, size=9, weight="bold")

    parts = []
    # A: both branches begin at the same highlighted first row.
    body = title(2, 11, "A", "One pair, two tests")
    tx, ty, cell, cell_size = 86, 38, 7.4, 6.6
    cells = ""
    for i in range(8):
        for j in range(12):
            cells += overview_svg_cell(tx + j * cell, ty + i * cell, cell_size, cell_size,
                                       values[i, j], missing[i, j])
    body += group("A-measured-cells", cells)
    body += rect(tx - 1.2, ty - 1.2, 90.4, 9.0, stroke=PURPLE, sw=1.1)
    body += text(tx, 110.2, "Grignard yields, 8 of 136 pairs", fill=SVG_GREY)
    first_y, last_y = ty + cell_size / 2, ty + 7 * cell + cell_size / 2
    molecules = chem.svg_group("chloronicotinaldehyde", 12.5, 12.575)
    molecules += svg_reagent_label(44, 69, (("+ 4-FC", False), ("6", True), ("H", False),
                                           ("4", True), ("MgBr", False)))
    molecules += text(44, 79.5, f"pair mean {means.iloc[0] * 100:.0f}%", anchor="middle")
    molecules += line(79.5, first_y, tx - 3, first_y, sw=0.5)
    low_height = chem.extent("methylamino_acetophenone")[1] * 72
    molecules += chem.svg_group("methylamino_acetophenone", 0.5, last_y - low_height / 2)
    molecules += svg_reagent_label(44, 121, (("+ 4-Me", False), ("2", True), ("NC", False),
                                   ("6", True), ("H", False), ("4", True), ("MgBr", False)))
    molecules += text(44, 131.5, f"pair mean {means.iloc[-1] * 100:.0f}%", anchor="middle")
    molecules += line(79.5, last_y, tx - 3, last_y, sw=0.5)
    body += group("A-native-chemistry", molecules)
    for y, value, flagged, label in ((144, 0, False, "0% yield"),
                                    (157, np.nan, False, "No record")):
        body += overview_svg_cell(tx, y, 7, 7, value, flagged)
        body += text(tx + 11, y + 6.5, label, fill=SVG_GREY)

    sx, step = 201, 7.4
    branches = arrow(tx + 12 * cell + 2, first_y, sx - 4, 48, c1=(sx - 12, 48))
    branches += arrow(tx + 12 * cell + 2, first_y, sx - 4, 92, c1=(sx - 12, 92))
    branches += text(sx, 37, "Held-out wells", weight="bold")
    for j, value in enumerate(values[0]):
        hidden = j in (2, 6, 9)
        branches += rect(sx + j * step + (0.6 if hidden else 0), 43 + (0.6 if hidden else 0),
                         5.6 if hidden else 6.6, 8.8 if hidden else 10,
                         fill="white" if hidden else overview_colour(value),
                         stroke=TEAL if hidden else "none", sw=1.0)
    branches += text(sx, 81, "New pairs", weight="bold")
    branches += rect(sx + 0.6, 87.6, 87.4, 8.8, fill="white", stroke=ORANGE, sw=1.2)
    branches += text(sx + 44.4, 95, "?", anchor="middle", weight="bold")
    body += group("A-two-test-branches", branches)
    scores = text(sx, 119, "Grignard array, R²", weight="bold")
    score_scale = 76.0
    for y, label, score, colour in ((138, "Held-out wells", grignard.rf_row_r2, TEAL),
                                     (164, "New pairs", grignard.rf_group_r2, ORANGE)):
        width = float(score) * score_scale
        scores += line(sx, y - 1, sx, y + 9, sw=0.5)
        scores += text(sx, y - 4, label, fill=SVG_GREY)
        scores += group("A-score-" + ("held-out" if colour == TEAL else "new-pairs"),
                        rect(sx, y, width, 8, fill=colour))
        scores += text(sx + width + 4, y + 7, f"{score:.2f}", size=9, weight="bold")
    body += group("A-common-zero-scores", scores)
    parts.append(group("A-one-pair-two-tests", body))

    # B: rankings use the five-seed mean scores of all 11 higher-yield arrays.
    lx0, lx1, ly0, step = 388, 412, 41, 12.2
    body = title(280, 11, "B", "The ranking of reactions flips")
    body += text(lx0 - 4, 29, "held-out", anchor="end", fill=SVG_GREY)
    body += text(lx1 + 4, 29, "new pairs", fill=SVG_GREY)
    order = sorted(ranks.index, key=lambda s: (s in ("Grignard", "Buchwald-Hartwig"), s))
    for screen in order:
        r, g = ranks.loc[screen, "row_rank"], ranks.loc[screen, "group_rank"]
        stroke, sw = SVG_LIGHT, 0.7
        if screen == "Grignard":
            stroke, sw = PURPLE, 2.0
        elif screen == "Buchwald-Hartwig":
            stroke, sw = INK, 1.4
        body += line(lx0, ly0 + (r - 1) * step, lx1, ly0 + (g - 1) * step, stroke=stroke, sw=sw)
        emphasized = screen in ("Grignard", "Buchwald-Hartwig")
        left_name, right_name = name(screen), name(screen)
        if emphasized:
            ordinal = {2: "2nd", 4: "4th", 11: "11th"}
            left_name += " " + ordinal[r]
            right_name = ordinal[g] + " " + right_name
        body += text(lx0 - 7, ly0 + (r - 1) * step + 3, left_name, anchor="end",
                     weight="bold" if emphasized else "normal", fill=INK if emphasized else SVG_GREY)
        body += text(lx1 + 7, ly0 + (g - 1) * step + 3, right_name,
                     weight="bold" if emphasized else "normal", fill=INK if emphasized else SVG_GREY)
    for k in range(11):
        for x in (lx0, lx1):
            body += f'<circle cx="{x}" cy="{ly0 + k * step:.1f}" r="1.5" fill="{SVG_GREY}"/>'
    body += text(400, 183, f"ρ = {rho11:.2f}", size=9, anchor="middle", weight="bold")
    parts.append(group("B-ranks", body))

    # C: compact visual pointers to the two full-table properties in Figure 5.
    body = title(2, 202, "C", "Read before modelling:")
    dark, mid, pale = "#55606a", "#a4acb2", "#dde1e4"
    ix, iy, ic = 140, 191, 3.6
    for i, shade in enumerate((dark, mid, pale, mid)):
        for j in range(5):
            body += rect(ix + j * ic, iy + i * ic, 3.1, 3.1, fill=shade)
    body += text(163, 202, "between-pair share")
    ix, iy2 = 252, 191
    filled = {(0, 0), (0, 4), (1, 1), (2, 2), (2, 3), (3, 4)}
    for i in range(4):
        for j in range(5):
            body += rect(ix + j * ic, iy2 + i * ic, 3.1, 3.1, fill=dark if (i, j) in filled else "white",
                         stroke="none" if (i, j) in filled else SVG_LIGHT, sw=0.5)
    body += text(275, 202, "pairing sparsity")
    body += text(352, 202, "(Figure 5)", fill=SVG_GREY)
    parts.append(group("C-properties", body))

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{180 * H / W:.12g}mm" '
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
    style.panel_title(ax, "A  Grignard falls from 2nd to 11th")
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
    ax.text(0.5, 12.25, f"ρ = {rho11:.2f}", ha="center", va="top", color=INK, weight="bold")
    assert reversals["pairs"] == 55
    ax.set_xlim(-1.75, 2.05)
    ax.set_ylim(13.4, -2.25)
    ax.axis("off")

    # b: held-out vs new R2, all 19 arrays
    ax = fig.add_subplot(grid[0, 1])
    style.panel_title(ax, "B  Every array loses accuracy on new pairs")
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
    ax.text(-0.27, 0.8, f"{losses_all}/{len(rf_seed)} below y = x", ha="left", va="top", color=INK, weight="bold")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("R² on held-out wells")
    ax.set_ylabel("R² on new pairs")
    despine(ax)

    # c: picking arrays by either score
    ax = fig.add_subplot(grid[1, 1])
    style.panel_title(ax, "C  The top three by held-out wells\nfall below a random pick")
    ks = [r["k"] for r in selection]
    by_row = [r["group_r2_mean_row_selected"] for r in selection]
    by_group = [r["group_r2_mean_group_selected"] for r in selection]
    ax.axhline(random_pick, color=GRAY, ls=(0, (3, 2)), lw=0.7)
    ax.plot(ks, by_group, color=ORANGE, lw=1.6, marker="o", ms=3.2)
    ax.plot(ks, by_row, color=TEAL, lw=1.6, marker="o", ms=3.2)
    ax.text(5.3, 0.475, "by new-pair R²", color=INK, ha="left", va="bottom")
    ax.text(4.4, 0.222, "by held-out-well R²", color=INK, ha="left", va="bottom")
    ax.text(11, random_pick - 0.014, f"random, {random_pick:.2f}", color=INK, ha="right", va="top")
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

    fig = plt.figure(figsize=(WIDTH, 6.4))
    grid = GridSpec(3, 2, figure=fig, width_ratios=[0.95, 1], height_ratios=[0.8, 1.1, 1.1], wspace=0.42,
                    hspace=0.62, left=0.215, right=0.985, top=0.93, bottom=0.07)

    # a: per-array dumbbells (hero)
    ax = fig.add_subplot(grid[:, 0])
    style.panel_title(ax, "A  New pairs lose mainly the pair mean")
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
    ax.text(-0.3, -1.05, "between-pair share", ha="right", va="bottom", color=GRAY)
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
    style.panel_title(ax, "B  Three models all lose the pair mean")
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
    style.panel_title(ax, "C  Most held-out-well accuracy\nneeds no molecular representation")
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
    g = high.set_index("screen").loc["Grignard"]
    ax.annotate("Grignard", (g.rf_row_r2, g.row_sister_well_mean_r2), xytext=(0.72, 0.62), textcoords="data",
                ha="center", va="center", color=INK, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkB=3))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Random forest R², held-out wells")
    ax.set_ylabel("Reference R², held-out wells")
    despine(ax)

    # d: wells whose two substrates were both in training still lose accuracy, most where the pair sets the yield
    ax = fig.add_subplot(grid[2, 1])
    style.panel_title(ax, "D  Familiar substrates still lose accuracy,\nmost where the pair sets the yield")
    rho_seen = load_json(DIAG / "rf_ranking_structure_summary.json")[
        "pair_share_vs_both_seen_relative_mae_increase_rho"]["mean_yield_at_least_5pct"]
    for row in high.itertuples():
        colour = HIGHLIGHT.get(row.screen, MID)
        ax.scatter(100 * row.pair_variance_share, 100 * row.both_seen_relative_mae_increase, s=26, color=colour,
                   edgecolor="white", linewidths=0.5, zorder=3)
    g, e = high.set_index("screen").loc["Grignard"], high.set_index("screen").loc["Esterification"]
    ax.annotate(f"Grignard, error ×{1 + g.both_seen_relative_mae_increase:.1f}",
                (100 * g.pair_variance_share, 100 * g.both_seen_relative_mae_increase), xytext=(-6, -2),
                textcoords="offset points", ha="right", va="center", color=INK)
    ax.annotate("Esterification", (100 * e.pair_variance_share, 100 * e.both_seen_relative_mae_increase),
                xytext=(6, -5), textcoords="offset points", ha="left", va="top", color=INK)
    ax.text(22, 118, f"ρ = {rho_seen:.2f}", ha="left", va="top", color=INK, weight="bold")
    ax.set_xlim(20, 75)
    ax.set_ylim(0, 120)
    ax.set_xlabel("Between-pair share of yield variance, %")
    ax.set_ylabel("Error increase, %")
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


BLOCK_CMAP = LinearSegmentedColormap.from_list("block", ["#ffffff", "#c6d5e6", "#5d86b0", "#1d3557"])
BLOCK_ROWS, BLOCK_COLS = 16, 12


def shares(values):
    """Between-row and between-column shares of the sum of squares over the measured cells of a block."""
    mean = np.nanmean(values)
    total = np.nansum((values - mean) ** 2)
    rows = sum(np.isfinite(r).sum() * (np.nanmean(r) - mean) ** 2 for r in values if np.isfinite(r).any())
    cols = sum(np.isfinite(c).sum() * (np.nanmean(c) - mean) ** 2 for c in values.T if np.isfinite(c).any())
    return rows / total, cols / total


def pick_block(grid):
    """16 pairs evenly spaced by pair mean and 12 conditions evenly spaced by condition mean among the 60% of
    conditions measured most often; the full table is already ordered by both means."""
    rows = [int(round(i)) for i in np.linspace(0, grid.shape[0] - 1, BLOCK_ROWS)]
    coverage = np.isfinite(grid.values).mean(0)
    candidates = [c for c in range(grid.shape[1]) if coverage[c] >= np.quantile(coverage, 0.4)]
    cols = [candidates[int(round(i))] for i in np.linspace(0, len(candidates) - 1, BLOCK_COLS)]
    return rows, cols


def block_panel(fig, frame, grid, flags, rows, cols, colour, scheme, name_, table_row, row_labels=None):
    """A reaction scheme over a numbered yield block with row-mean and column-mean margins."""
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    box = frame.get_position()
    left, top = box.x0 * fig_w, (box.y0 + box.height) * fig_h  # inches
    cell_w, cell_h = 0.2, 0.16
    bx, by_top = left + 0.42, top - 0.76
    bw, bh = BLOCK_COLS * cell_w, BLOCK_ROWS * cell_h
    values = grid.values[np.ix_(rows, cols)] * 100
    missing = flags[np.ix_(rows, cols)]
    # scheme header
    sw, sh = chem.extent(scheme)
    chem.place(fig, scheme, centre_in=(bx + sw / 2 - 0.05, top - 0.12 - sh / 2))
    # block
    ax = fig.add_axes([bx / fig_w, (by_top - bh) / fig_h, bw / fig_w, bh / fig_h])
    ax.imshow(values, cmap=BLOCK_CMAP, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    for r in range(BLOCK_ROWS):
        for c in range(BLOCK_COLS):
            v = values[r, c]
            if missing[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=MISSING, edgecolor=MISSING_HATCH, lw=0,
                                       hatch="//////"))
            elif not np.isfinite(v):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="white", lw=0))
            else:
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", color="white" if v >= 60 else GRAY,
                        gid=None if v >= 60 else "pale-yield-number")
    ax.set_xticks(np.arange(-0.5, BLOCK_COLS), minor=True)
    ax.set_yticks(np.arange(-0.5, BLOCK_ROWS), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_ylabel(f"{BLOCK_ROWS} of {grid.shape[0]} substrate pairs", labelpad=12 if row_labels else 3)
    if row_labels:
        for r, label in row_labels.items():
            ax.text(-0.75, r, label, ha="right", va="center", color=INK, weight="bold")
    # margins: row means at right, column means below, in the reaction's colour
    row_mean, col_mean = np.nanmean(values, 1), np.nanmean(values, 0)
    rx = fig.add_axes([(bx + bw + 0.05) / fig_w, (by_top - bh) / fig_h, 0.38 / fig_w, bh / fig_h])
    rx.barh(range(BLOCK_ROWS), row_mean, height=0.7, color=colour)
    rx.set_ylim(BLOCK_ROWS - 0.5, -0.5)
    rx.set_xlim(0, 100)
    rx.axis("off")
    rx.text(0, -0.75, "pair mean", ha="left", va="bottom", color=GRAY)
    cx = fig.add_axes([bx / fig_w, (by_top - bh - 0.3) / fig_h, bw / fig_w, 0.26 / fig_h])
    cx.bar(range(BLOCK_COLS), col_mean, width=0.7, color=colour)
    cx.set_xlim(-0.5, BLOCK_COLS - 0.5)
    cx.set_ylim(100, 0)
    cx.axis("off")
    cx.text(BLOCK_COLS - 0.3, 50, "condition" + chr(10) + "mean", ha="left", va="center", color=GRAY)
    pair_share, condition_share = 100 * table_row.pair_variance_share, 100 * table_row.condition_variance_share
    fig.text((bx + bw / 2) / fig_w, (by_top - bh - 0.36) / fig_h, f"{BLOCK_COLS} of {grid.shape[1]} conditions",
             ha="center", va="top", color=INK)
    fig.text((bx + bw / 2) / fig_w, (by_top - bh - 0.56) / fig_h,
             f"full table: between pairs {pair_share:.1f}%, between conditions {condition_share:.1f}%",
             ha="center", va="top", color=INK, weight="bold")
    return shares(values / 100)


def productive_grignard_pairs(frame):
    """Equal-weight measured pair means after excluding only all-zero reagents."""
    numeric = frame[(frame.screen == "Grignard") & frame.measurement_status.isin(ANALYSED)]
    reagent_max = numeric.groupby("sub_1_smiles").yield_fraction.max()
    productive = set(reagent_max[reagent_max > 0].index)
    pairs = numeric[numeric.sub_1_smiles.isin(productive)].groupby("pair_key").agg(
        reagent=("sub_1_smiles", "first"), carbonyl=("sub_2_smiles", "first"),
        mean_percent=("yield_fraction", lambda values: 100 * values.mean()))
    classification = {}
    for smiles in pairs.carbonyl.unique():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid carbonyl partner SMILES: {smiles}")
        classification[smiles] = molecule.HasSubstructMatch(AMINO) or molecule.HasSubstructMatch(PROTIC)
    pairs["amino_or_protic"] = pairs.carbonyl.map(classification)
    assert len(pairs) == 123 and pairs.reagent.nunique() == 19
    assert pairs.amino_or_protic.sum() == 12 and (~pairs.amino_or_protic).sum() == 111
    return pairs


def stacked_pair_positions(ax, values, row, separation_pt=3.5):
    """Greedily stack at fixed x values, with collision distances measured in print points."""
    transform = ax.transData
    px_per_pt = ax.figure.dpi / 72
    y_per_pt = separation_pt * px_per_pt / (transform.transform((0, 1))[1] - transform.transform((0, 0))[1])
    placed, offsets = [], np.zeros(len(values))
    for index in np.argsort(values, kind="stable"):
        x = transform.transform((values[index], row))[0] / px_per_pt
        level = 0
        while True:
            offset = (0 if level == 0 else ((level + 1) // 2) * (1 if level % 2 else -1))
            if all((x - other_x) ** 2 + ((offset - other_y) * separation_pt) ** 2
                   >= separation_pt ** 2 - 1e-9 for other_x, other_y in placed):
                break
            level += 1
        placed.append((x, offset))
        offsets[index] = offset * y_per_pt
    return row + offsets


def figure4():
    plt.rcParams["hatch.linewidth"] = 0.3
    table = structure().set_index("screen")
    data = records(["Grignard", "Esterification"])
    grignard, g_flags, g_analysed = matrix(data[data.screen == "Grignard"])
    ester, e_flags, _ = matrix(data[data.screen == "Esterification"])
    assert grignard.shape == (136, 47) and ester.shape == (80, 95), (grignard.shape, ester.shape)

    pair_info = g_analysed.groupby("pair_key").agg(reagent=("sub_1_smiles", "first"), carbonyl=("sub_2_smiles", "first"))
    reagent_max = g_analysed.groupby("sub_1_smiles").yield_fraction.max()
    dead_reagents = set(reagent_max[reagent_max == 0].index)
    assert len(dead_reagents) == 4
    flag_carbonyl = {s: (lambda m: m.HasSubstructMatch(AMINO) or m.HasSubstructMatch(PROTIC))(Chem.MolFromSmiles(s))
                     for s in pair_info.carbonyl.unique()}
    rows = pair_info.reindex(grignard.index)
    pair_mean = g_analysed.groupby("pair_key").yield_fraction.mean()
    examples = []
    for reagent, carbonyl, asset, formula in EXAMPLES:
        match = rows.index[(rows.reagent == reagent) & (rows.carbonyl == carbonyl)]
        assert len(match) == 1 and reagent not in dead_reagents, (reagent, carbonyl)
        examples.append((grignard.index.get_loc(match[0]), 100 * pair_mean[match[0]], asset, formula, carbonyl))
    examples.sort(key=lambda e: e[0])  # number from the top row down
    top4 = sorted(pair_mean.values, reverse=True)[:4]
    assert [round(e[1], 1) for e in examples[:4]] == [round(100 * v, 1) for v in top4]
    assert all(flag_carbonyl[e[4]] for e in examples[4:]) and not any(flag_carbonyl[e[4]] for e in examples[:4])

    g_rows, g_cols = pick_block(grignard)
    e_rows, e_cols = pick_block(ester)

    fig = plt.figure(figsize=(WIDTH, 8.5))
    grid = GridSpec(3, 2, figure=fig, height_ratios=[4.07, 2.6, 0.75], width_ratios=[1, 1],
                    hspace=0.093, wspace=0.08, left=0.01, right=0.99, top=0.975, bottom=0.048)
    g_frame = fig.add_subplot(grid[0, 0])
    e_frame = fig.add_subplot(grid[0, 1])
    style.panel_title(g_frame, "A  Grignard yields follow the substrate pair")
    style.panel_title(e_frame, "B  Esterification yields follow the conditions")
    for frame in (g_frame, e_frame):
        frame.axis("off")
    fig.canvas.draw()
    g_shares = block_panel(fig, g_frame, grignard, g_flags, g_rows, g_cols, PURPLE, "grignard_addition",
                           "Grignard addition", table.loc["Grignard"])
    e_shares = block_panel(fig, e_frame, ester, e_flags, e_rows, e_cols, GREEN, "esterification",
                           "Esterification", table.loc["Esterification"])
    # each block keeps the direction of its full table
    assert g_shares[0] > g_shares[1] and e_shares[1] > e_shares[0], (g_shares, e_shares)
    figure4.block_shares = {"Grignard": g_shares, "Esterification": e_shares}

    # C: the eight selected carbonyl partners, two rows of four
    gallery = fig.add_subplot(grid[1, :])
    style.panel_title(gallery, "C  Amino or N–H carbonyl partners give little Grignard product")
    gallery.axis("off")
    fig.canvas.draw()
    gb = gallery.get_position()
    fig_w, fig_h = WIDTH, fig.get_figheight()
    col_w = gb.width * fig_w / 4
    left_in = gb.x0 * fig_w
    row_h = gb.height * fig_h / 2
    box_h = [max(chem.extent(e[2])[1] for e in examples[4 * r:4 * r + 4]) for r in range(2)]
    for k, (row, mean, asset, formula, _) in enumerate(examples):
        r, c = divmod(k, 4)
        cx = left_in + (c + 0.5) * col_w
        top_in = (gb.y0 + gb.height) * fig_h - r * row_h
        ex, ey = chem.extent(asset)
        if ex > col_w - 0.08 or box_h[r] + 0.42 > row_h:
            raise ValueError(f"Native structure needs more gallery space: {asset}")
        cy = top_in - 0.06 - box_h[r] / 2
        chem.place(fig, asset, centre_in=(cx, cy))
        fig.text((cx - col_w / 2 + 0.12) / fig_w, (top_in - 0.02) / fig_h, str(k + 1), ha="left", va="top",
                 weight="bold", color=INK)
        base = top_in - 0.06 - box_h[r]
        reagent_label(fig, cx, base - 0.14, formula)
        fig.text(cx / fig_w, (base - 0.28) / fig_h, f"pair mean {mean:.0f}%", ha="center", va="baseline",
                 color=INK, weight="bold")

    # D includes every measured pair from the 19 reagents that produced product anywhere in the array.
    comparison = productive_grignard_pairs(data)
    summary = fig.add_subplot(grid[2, :])
    style.panel_title(summary, "D  Amino or N–H partners average 21% versus 36%")
    summary.axis("off")
    box = summary.get_position()
    ax = fig.add_axes([0.215, box.y0 + 0.015, 0.68, box.height - 0.015])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 1.6)
    ax.set_yticks([1, 0], ["Amino or N–H", "Other partners"])
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Measured pair mean yield, %", labelpad=2)
    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0, pad=5)
    fig.canvas.draw()
    distributions = {}
    for included, row, colour in ((True, 1, PURPLE), (False, 0, MID)):
        values = comparison.loc[comparison.amino_or_protic == included, "mean_percent"].to_numpy()
        ys = stacked_pair_positions(ax, values, row)
        ax.scatter(values, ys, s=7.8, color=colour, linewidths=0, zorder=3, clip_on=False)
        mean = values.mean()
        mean_y = ys.min() - 0.22
        ax.plot([mean], [mean_y], marker="|", ms=5, mew=1.4, color=INK, zorder=4, clip_on=False)
        ax.text(mean + 1.6, mean_y, f"{mean:.0f}%", ha="left", va="center", color=INK, weight="bold")
        ax.text(102, row, f"n = {len(values)}", ha="left", va="center", color=GRAY)
        distributions["amino_or_protic" if included else "other"] = {
            "n": len(values), "mean_percent": float(mean), "values": values.tolist(),
            "y": ys.tolist(), "mean_marker_y": float(mean_y)}
    figure4.pair_distributions = distributions
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
    table["pair_pct"] = 100 * table.pair_variance_share
    high = table[table.high].copy()
    assert len(high) == 11
    summary = load_json(DIAG / "rf_ranking_structure_summary.json")
    high["combined"] = high.pair_variance_share.rank() + high.substrate_levels_per_pair.rank()
    r_comb = rho(high.combined, high.rf_delta_r2)
    expected = summary["two_design_properties"]["mean_yield_at_least_5pct"]["rho_rank_sum_vs_rf_delta_r2"]
    assert abs(r_comb - expected) < 1e-9, (r_comb, expected)
    r_share, r_share19 = rho(high.pair_variance_share, high.rf_delta_r2), rho(table.pair_variance_share, table.rf_delta_r2)
    r_levels = rho(high.substrate_levels_per_pair, high.rf_delta_r2)
    r_levels19 = rho(table.substrate_levels_per_pair, table.rf_delta_r2)

    fig = plt.figure(figsize=(WIDTH, 3.2))
    grid = GridSpec(1, 3, figure=fig, width_ratios=[1.45, 1, 1], wspace=0.36, left=0.075, right=0.985, top=0.87,
                    bottom=0.16)

    # A: both properties at once; circle area is the loss
    ax = fig.add_subplot(grid[0, 0])
    style.panel_title(ax, "A  Pair-led, sparsely paired arrays lose more")
    area = lambda loss: 700 * loss  # noqa: E731  marker area in points^2, proportional to the loss
    for row in high.sort_values("rf_delta_r2", ascending=False).itertuples():
        colour = HIGHLIGHT.get(row.screen, MID)
        ax.scatter(row.pair_pct, row.substrate_levels_per_pair, s=area(row.rf_delta_r2), color=colour,
                   alpha=0.85 if row.screen in HIGHLIGHT else 0.5, edgecolor="white", linewidths=0.6, zorder=3)
    for screen, (dx, dy, ha) in {"Grignard": (0, 14, "center"), "Esterification": (0, 9, "center"),
                                 "Heck": (7, 0, "left"), "Suzuki Bpin": (0, 12, "center")}.items():
        row = high.set_index("screen").loc[screen]
        ax.annotate(name(screen), (row.pair_pct, row.substrate_levels_per_pair), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", color=INK,
                    weight="bold" if screen in HIGHLIGHT else "normal")
    ax.text(0.03, 0.04, f"rank-sum ρ = {r_comb:.2f}", transform=ax.transAxes, ha="left", va="bottom", color=INK,
            weight="bold")
    for k, loss in enumerate((0.1, 0.3, 0.6)):  # size key, lower right, clear of the arrays
        ax.scatter(61 + 6.5 * k, 0.17, s=area(loss), color="white", edgecolor=GRAY, linewidths=0.6)
        ax.text(61 + 6.5 * k, 0.01, f"{loss:.1f}", ha="center", va="bottom", color=INK)
    ax.text(68.5, 0.3, "R² lost", ha="center", va="bottom", color=GRAY)
    ax.set_xlim(18, 78)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("Between-pair share of yield variance, %")
    ax.set_ylabel("Pairing sparsity")
    despine(ax)

    # B, C: each property alone against the loss, all 19 arrays
    for spec, column, title, rho11, rho19, xlabel, xlim in (
            (grid[0, 1], "pair_pct", "B  Pair-led yields lose more", r_share, r_share19,
             "Between-pair share, %", (20, 75)),
            (grid[0, 2], "substrate_levels_per_pair", "C  Sparse pairing loses more", r_levels, r_levels19,
             "Pairing sparsity", (0, 1.05))):
        ax = fig.add_subplot(spec)
        style.panel_title(ax, title)
        points(ax, table, column)
        label(ax, table, column, "Grignard", (0, -9) if column == "pair_pct" else (-6, 0),
              "center" if column == "pair_pct" else "right")
        corner = (0.97, 0.03, "right", "bottom") if column == "pair_pct" else (0.03, 0.75, "left", "top")
        ax.text(corner[0], corner[1], f"ρ = {rho11:.2f} (11)" + chr(10) + f"ρ = {rho19:.2f} (19)",
                transform=ax.transAxes, ha=corner[2], va=corner[3], color=INK)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 0.72)
        if column == "pair_pct":
            ax.set_ylabel("R² lost on new pairs")
        despine(ax)
    save(fig, "Figure_5")


def toc():
    """Compact graphical abstract: hide a few wells or the whole row of the same measured pair."""
    table = structure().set_index("screen")
    values, _, _ = overview_data()
    g = table.loc["Grignard"]
    size = 3.3
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1])
    extent = size * 72
    ax.set(xlim=(0, extent), ylim=(extent, 0))
    ax.axis("off")
    _, scheme_h = chem.extent("grignard_addition")
    chem.place(fig, "grignard_addition", centre_in=(size / 2, size - 14 / 72 - scheme_h / 2))
    ax.text(extent / 2, 78, "Same pair, 12 conditions", ha="center", va="baseline", color=INK)
    row_x, row_y, row_step = (extent - 120) / 2, 86, 10
    for j, value in enumerate(values[0]):
        ax.add_patch(Rectangle((row_x + j * row_step, row_y), 9.2, 10,
                               facecolor=overview_colour(value), lw=0))
    ax.add_patch(Rectangle((row_x - 1.3, row_y - 1.3), 121.8, 12.6,
                           facecolor="none", edgecolor=PURPLE, lw=0.8))
    for centre in (58, extent - 58):
        ax.annotate("", xy=(centre, 129), xytext=(extent / 2, 102),
                    arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 0.8,
                                "connectionstyle": "arc3,rad=0"})
    for left, centre, label, colour, score, whole in (
            (13, 58, "Held-out wells", TEAL, g.rf_row_r2, False),
            (extent - 103, extent - 58, "New pairs", ORANGE, g.rf_group_r2, True)):
        ax.text(centre, 141, label, ha="center", va="baseline", color=INK, weight="bold")
        if whole:
            ax.add_patch(Rectangle((left, 149), 90, 10, facecolor="white", edgecolor=colour, lw=1))
            ax.text(centre, 154, "?", ha="center", va="center", color=INK, weight="bold")
        else:
            for j, value in enumerate(values[0]):
                hidden = j in (2, 6, 9)
                ax.add_patch(Rectangle((left + j * 7.5, 149), 6.7, 10,
                                       facecolor="white" if hidden else overview_colour(value),
                                       edgecolor=colour if hidden else "none", lw=1 if hidden else 0))
        ax.text(centre, 178, f"{score:.2f}", ha="center", va="baseline", color=INK, weight="bold", fontsize=16,
                gid="toc-score")
    ax.text(extent / 2, 197, "Grignard array, R²", ha="center", va="baseline", color=GRAY)
    ax.text(extent / 2, 222, "Read the yield table before modelling", ha="center", va="baseline",
            color=INK, weight="bold")
    save(fig, "TOC")


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
