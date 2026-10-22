"""Shared figure style for the approved Chem figures.

Arial throughout; 9-pt bold uppercase panel letters (Cell Press style) and titles; 8-pt axis titles, tick labels,
legends and annotations; neutral text (colour is carried by the marks only); no background gridlines.
Panel letters and titles are placed after layout from the rendered extent of each panel, so that letters
in one grid row share a baseline, letters in one grid column share a left edge, and each letter sits at or
left of the leftmost drawn element of every panel in its column.
"""

from __future__ import annotations

import matplotlib
from matplotlib.colors import to_hex
from matplotlib.text import Text
import re

INK = "#242b30"
GRAY = "#7a8389"
LETTER_SIZE, TITLE_SIZE, LABEL_SIZE, TEXT_SIZE = 9, 9, 8, 8
TITLE_GAP = 6  # points between the top of a panel row and the baseline of its letters and titles
TITLE_INDENT = 10  # points from a panel letter to the start of its title
LETTER_MARGIN = 1.5  # points the letter column stands left of the leftmost drawn element
PNG_DPI = 300
RC = {
    "font.family": "Arial", "font.size": TEXT_SIZE, "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TEXT_SIZE, "ytick.labelsize": TEXT_SIZE, "legend.fontsize": TEXT_SIZE,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def use_style():
    from pathlib import Path

    from matplotlib.font_manager import FontProperties, findfont, fontManager

    # Where a system-wide Arial is installed, ignore per-user copies of the same family, which can be older
    # releases carrying a different PostScript name; the PDFs then embed the system ArialMT faces.
    system = Path("C:/Windows/Fonts")
    if (system / "arial.ttf").exists():
        fontManager.ttflist = [f for f in fontManager.ttflist
                               if f.name != "Arial" or Path(f.fname).parent.resolve() == system.resolve()]
        fontManager._findfont_cached.cache_clear()
    findfont(FontProperties(family="Arial"), fallback_to_default=False)
    matplotlib.rcParams.update(RC)


def panel_title(ax, text):
    """Record a panel letter and title; 'a  Title' carries a letter, any other text is a bare title.

    A placeholder title of the same size is drawn so that the layout engine reserves its space.
    """
    letter, title = (text[0], text[3:]) if len(text) > 3 and text[0].isalpha() and text[1:3] == "  " else (None, text)
    ax._panel_label = (letter, title)
    ax.set_title(text, loc="left", fontsize=LETTER_SIZE, weight="bold", color=INK)


def _white_yield_number(text, figure_name):
    if figure_name not in {"Figure_4", "Figure_4.pdf"} or not re.fullmatch(r"(?:[6-9]\d|100)", text.get_text()):
        return False
    ax = text.axes
    if (ax is None or len(ax.images) != 1 or text.get_transform() != ax.transData
            or text.get_ha() != "center" or text.get_va() != "center"):
        return False
    image = ax.images[0]
    values = image.get_array()
    x, y = text.get_position()
    column, row = round(x), round(y)
    if (values.shape != (16, 12) or abs(x - column) > 1e-8 or abs(y - row) > 1e-8
            or not 0 <= column < 12 or not 0 <= row < 16
            or image.norm.vmin != 0 or image.norm.vmax != 100):
        return False
    value = values[row, column]
    if getattr(value, "mask", False) or not 60 <= float(value) <= 100 or text.get_text() != f"{float(value):.0f}":
        return False
    red, green, blue, alpha = image.cmap(image.norm(value))
    return (alpha == 1 and image.get_alpha() in (None, 1) and text.get_alpha() in (None, 1)
            and 0.2126 * red + 0.7152 * green + 0.0722 * blue <= 0.6)


def _neutral(texts, *, figure_name=None):
    allowed = {to_hex(INK), to_hex(GRAY)}
    for text in texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        if to_hex(text.get_color()) == "#000000":  # matplotlib default, e.g. legend entries
            text.set_color(INK)
        if to_hex(text.get_color()) == "#ffffff" and _white_yield_number(text, figure_name):
            continue
        if to_hex(text.get_color()) not in allowed:
            raise ValueError(f"Coloured text in figure: {text.get_text()!r} ({to_hex(text.get_color())})")


def finalize(fig, *, figure_name=None):
    """Apply the type scale, freeze the layout and place panel letters and titles."""
    notes = list(fig.texts)
    for legend in fig.legends:
        notes += legend.get_texts()
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(LABEL_SIZE)
        ax.yaxis.label.set_fontsize(LABEL_SIZE)
        ax.tick_params(labelsize=TEXT_SIZE)
        ax.grid(False)
        if ax.get_legend() is not None:
            notes += ax.get_legend().get_texts()
        notes += ax.texts
    for text in notes:
        text.set_fontsize(TEXT_SIZE)
    _neutral(fig.findobj(match=Text), figure_name=figure_name)

    fig.canvas.draw()  # layout with placeholder titles reserving their space
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    panels = [ax for ax in fig.axes if hasattr(ax, "_panel_label")]
    for ax in panels:
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_text("")
    boxes = {ax: ax.get_tightbbox(renderer) for ax in panels}
    rows, columns = {}, {}
    for ax in panels:
        spec = ax.get_subplotspec()
        rows.setdefault(spec.rowspan.start, []).append(ax)
        columns.setdefault(spec.colspan.start, []).append(ax)
    px = fig.dpi / 72
    left = {ax: max(min(boxes[other].x0 for other in column) - LETTER_MARGIN * px, 0)
            for column in columns.values() for ax in column}
    base = {ax: max(boxes[other].y1 for other in row) + TITLE_GAP * px for row in rows.values() for ax in row}
    to_figure = fig.transFigure.inverted()
    placed = []
    for ax in panels:
        letter, title = ax._panel_label
        x, y = to_figure.transform((left[ax], base[ax]))
        if letter:
            placed.append(fig.text(x, y, letter, fontsize=LETTER_SIZE, weight="bold", color=INK,
                                   ha="left", va="baseline"))
            x = to_figure.transform((left[ax] + TITLE_INDENT * px, base[ax]))[0]
        placed.append(fig.text(x, y, title, fontsize=TITLE_SIZE, weight="bold", color=INK,
                               ha="left", va="baseline"))
    fig.canvas.draw()
    check_labels(fig, placed, boxes)
    check_text_bounds(fig)


def check_text_bounds(fig):
    """Reject text that a PDF renderer could clip silently at the page edge."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    texts = list(fig.texts)
    legends = list(fig.legends)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        texts += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                if not axis.get_visible():
                    continue
                texts += [axis.label, axis.get_offset_text()]
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if low - 1e-10 <= tick.get_loc() <= high + 1e-10:
                        texts += [tick.label1, tick.label2]
    for legend in legends:
        texts += legend.get_texts()
    for text in texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width + 0.5 or box.y1 > height + 0.5:
            raise ValueError(f"Text outside figure: {text.get_text()!r}; bbox={box.bounds}, canvas={(width, height)}")


def check_labels(fig, placed, boxes):
    """Every letter and title lies inside the figure and clear of all panels and of each other."""
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    extents = [text.get_window_extent(renderer) for text in placed]
    for text, box in zip(placed, extents):
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width + 0.5 or box.y1 > height + 0.5:
            raise ValueError(f"Panel label outside the figure: {text.get_text()!r}")
        for ax, panel in boxes.items():
            if box.overlaps(panel):
                raise ValueError(f"Panel label {text.get_text()!r} overlaps a panel")
    for i, a in enumerate(extents):
        for b in extents[i + 1:]:
            if a.overlaps(b):
                raise ValueError("Panel labels overlap each other")
