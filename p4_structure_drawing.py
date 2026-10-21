"""Place verified ChemDraw SVG artwork at its native physical size.

The assets retain editable text and exported vector bonds. SVG CSS pixels are
converted to points once; artwork is never fitted to a target box.
"""

from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import math
import re
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "chemical_assets"
ASSET_NAMES = frozenset((
    "quinoline_8_carbaldehyde", "chloronicotinaldehyde", "dipyridyl_ketone",
    "chloropicolinaldehyde", "indolyl_acetone", "phenyl_methyl_pyrazolone",
    "piperidinyl_acetophenone",
    "methylamino_acetophenone", "grignard_addition", "esterification",
))
SVG_NS = "http://www.w3.org/2000/svg"
PT_PER_CSS_PIXEL = 72 / 96
_placements = weakref.WeakKeyDictionary()
_group_ids = itertools.count(1)
ET.register_namespace("", SVG_NS)


def _tag(node):
    return node.tag.rsplit("}", 1)[-1]


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite artwork dimension: {value!r}")
    return result


def _length_pt(value):
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))(pt|px|mm|cm|in)?\s*", value or "")
    if not match:
        raise ValueError(f"SVG physical length is not explicit: {value!r}")
    factor = {None: PT_PER_CSS_PIXEL, "px": PT_PER_CSS_PIXEL, "pt": 1,
              "mm": 72 / 25.4, "cm": 72 / 2.54, "in": 72}[match.group(2)]
    return _number(match.group(1)) * factor


def _asset(name):
    if name not in ASSET_NAMES:
        raise ValueError(f"Unknown native chemical asset: {name!r}")
    metadata = json.loads((ASSET_DIR / "assets.json").read_text(encoding="utf-8-sig"))
    entries = metadata.get("assets", [])
    if metadata.get("schema_version") != 1 or len(entries) != len(ASSET_NAMES):
        raise ValueError("Native chemical asset manifest has an invalid schema or inventory")
    indexed = {entry["name"]: entry for entry in entries}
    if set(indexed) != ASSET_NAMES:
        raise ValueError("Native chemical asset manifest has missing or duplicate names")
    style = metadata.get("style", {})
    if (style.get("bond_length_pt") != 14.4 or style.get("label_size_pt") != 8
            or style.get("label_font") != "Arial" or _number(style.get("line_width_pt", 0)) < 0.5):
        raise ValueError("Native chemical asset manifest does not declare the approved physical style")
    entry = indexed[name]
    files = {}
    for extension in ("cdxml", "svg"):
        filename = f"{name}.{extension}"
        if entry.get(extension) != filename:
            raise ValueError(f"Unexpected {extension} path for {name}")
        data = (ASSET_DIR / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.get(f"{extension}_sha256"):
            raise ValueError(f"Native asset checksum mismatch: {filename}")
        files[extension] = data
    root = ET.fromstring(files["svg"])
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError(f"Native asset is not an SVG document: {name}")
    width, height = _length_pt(root.get("width")), _length_pt(root.get("height"))
    viewbox = tuple(_number(item) for item in re.split(r"[\s,]+", root.get("viewBox", "").strip()))
    if len(viewbox) != 4 or min(width, height, viewbox[2], viewbox[3]) <= 0:
        raise ValueError(f"Native SVG dimensions are not known: {name}")
    declared_viewbox = tuple(_number(item) for item in entry["svg_viewbox"])
    if (len(declared_viewbox) != 4 or any(abs(a - b) > 1e-6 for a, b in zip(viewbox, declared_viewbox))
            or abs(width - _number(entry["width_pt"])) > 1e-6
            or abs(height - _number(entry["height_pt"])) > 1e-6):
        raise ValueError(f"Native SVG dimensions differ from the manifest: {name}")
    scales = (width / viewbox[2], height / viewbox[3])
    if (abs(_number(entry["svg_units_per_point"]) - 1 / PT_PER_CSS_PIXEL) > 1e-6
            or any(abs(scale - PT_PER_CSS_PIXEL) > 1e-6 for scale in scales)):
        raise ValueError(f"Native SVG must retain its fixed CSS-pixel physical scale: {name}")
    vectors, texts = 0, 0
    for node in root.iter():
        tag = _tag(node)
        if tag in {"image", "foreignObject", "script"}:
            raise ValueError(f"Non-vector or executable object in native SVG: {name}")
        vectors += tag in {"path", "line", "polyline", "polygon"}
        texts += tag == "text"
        for key, value in node.attrib.items():
            if key.rsplit("}", 1)[-1] in {"href", "src"} and not value.startswith("#"):
                raise ValueError(f"External resource in native SVG: {name}")
        family = node.get("font-family")
        if family is not None and family.strip("\"'").casefold() != "arial":
            raise ValueError(f"Non-Arial text in native SVG: {name}")
    if not vectors or not texts:
        raise ValueError(f"Native asset must retain vector bonds and editable text: {name}")
    return entry, files["svg"], root, width, height, viewbox


def extent(name):
    """Return the verified native width and height in inches."""
    _, _, _, width, height, _ = _asset(name)
    return width / 72, height / 72


def svg_group(name, x_pt, y_pt):
    """Return an SVG group with its native top-left at point coordinates x/y.

    The enclosing SVG must use points as its user units. The coordinate system
    is SVG's usual top-left origin with y increasing downwards.
    """
    _, _, root, _, _, viewbox = _asset(name)
    x, y = _number(x_pt), _number(y_pt)
    prefix = f"native_{name}_{next(_group_ids)}_"
    outer = ET.Element(f"{{{SVG_NS}}}g", id=prefix.rstrip("_"),
                       transform=f"translate({x:.12g} {y:.12g})")
    attributes = {key: value for key, value in root.attrib.items()
                  if key not in {"width", "height", "viewBox", "version", "id"}}
    attributes["transform"] = (f"scale({PT_PER_CSS_PIXEL:.12g}) "
                               f"translate({-viewbox[0]:.12g} {-viewbox[1]:.12g})")
    artwork = ET.SubElement(outer, f"{{{SVG_NS}}}g", attributes)
    for child in root:
        if _tag(child) != "metadata":
            artwork.append(copy.deepcopy(child))
    replacements = {node.get("id"): prefix + node.get("id")
                    for node in artwork.iter() if node.get("id")}
    for node in artwork.iter():
        if node.get("id"):
            node.set("id", replacements[node.get("id")])
        for key, value in list(node.attrib.items()):
            for old, new in replacements.items():
                value = value.replace(f"url(#{old})", f"url(#{new})")
                if key.rsplit("}", 1)[-1] == "href" and value == f"#{old}":
                    value = f"#{new}"
            node.set(key, value)
    return ET.tostring(outer, encoding="unicode")


def place(fig, name, centre_in):
    """Register artwork centred in inches from the figure's bottom-left."""
    entry, _, _, width, height, _ = _asset(name)
    if len(centre_in) != 2:
        raise ValueError("Native artwork centre must contain x/y inches")
    cx, cy = (_number(value) for value in centre_in)
    fw, fh = fig.get_size_inches()
    if cx * 72 < width / 2 or (fw - cx) * 72 < width / 2 or cy * 72 < height / 2 or (fh - cy) * 72 < height / 2:
        raise ValueError(f"Native chemical artwork would extend beyond the figure: {name}")
    _placements.setdefault(fig, []).append((name, cx, cy, entry["svg_sha256"]))


def _register_arial():
    from reportlab.pdfbase import pdfmetrics
    from svglib import fonts as svg_fonts

    for weight, slant, filename, registered, expected in (
            ("normal", "normal", "arial.ttf", "Arial", "ArialMT"),
            ("bold", "normal", "arialbd.ttf", "Arial-Bold", "Arial-BoldMT"),
            ("normal", "italic", "ariali.ttf", "Arial-Italic", "Arial-ItalicMT")):
        candidates = (Path("C:/Windows/Fonts") / filename,
                      Path("/usr/share/fonts/truetype/msttcorefonts") / filename,
                      Path("/Library/Fonts") / filename)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"System Arial font file is required: {filename}")
        name, _ = svg_fonts.register_font("Arial", font_path=str(path), weight=weight,
                                          style=slant, rlgFontName=registered)
        face = pdfmetrics.getFont(registered).face.name
        if isinstance(face, bytes):
            face = face.decode("ascii")
        if name != registered or face != expected:
            raise ValueError(f"System font has an unexpected PostScript name: {filename}: {face}")


def _svg_pdf(svg, width, height):
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg

    drawing = svg2rlg(io.BytesIO(svg))
    if drawing is None or abs(drawing.width - width) > 1e-6 or abs(drawing.height - height) > 1e-6:
        raise ValueError("SVG converter changed the native artwork physical dimensions")
    drawing.initialFontName = "Arial"
    drawing.initialFontSize = 8
    return renderPDF.drawToString(drawing, invariant=1)


def compose_pdf(fig, pdf_path):
    """Overlay registered native vectors on an existing one-page figure PDF.

    Return False without touching the PDF when nothing is registered. PNG
    previews for composed figures must be rendered from this final PDF.
    """
    placements = _placements.get(fig, [])
    if not placements:
        return False
    import pymupdf as fitz

    _register_arial()
    pdf_path = Path(pdf_path)
    fw, fh = (float(value) * 72 for value in fig.get_size_inches())
    allowed_fonts = {"ArialMT", "Arial-BoldMT", "Arial-ItalicMT"}
    with fitz.open(pdf_path) as document:
        if len(document) != 1 or abs(document[0].rect.width - fw) > 1e-4 or abs(document[0].rect.height - fh) > 1e-4:
            raise ValueError("Native artwork requires the unchanged full-size one-page figure canvas")
        page = document[0]
        for name, cx, cy, expected_sha in placements:
            entry, svg, _, width, height, _ = _asset(name)
            if entry["svg_sha256"] != expected_sha:
                raise ValueError(f"Native artwork changed after placement: {name}")
            left, top = cx * 72 - width / 2, fh - cy * 72 - height / 2
            target = fitz.Rect(left, top, left + width, top + height)
            if target.x0 < 0 or target.y0 < 0 or target.x1 > fw or target.y1 > fh:
                raise ValueError(f"Native artwork is outside the final figure canvas: {name}")
            with fitz.open(stream=_svg_pdf(svg, width, height), filetype="pdf") as artwork:
                fonts = {re.sub(r"^[A-Z]{6}\+", "", item[3]) for item in artwork[0].get_fonts(full=True)}
                spans = [span for block in artwork[0].get_text("dict")["blocks"]
                         for line in block.get("lines", []) for span in line["spans"] if span["text"].strip()]
                if artwork[0].get_images() or not fonts or not fonts.issubset(allowed_fonts):
                    raise ValueError(f"Native SVG conversion introduced raster content or nonstandard Arial: {name}: {fonts}")
                if not spans or any(abs(span["size"] - 8) > 0.01 for span in spans):
                    raise ValueError(f"Native SVG text does not retain its approved 8 pt physical size: {name}")
                page.show_pdf_page(target, artwork, 0, keep_proportion=True, overlay=True)
        document.set_metadata({**document.metadata, "creationDate": "", "modDate": ""})
        result = document.tobytes(garbage=4, deflate=True, no_new_id=True)
    pdf_path.write_bytes(result)
    return True
