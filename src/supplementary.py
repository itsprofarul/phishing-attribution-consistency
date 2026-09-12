"""Electronic Supplementary Material builder.

Produces submission/supplementary/:

  ESM_1  the frozen UCI <-> PhiUSIIL feature concept mapping, as CSV
  ESM_2  PDF: progressive removal curve, the 16 confusion matrices, and the
         SHAP sample-size convergence check rendered as a table
  ESM_3  PDF: the exported result tables other than t1, t2, t5, t8, t9, t10, t15

Anonymity is enforced rather than assumed. Every string written into any of the
three files -- cell values, column names, labels, captions, and the PDF metadata
dictionaries -- passes through `scrub`, which RAISES on an author name, an
institution, a repository URL, an email address or an absolute filesystem path.
A silent redaction would be worse than a failure, because it could leave a
partially identifying fragment behind.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, banner, get_logger  # noqa: E402

log = get_logger("supplementary")

OUT_DIR = PROJECT_ROOT / "submission" / "supplementary"
TABLE_DIR = RESULTS_DIR / "paper_tables"

PAGE_W, PAGE_H = 8.27, 11.69          # A4 portrait, inches
DPI = 600

# --------------------------------------------------------------------------- #
# Anonymity
# --------------------------------------------------------------------------- #
# The author-identifying terms live in src/identity_terms.py, which is the one
# module excluded from the ESM_4 code archive: a file that must name the strings
# it protects cannot itself travel to reviewers. Absent that module there is no
# author identity in this copy to protect, so only the generic patterns apply.
try:
    from src.identity_terms import IDENTITY_TERMS
except ImportError:
    try:
        from identity_terms import IDENTITY_TERMS
    except ImportError:                    # anonymised copy: no terms configured
        IDENTITY_TERMS = []

FORBIDDEN = [(re.compile(pat, re.I), what) for pat, what in IDENTITY_TERMS] + [
    (re.compile(r"[A-Za-z]:\\"), "absolute Windows path"),
    # Written as an alternation rather than a literal path prefix, so the
    # anonymity tooling does not trip its own detector when it is archived.
    (re.compile(r"[A-Za-z]:/(?:Users|home)/"), "absolute Windows path"),
    (re.compile(r"/(?:Users|home)/"), "absolute POSIX path"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "email address"),
]


def scrub(text, where: str) -> str:
    """Return `text` unchanged, or raise if it carries identifying content."""
    s = str(text)
    for pat, what in FORBIDDEN:
        m = pat.search(s)
        if m:
            raise ValueError(f"{where}: refusing to write {what} -> {m.group(0)!r}")
    return s


def scrub_frame(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """Check column names and every non-numeric cell value.

    Tested on dtype rather than `== object`: pandas reads these CSVs into
    PyArrow-backed string columns, which are not object dtype, so an object
    test silently skips exactly the cells most likely to carry a name or path.
    """
    for col in df.columns:
        scrub(col, f"{where} column name")
        if not pd.api.types.is_numeric_dtype(df[col]):
            for v in df[col].dropna().unique():
                scrub(v, f"{where} [{col}]")
    return df


# --------------------------------------------------------------------------- #
# Display names
# --------------------------------------------------------------------------- #
CONFIG_NAMES = {
    "phiusiil_full": "PhiUSIIL",
    "phiusiil_leakfree": "PhiUSIIL (leak-controlled)",
    "uci_full": "UCI Phishing Websites",
    "uci_dedup": "UCI Phishing Websites (de-duplicated)",
}
MODEL_NAMES = {"rf": "Random Forest", "xgb": "XGBoost",
               "lgbm": "LightGBM", "ensemble": "Soft-voting ensemble"}
# Ordered so the four models of one configuration stay together.
CONFIG_ORDER = ["phiusiil_full", "phiusiil_leakfree", "uci_full", "uci_dedup"]
MODEL_ORDER = ["rf", "xgb", "lgbm", "ensemble"]

TABLE_CAPTIONS = {
    "t3": "Single-feature predictive power: held-out accuracy and mutual "
          "information for the fifteen highest-ranked features of each benchmark.",
    "t4": "Progressive removal on PhiUSIIL: held-out performance after greedily "
          "removing the highest-attributed feature at each iteration.",
    "t6": "Zero-shot transfer between benchmarks on the shared nine-concept "
          "schema, five seeds per direction.",
    "t7": "Within-dataset ceiling: Kendall's tau between attribution rankings "
          "from independent models on disjoint halves of the same data.",
    "t11": "Frozen feature concept mapping between UCI Phishing Websites and "
           "PhiUSIIL, with per-concept directional agreement.",
    "t12": "Explanation cost profile for progressive removal, annotated with "
           "whether each timing was taken under competing CPU load.",
    "t13": "Directional agreement per shared concept, ranked by how much the "
           "disagreement matters.",
    "t14": "Concepts retained and lost when PhiUSIIL is restricted to its "
           "leak-controlled feature set.",
    "t16": "Runtime observations for the reproducibility section, each labelled "
           "clean or contended.",
}
EXCLUDED_TABLES = {"t1", "t2", "t5", "t8", "t9", "t10", "t15"}

# Held record-wise by request: too many columns for any readable table.
RECORD_WISE_TABLES = {"t6", "t7", "t12", "t16"}
# At or below this many columns a table is rendered as a table, always.
MAX_COLS_TABULAR = 8

# Display names for dataset identifier cells in ESM_3, matching the figures.
# The pipeline spells the unrestricted variants phiusiil_full and uci_full in
# some tables and phiusiil and uci in others; both resolve to the same dataset
# and so to the same display name.
DATASET_DISPLAY = {
    "phiusiil": "PhiUSIIL",
    "phiusiil_full": "PhiUSIIL",
    "phiusiil_leakfree": "PhiUSIIL (leak-controlled)",
    "uci": "UCI Phishing Websites",
    "uci_full": "UCI Phishing Websites",
    "uci_dedup": "UCI Phishing Websites (de-duplicated)",
    "uci379_website_phishing": "UCI Website Phishing",
    "uci379_website_phishing__suspicious_dropped": "UCI Website Phishing (variant)",
    "mendeley_hannousse": "Mendeley (Hannousse)",
    "mendeley_tan": "Mendeley (Tan)",
}
SIGFIGS = 6            # significant figures for floats shown in ESM_3


def _pdf_metadata(pdf: PdfPages, title: str) -> None:
    """Anonymous PDF metadata; defaults would name the toolchain and the file."""
    d = pdf.infodict()
    d["Title"] = scrub(title, "pdf title")
    d["Author"] = ""
    d["Subject"] = "Electronic Supplementary Material"
    d["Keywords"] = ""
    d["Creator"] = "Electronic Supplementary Material builder"
    d["Producer"] = "matplotlib pdf backend"
    for k in ("CreationDate", "ModDate"):
        d.pop(k, None)


# --------------------------------------------------------------------------- #
# ESM_1
# --------------------------------------------------------------------------- #
ESM1_HEADER = [
    "# Frozen feature concept mapping.",
    "#",
    "# Scope: this mapping covers the UCI Phishing Websites <-> PhiUSIIL pairing",
    "# only. The other cross-dataset pairings (UCI Website Phishing, Mendeley",
    "# (Hannousse)) were matched on normalized feature names rather than by",
    "# concept, because their feature sets share names directly and required no",
    "# semantic assignment.",
    "#",
    "# status:  mapped | unmapped_uci | unmapped_phiusiil",
    "# rho_*:   Spearman correlation between feature value and SHAP value, per",
    "#          dataset; polarity_status flags inverted or absent signal.",
]


def build_esm1() -> dict:
    src = RESULTS_DIR / "transfer" / "feature_mapping.csv"
    if not src.exists():
        raise FileNotFoundError("run `python -m src.transfer --stage mapping` first")
    df = scrub_frame(pd.read_csv(src), "ESM_1")

    out = OUT_DIR / "ESM_1_feature_concept_mapping.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        for line in ESM1_HEADER:
            fh.write(scrub(line, "ESM_1 header") + "\n")
        # na_rep pinned explicitly: an empty cell must reach the reader as a
        # blank, never as the string 'nan'.
        df.to_csv(fh, index=False, na_rep="")

    counts = df["status"].value_counts().to_dict()
    log.info("ESM_1 -> %s", out.name)
    log.info("  %d rows, %d columns: %s", len(df), len(df.columns), counts)
    return {"path": out, "rows": len(df), "counts": counts,
            "columns": list(df.columns)}


# --------------------------------------------------------------------------- #
# Page helpers
# --------------------------------------------------------------------------- #
def _image_page(pdf: PdfPages, img_path: Path, label: str) -> None:
    """One raster figure per page at native size, with a one-line label above."""
    img = mpimg.imread(str(img_path))
    h_px, w_px = img.shape[0], img.shape[1]
    w_in, h_in = w_px / DPI, h_px / DPI

    max_w = PAGE_W - 1.2
    if w_in > max_w:                      # scale down; never up
        h_in *= max_w / w_in
        w_in = max_w

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.text(0.5, 0.94, scrub(label, "ESM_2 label"), ha="center", va="top",
             fontsize=10, fontweight="bold", wrap=True)
    left = (PAGE_W - w_in) / 2 / PAGE_W
    bottom = 0.90 - h_in / PAGE_H
    ax = fig.add_axes([left, bottom, w_in / PAGE_W, h_in / PAGE_H])
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def _text_pages(pdf: PdfPages, heading: str, body_lines: list[str],
                lines_per_page: int = 62, mono: float = 7.0) -> int:
    """Monospace text pages; returns how many pages were written."""
    pages = 0
    chunks = [body_lines[i:i + lines_per_page]
              for i in range(0, len(body_lines), lines_per_page)] or [[]]
    for n, chunk in enumerate(chunks):
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        head = heading if n == 0 else f"{heading}  (continued)"
        fig.text(0.06, 0.955, scrub(head, "page heading"), ha="left", va="top",
                 fontsize=10, fontweight="bold", wrap=True)
        fig.text(0.06, 0.915, "\n".join(scrub(ln, "page body") for ln in chunk),
                 ha="left", va="top", fontsize=mono, family="monospace")
        pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        pages += 1
    return pages


# --------------------------------------------------------------------------- #
# Frame rendering
#
# Two layouts. A real table is preferable wherever the columns can be given
# enough width to stay legible, so it is tried first and the font is stepped
# down until the rows stop wrapping. Record-wise key/value blocks are the
# fallback for frames whose cells hold prose rather than values, where a table
# would wrap every row into an unreadable stack.
# --------------------------------------------------------------------------- #
MONO_EM = 0.602        # DejaVu Sans Mono advance width, in em
TEXT_FRAC = 0.88       # fraction of the page width the text block occupies
BLOCK_IN = 62 * 1.2 * 7.0 / 72.0      # height of the 62-line 7pt block, inches
FONT_LADDER = (7.0, 6.5, 6.0, 5.5, 5.0)
WRAP_BUDGET = 1.35     # accept a font once rows average at most this many lines
RECORDS_ABOVE = 2.0    # past this even the smallest font is not worth a table


def _cell(v) -> str:
    """Text for one cell. A missing value renders blank, never as 'nan'."""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):       # arrays and other non-scalars
        pass
    s = str(v)
    return "" if s.strip().lower() in {"nan", "nat", "none", "<na>"} else s


def _numeric_like(series) -> bool:
    """True for a column of numbers, including one already rendered to text.

    Width fitting needs this after ESM_3 formats its floats to strings: a
    number is still a single token that must not be wrapped.
    """
    if pd.api.types.is_numeric_dtype(series):
        return True
    values = [v for v in (_cell(x) for x in series) if v]
    if not values:
        return False
    try:
        for v in values:
            float(v)
    except ValueError:
        return False
    return True


def _display_value(s: str) -> str:
    """Rewrite a cell that IS a dataset identifier; leave prose untouched.

    Only a whole-cell identifier and the 'a->b' transfer-direction form are
    rewritten. Substituting inside free text would corrupt values like
    'unmapped_uci', which is a mapping status rather than a dataset.
    """
    key = s.strip()
    if key in DATASET_DISPLAY:
        return DATASET_DISPLAY[key]
    if "->" in key:
        parts = [p.strip() for p in key.split("->")]
        if all(p in DATASET_DISPLAY for p in parts):
            return " -> ".join(DATASET_DISPLAY[p] for p in parts)
    return s


def _prepare_esm3(df: pd.DataFrame) -> pd.DataFrame:
    """Presentation pass for ESM_3: significant figures and display names.

    Applies to the rendered document only. The CSVs under results/ and the
    ESM_1 export keep full float64 precision and the pipeline's own keys.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = [("" if _cell(v) == "" else f"{float(v):.{SIGFIGS}g}")
                      for v in out[c]]
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = [_display_value(_cell(v)) for v in out[c]]
    return out


def _char_budget(pt: float) -> int:
    """How many monospace characters fit across the text block at this size."""
    return int(PAGE_W * TEXT_FRAC / (MONO_EM * pt / 72.0))


def _lines_per_page(pt: float) -> int:
    """Keep the text block the same height whatever the font size."""
    return int(BLOCK_IN / (1.2 * pt / 72.0))


def _fit_widths(df: pd.DataFrame, cols: list[str], budget: int,
                floor: int = 6) -> dict:
    """Shrink the widest column repeatedly until a row fits the budget.

    A number is one token: wrapping 0.9967374960991802 across two lines makes
    it unreadable and invites misreading. Numeric columns are therefore held at
    their full width while any text column can still give ground, and are only
    shrunk when nothing else is left to take.
    """
    w = {c: max([len(str(c))] + [len(_cell(v)) for v in df[c]]) for c in cols}
    text_cols = [c for c in cols if not _numeric_like(df[c])]

    def span() -> int:
        return sum(w.values()) + 2 * (len(cols) - 1)

    while span() > budget:
        pool = [c for c in text_cols if w[c] > floor] or \
               [c for c in cols if w[c] > floor]
        if not pool:                      # everything is already at the floor
            break
        w[max(pool, key=lambda c: w[c])] -= 1
    return w


def _table_body(df: pd.DataFrame, cols: list[str],
                w: dict) -> tuple[list[str], int]:
    """Header, rule and rows, wrapped inside each column so nothing is cut."""
    def line(parts: list[list[str]], i: int) -> str:
        return "  ".join((parts[j][i] if i < len(parts[j]) else "").ljust(w[c])
                         for j, c in enumerate(cols)).rstrip()

    head = [textwrap.wrap(str(c), w[c]) or [""] for c in cols]
    out = [line(head, i) for i in range(max(len(h) for h in head))]
    out.append("-" * (sum(w.values()) + 2 * (len(cols) - 1)))

    blocks, tall = [], False
    for _, r in df.iterrows():
        cells = [textwrap.wrap(_cell(r[c]), w[c]) or [""] for c in cols]
        height = max(len(c) for c in cells)
        tall = tall or height > 1
        blocks.append([line(cells, i) for i in range(height)])

    for block in blocks:
        out.extend(block)
        if tall:                          # blank line keeps wrapped rows apart
            out.append("")
    return out, sum(len(b) for b in blocks)


def _records(df: pd.DataFrame, width: int = 118) -> list[str]:
    """One key/value block per row; nothing is truncated."""
    cols = list(df.columns)
    out = []
    label_w = max(len(str(c)) for c in cols)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        out.append(f"[record {i} of {len(df)}]")
        for c in cols:
            wrapped = textwrap.wrap(_cell(r[c]), width=width - label_w - 3) or [""]
            out.append(f"{str(c).rjust(label_w)} : {wrapped[0]}")
            for cont in wrapped[1:]:
                out.append(f"{' ' * label_w}   {cont}")
        out.append("")
    return out


def _render_frame(df: pd.DataFrame, force_records: bool = False,
                  force_table: bool = False) -> tuple[list[str], float, int, str]:
    """Return (lines, mono_pt, lines_per_page, layout)."""
    if not force_records:
        cols = list(df.columns)
        chosen = None
        for pt in FONT_LADDER:
            lines, used = _table_body(df, cols,
                                      _fit_widths(df, cols, _char_budget(pt)))
            chosen = (lines, pt, used / max(len(df), 1))
            if chosen[2] <= WRAP_BUDGET:
                break
        lines, pt, avg = chosen
        if force_table or avg <= RECORDS_ABOVE:
            return lines, pt, _lines_per_page(pt), "table"

    return _records(df), 7.0, 62, "records"


# --------------------------------------------------------------------------- #
# ESM_2
# --------------------------------------------------------------------------- #
def build_esm2() -> dict:
    out = OUT_DIR / "ESM_2_supplementary_figures.pdf"
    contents, missing, pages = [], [], 0

    with PdfPages(out) as pdf:
        _pdf_metadata(pdf, "Supplementary figures")

        f4 = FIGURES_DIR / "paper" / "f4_progressive_removal.png"
        if f4.exists():
            label = ("Progressive removal on PhiUSIIL: held-out accuracy against "
                     "the number of features removed.")
            _image_page(pdf, f4, label)
            contents.append(("figure", "progressive removal curve"))
            pages += 1
        else:
            missing.append(f4.name)

        for cfg in CONFIG_ORDER:
            for mdl in MODEL_ORDER:
                p = FIGURES_DIR / f"confusion_{cfg}_{mdl}.png"
                if not p.exists():
                    missing.append(p.name)
                    continue
                label = (f"Confusion matrix, summed over five seeds: "
                         f"{CONFIG_NAMES[cfg]} - {MODEL_NAMES[mdl]}.")
                _image_page(pdf, p, label)
                contents.append(("figure", f"{CONFIG_NAMES[cfg]} - {MODEL_NAMES[mdl]}"))
                pages += 1

        conv = RESULTS_DIR / "shap_convergence.csv"
        if conv.exists():
            df = scrub_frame(pd.read_csv(conv), "ESM_2 convergence")
            show = df[["dataset", "size_a", "size_b", "effective_a", "effective_b",
                       "kendall_tau", "n_available", "degenerate"]].copy()
            show["dataset"] = show["dataset"].map(
                {"phiusiil": "PhiUSIIL", "uci": "UCI Phishing Websites"}
            ).fillna(show["dataset"])
            show["kendall_tau"] = show["kendall_tau"].map(lambda v: f"{v:.4f}")
            show["degenerate"] = show["degenerate"].map(
                lambda b: "YES - see note" if bool(b) else "no")
            n_deg = int(df["degenerate"].sum())
            note = [
                "",
                "Note on the rows marked degenerate:",
                "",
                textwrap.fill(
                    "UCI Phishing Websites has a held-out pool of 878 instances. The "
                    "878/5000, 5000/10000 and 10000/20000 comparisons therefore draw "
                    "the same sample at both nominal sizes, so their tau = 1.0000 "
                    "holds by construction and is not evidence of convergence. The "
                    "informative UCI rows are the adaptive ladder (219/439 and "
                    "439/878). PhiUSIIL's pool is large enough for the requested "
                    "ladder to run in full.", width=112),
            ]
            label = ("SHAP sample-size convergence check: Kendall's tau between "
                     "attribution rankings computed at successive sample sizes.")
            lines, mono, lpp, _ = _render_frame(show)
            pages += _text_pages(pdf, label, lines + note, lpp, mono)
            contents.append(("table",
                             f"SHAP convergence ({len(show)} rows, {n_deg} degenerate)"))
        else:
            missing.append(conv.name)

    log.info("ESM_2 -> %s (%d pages)", out.name, pages)
    for kind, name in contents:
        log.info("  %-7s %s", kind, name)
    if missing:
        log.warning("  MISSING: %s", missing)
    return {"path": out, "contents": contents, "missing": missing, "pages": pages}


# --------------------------------------------------------------------------- #
# ESM_3
# --------------------------------------------------------------------------- #
def build_esm3() -> dict:
    out = OUT_DIR / "ESM_3_supplementary_tables.pdf"
    included, skipped = [], []

    paths = sorted(TABLE_DIR.glob("t*.csv"),
                   key=lambda p: int(re.match(r"t(\d+)", p.stem).group(1)))
    with PdfPages(out) as pdf:
        _pdf_metadata(pdf, "Supplementary tables")
        for p in paths:
            tid = re.match(r"(t\d+)", p.stem).group(1)
            if tid in EXCLUDED_TABLES:
                skipped.append(tid)
                continue
            df = _prepare_esm3(scrub_frame(pd.read_csv(p), f"ESM_3 {tid}"))
            heading = f"Table {tid.upper()}. {TABLE_CAPTIONS.get(tid, '')}"
            wide = tid in RECORD_WISE_TABLES
            lines, mono, lpp, layout = _render_frame(
                df, force_records=wide,
                force_table=not wide and len(df.columns) <= MAX_COLS_TABULAR)
            n = _text_pages(pdf, heading, lines, lpp, mono)
            included.append((tid, len(df), len(df.columns), n, layout, mono))

    total = sum(row[3] for row in included)
    log.info("ESM_3 -> %s (%d pages)", out.name, total)
    for tid, rows, cols, n, layout, mono in included:
        log.info("  %-4s %3d rows x %2d cols  %-8s %.1fpt  (%d page%s)",
                 tid.upper(), rows, cols, layout, mono, n, "" if n == 1 else "s")
    log.info("  excluded by request: %s", sorted(skipped))
    return {"path": out, "included": included, "skipped": sorted(skipped),
            "pages": total}


def main() -> None:
    banner(log, "ELECTRONIC SUPPLEMENTARY MATERIAL")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [build_esm1(), build_esm2(), build_esm3()]
    banner(log, "WRITTEN", "-")
    for r in results:
        p = r["path"]
        log.info("  %-42s %9.1f KB  (exists=%s)",
                 p.name, p.stat().st_size / 1024, p.exists())


if __name__ == "__main__":
    main()
