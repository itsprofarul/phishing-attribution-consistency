"""Task B6 - consolidated, publication-ready export.

Collects the scattered result files into results/paper_tables/ and figures/paper/.

Two rules are enforced here rather than left to whoever writes the manuscript:

  1. A table that depends on an experiment which did not run is NOT emitted as an
     empty file. It is skipped and recorded as missing, with the reason, so a gap
     in the evidence cannot be mistaken for a null result.
  2. Figure captions describing the explanation-cost curve are generated from the
     data (see explanation_cost.describe_cost_curve), never hand-written. The
     curve is non-monotonic, and a caption claiming steady growth would be
     contradicted by the very figure it labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FIGURES_DIR, LEAK_DIR, RESULTS_DIR, banner, get_logger, write_csv_atomic  # noqa: E402
from src.benchmark_screen import REGIME_RETUNED, REGIME_STATIC  # noqa: E402

log = get_logger("paper_export")

TABLE_DIR = RESULTS_DIR / "paper_tables"
PAPER_FIG_DIR = FIGURES_DIR / "paper"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Greyscale-safe, colour-blind-safe; distinguishable when printed in mono.
PALETTE = ["#1b1b1b", "#6e6e6e", "#b0b0b0", "#3b6ea5", "#a5443b"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
# Springer two-column: ~3.35in single column, ~6.9in full width.
COL_W, FULL_W = 3.35, 6.9

# Internal identifiers -> the names used in the manuscript. Figures are read by
# people who have never seen the config keys, so no codename may reach a label.
#
# NOTE: `uci_dedup` has no counterpart in the manuscript's dataset list because it
# is a sensitivity VARIANT of UCI Phishing Websites rather than a separate
# benchmark. It is named consistently with that scheme here.
DISPLAY = {
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
PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def disp(key) -> str:
    """Manuscript name for an internal key, unchanged if already display-form."""
    return DISPLAY.get(str(key), str(key))


def panel_letter(ax, i: int) -> None:
    """Panel letter above the axes, where no title now competes for the space."""
    ax.text(0.0, 1.02, PANEL_LETTERS[i], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) else None
    except Exception as exc:            # pragma: no cover - diagnostics only
        log.warning("could not read %s: %s", path, exc)
        return None


def _emit(name: str, df: pd.DataFrame | None, missing: dict, reason: str) -> bool:
    if df is None or df.empty:
        missing[name] = reason
        log.warning("SKIPPED %-34s %s", name, reason)
        return False
    write_csv_atomic(df, TABLE_DIR / name)
    log.info("wrote   %-34s %d rows", name, len(df))
    return True


def _emit_regime_reconciliation(prog, e1, missing: dict) -> None:
    """Pair every accuracy reported for the SAME feature set under both regimes.

    Exists so the discrepancy is a documented result rather than an apparent
    contradiction between two tables.
    """
    if prog is None or e1 is None:
        missing["t15_hyperparameter_regime_reconciliation.csv"] = (
            "needs both the A1.3 curve and the E1 v2 summary")
        return
    rows = []
    ens = e1[e1["model"] == "ensemble"] if "model" in e1.columns else e1
    lf = ens[ens["config"] == "phiusiil_leakfree"] if "config" in ens.columns else None
    if lf is not None and len(lf) and len(prog):
        static = float(prog["accuracy"].iloc[-1])
        n_feat = int(prog["n_features_remaining"].iloc[-1])
        retuned = float(lf["accuracy_mean"].iloc[0])
        rows.append({
            "feature_set": f"phiusiil leak-free ({n_feat} features)",
            "accuracy_static_regime": round(static, 6),
            "accuracy_retuned_regime": round(retuned, 6),
            "difference": round(retuned - static, 6),
            "static_source": "t4 / A1.3 progressive removal (final iteration)",
            "retuned_source": "t5 / E1 v2 (phiusiil_leakfree, mean over 5 seeds)",
            "crosses_0.99_threshold": bool((static < 0.99) != (retuned < 0.99)),
            "note": ("Same 36 features, same data, different hyperparameters. The "
                     "removal curve reuses parameters tuned on the full 50-feature "
                     "set; E1 re-tunes. The gap moves the figure across the 0.99 "
                     "verdict line, so the de-leak verdict is hyperparameter-"
                     "sensitive and must not be quoted without its regime."),
        })
    _emit("t15_hyperparameter_regime_reconciliation.csv",
          pd.DataFrame(rows) if rows else None, missing,
          "no feature set was evaluated under both regimes")
    for r in rows:
        log.warning("REGIME GAP: %s -> %.6f static vs %.6f re-tuned (%+.6f)%s",
                    r["feature_set"], r["accuracy_static_regime"],
                    r["accuracy_retuned_regime"], r["difference"],
                    "  [CROSSES THE 0.99 VERDICT LINE]"
                    if r["crosses_0.99_threshold"] else "")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def build_tables() -> dict:
    banner(log, "B6 PAPER TABLES")
    missing: dict[str, str] = {}

    # t1 dataset characteristics
    rows = []
    for name, path in (("phiusiil", "preprocessing_phiusiil_dedup.json"),
                       ("uci", "preprocessing_uci_full.json")):
        p = RESULTS_DIR / path
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"dataset": name, "rows": m["final_shape"][0],
                         "features": m["n_features"],
                         "phishing_rate": round(m["phishing_rate"], 4),
                         "published_phishing_rate": m["published_phishing_rate"],
                         "deduplicated": m.get("deduplicated"),
                         "duplicate_rate": m["duplicates"].get("duplicate_rate")})
    scr = _read(RESULTS_DIR / "screening" / "benchmark_screening_summary.csv")
    if scr is not None:
        for _, r in scr.iterrows():
            if r.get("verdict") == "NOT_ACQUIRED":
                continue
            rows.append({"dataset": r["dataset"], "rows": r.get("n_rows"),
                         "features": r.get("n_features"),
                         "phishing_rate": r.get("phishing_rate"),
                         "published_phishing_rate": "", "deduplicated": "",
                         "duplicate_rate": ""})
    _emit("t1_dataset_characteristics.csv",
          pd.DataFrame(rows) if rows else None, missing,
          "no preprocessing reports found")

    # t2 is the LEAD artifact of the new framing: the population contrast. It must
    # contain PhiUSIIL and UCI as well as the external benchmarks, or the
    # "3 near-oracle features versus 0 elsewhere" claim has no table behind it.
    pop = _read(RESULTS_DIR / "screening" / "benchmark_population_table.csv")
    if pop is None:
        try:
            from src.benchmark_screen import population_table
            pop = population_table()
        except Exception as exc:
            log.warning("could not build the population table: %s", exc)
            pop = None
    _emit("t2_benchmark_screening.csv", pop if pop is not None else scr, missing,
          "B2 screening has produced neither a population table nor a summary")
    if pop is not None:
        indep = pop[pop["is_independent_benchmark"].astype(bool)]
        leaky = indep[indep["n_features_above_95"] > 0]
        log.info("t2 population contrast: %d of %d independent benchmarks carry a "
                 "feature above 0.95 (%s)", len(leaky), len(indep),
                 ", ".join(leaky["dataset"]) or "none")

    # t3 single-feature power, top 15 per dataset
    frames = []
    sf_main = _read(RESULTS_DIR / "single_feature_predictive_power.csv")
    if sf_main is not None:
        for ds, g in sf_main.groupby("dataset") if "dataset" in sf_main.columns \
                else [("combined", sf_main)]:
            frames.append(g.nlargest(15, "accuracy").assign(dataset=ds))
    for p in sorted((RESULTS_DIR / "screening").glob("single_feature_*.csv")):
        g = _read(p)
        if g is not None:
            frames.append(g.nlargest(15, "accuracy").assign(
                dataset=p.stem.replace("single_feature_", "")))
    _emit("t3_single_feature_power.csv",
          pd.concat(frames, ignore_index=True) if frames else None, missing,
          "no single-feature tables found")

    # A1.3's curve holds hyperparameters fixed at the values tuned on the FULL
    # feature set while features are removed, so every row after iteration 0
    # understates what its feature set can do. E1 re-tunes per config. Recording
    # the regime on both tables means the 0.9892-vs-0.9927 gap is explained where
    # a reader meets it, rather than discovered later as an inconsistency.
    prog = _read(LEAK_DIR / "progressive_removal.csv")
    if prog is not None and "hyperparameter_regime" not in prog.columns:
        prog = prog.copy()
        prog["hyperparameter_regime"] = np.where(
            prog["iteration"] == 0, REGIME_RETUNED, REGIME_STATIC)
        prog["regime_note"] = (
            "hyperparameters tuned once on the full 50-feature set and held fixed; "
            "rows after iteration 0 therefore understate their feature set")
    _emit("t4_progressive_removal.csv", prog, missing,
          "A1.3 progressive removal has not completed")

    e1 = _read(RESULTS_DIR / "e1_performance_summary_v2.csv")
    if e1 is not None:
        e1 = e1.copy()
        e1["hyperparameter_regime"] = REGIME_RETUNED
        e1["regime_note"] = "hyperparameters re-tuned on each config's own feature set"
    _emit("t5_performance.csv", e1, missing,
          "A4 (E1 v2) has not run - e1_performance_summary_v2.csv absent")

    _emit_regime_reconciliation(prog, e1, missing)

    tr = _read(RESULTS_DIR / "transfer" / "transfer_summary.csv")
    _emit("t6_transfer.csv", tr, missing, "B1 transfer matrix has not run")

    e2 = _read(RESULTS_DIR / "e2_within_dataset_summary_v2.csv")
    _emit("t7_within_dataset_ceiling.csv", e2, missing,
          "A5 (E2 v2) has not run - e2_within_dataset_summary_v2.csv absent")

    cross = _read(RESULTS_DIR / "cross_dataset_consistency.csv")
    _emit("t8_cross_dataset_consistency.csv", cross, missing,
          "B5 did not run (no clean/repairable partner, or not yet executed)")

    _emit("t9_faithfulness.csv", _read(RESULTS_DIR / "faithfulness.csv"), missing,
          "B4.1 has not run")
    _emit("t10_spurious_injection.csv", _read(RESULTS_DIR / "spurious_injection.csv"),
          missing, "B4.2 has not run")
    _emit("t11_feature_mapping.csv",
          _read(RESULTS_DIR / "transfer" / "feature_mapping.csv"), missing,
          "B1.1 feature mapping has not been built")
    _emit("t12_explanation_cost_profile.csv",
          _read(LEAK_DIR / "explanation_cost_profile.csv"), missing,
          "B5b cost profile has not been built")

    # Not in the brief's list, but the directional inversion is a headline result
    # and needs a table of its own rather than living only inside t11.
    _emit("t13_directional_inconsistency.csv",
          _read(RESULTS_DIR / "transfer" / "directional_inconsistency.csv"), missing,
          "B1.1 polarity analysis has not been built")

    # t14: the concept loss caused by de-leaking. The second artifact the framing
    # rests on - repair strips the strongest shared concepts, which is why the
    # leak-controlled variant is usable for classification but degraded for
    # cross-dataset explanation work.
    lf = RESULTS_DIR / "cross_dataset_leakfree_schema.json"
    if lf.exists():
        m = json.loads(lf.read_text(encoding="utf-8"))
        rows = [{"concept": c, "status": "retained_after_deleaking", "reason": ""}
                for c in m.get("concepts_used", [])]
        rows += [{"concept": c, "status": "LOST_to_deleaking", "reason": why}
                 for c, why in m.get("concepts_lost_to_deleaking", {}).items()]
        pol = _read(RESULTS_DIR / "transfer" / "polarity_agreement.csv")
        tbl = pd.DataFrame(rows)
        if pol is not None:
            tbl = tbl.merge(pol[["concept", "rho_uci", "rho_phiusiil", "status"]]
                            .rename(columns={"status": "polarity_status"}),
                            on="concept", how="left")
        _emit("t14_deleaking_concept_loss.csv", tbl, missing, "")
    else:
        missing["t14_deleaking_concept_loss.csv"] = (
            "B5 leak-free pairing has not run - cross_dataset_leakfree_schema.json absent")

    # t16: runtime observations for the reproducibility section, each labelled
    # clean or contended. Wall-clock is a reported quantity in this paper, so it
    # cannot appear without that label.
    rep = _read(RESULTS_DIR / "reproducibility_notes.csv")
    if rep is None:
        try:
            from src.explanation_cost import reproducibility_notes
            rep = reproducibility_notes()
        except Exception as exc:
            log.warning("could not build reproducibility notes: %s", exc)
    _emit("t16_reproducibility_notes.csv", rep, missing,
          "reproducibility notes have not been generated")

    (TABLE_DIR / "_missing_tables.json").write_text(
        json.dumps(missing, indent=2), encoding="utf-8")
    if missing:
        banner(log, "TABLES NOT EMITTED (evidence gaps, not null results)", "-")
        for k, v in missing.items():
            log.warning("  %-34s %s", k, v)
    return missing


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_single_feature_power() -> bool:
    frames = []
    for p, label in ((RESULTS_DIR / "single_feature_predictive_power.csv", None),):
        df = _read(p)
        if df is not None:
            frames.append(df)
    if not frames:
        return False
    df = frames[0]
    if "dataset" not in df.columns:
        return False
    fig, ax = plt.subplots(figsize=(FULL_W, 3.0))
    for i, (ds, g) in enumerate(df.groupby("dataset")):
        vals = g.nlargest(15, "accuracy")["accuracy"].to_numpy()
        ax.plot(range(1, len(vals) + 1), vals, marker="o", ms=3.5, lw=1.2,
                color=PALETTE[i % len(PALETTE)], label=disp(ds))
    ax.axhline(0.95, ls="--", lw=1.0, color=PALETTE[4],
               label="0.95 single-feature threshold")
    ax.set_xlabel("feature rank"); ax.set_ylabel("single-feature accuracy")
    # No in-figure title: Springer places that in the caption.
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(PAPER_FIG_DIR / "f2_single_feature_power.png")
    plt.close(fig)
    return True


def fig_progressive_removal() -> bool:
    df = _read(LEAK_DIR / "progressive_removal.csv")
    if df is None:
        return False
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(df["iteration"], df["accuracy"], marker="o", ms=3.5, lw=1.3,
            color=PALETTE[0])
    ax.axhline(0.99, ls="--", lw=1.0, color=PALETTE[4], label="0.99 de-leak target")
    ax.set_xlabel("features removed"); ax.set_ylabel("held-out accuracy")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(PAPER_FIG_DIR / "f4_progressive_removal.png")
    plt.close(fig)
    return True


def fig_transfer_matrix() -> bool:
    df = _read(RESULTS_DIR / "transfer" / "transfer_summary.csv")
    if df is None:
        return False
    fig, ax = plt.subplots(figsize=(FULL_W, 3.0))
    x = np.arange(len(df))
    colours = [PALETTE[0] if b else PALETTE[2] for b in df["in_distribution"]]
    ax.bar(x, df["accuracy_mean"], yerr=df["accuracy_std"], capsize=4,
           color=colours, width=0.6, error_kw={"elinewidth": 1.0})
    for i, r in df.iterrows():
        ax.plot([i - 0.3, i + 0.3], [r["majority_class_accuracy"]] * 2,
                ls=":", lw=1.4, color=PALETTE[4])
    pretty = [" \u2192 ".join(disp(part) for part in str(d).split("->"))
              for d in df["direction"]]
    ax.set_xticks(x); ax.set_xticklabels(pretty, rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)

    # The removed title was carrying the key; restore it as a legend outside the
    # plot area so the shading and the dotted line remain interpretable.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PALETTE[0], label="in-distribution"),
               Patch(facecolor=PALETTE[2], label="cross-distribution"),
               Line2D([0], [0], color=PALETTE[4], ls=":", lw=1.4,
                      label="majority-class baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PAPER_FIG_DIR / "f5_transfer_matrix.png")
    plt.close(fig)
    return True


def fig_explanation_cost() -> bool:
    """SHAP seconds and accuracy on twin axes, CLEAN iterations only.

    Contended iterations are plotted as hollow markers and excluded from the
    trend, so the figure cannot silently present a contended timing as a
    measurement. The caption is generated from the data.
    """
    df = _read(LEAK_DIR / "explanation_cost_profile.csv")
    if df is None:
        return False
    df = df[df["source_log"] != "screening"].sort_values("iteration")
    if df.empty:
        return False
    ok = df[df["cost_analysis_eligible"].fillna(False).astype(bool)]
    bad = df[~df["cost_analysis_eligible"].fillna(False).astype(bool)]

    fig, ax = plt.subplots(figsize=(FULL_W, 3.2))
    ax.plot(ok["iteration"], ok["shap_seconds"], marker="o", ms=4, lw=1.4,
            color=PALETTE[0], label="TreeSHAP seconds (clean)")
    if len(bad):
        ax.plot(bad["iteration"], bad["shap_seconds"], marker="o", ms=4, lw=0,
                mfc="none", mec=PALETTE[0], label="excluded (CPU contention)")
    ax.plot(df["iteration"], df["fit_seconds"], marker="s", ms=3, lw=1.2,
            color=PALETTE[2], label="ensemble fit seconds")
    ax.set_xlabel("features removed"); ax.set_ylabel("wall-clock seconds")

    ax2 = ax.twinx()
    ax2.plot(df["iteration"], df["accuracy"], marker="^", ms=3.5, lw=1.2,
             color=PALETTE[3], label="held-out accuracy")
    ax2.set_ylabel("accuracy"); ax2.grid(False)

    # The accuracy series starts at the top-left, where the legend used to sit.
    # Placing it below the axes removes the overlap entirely.
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center", ncol=4,
               fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(PAPER_FIG_DIR / "f8_explanation_cost_curve.png")
    plt.close(fig)

    desc_path = LEAK_DIR / "explanation_cost_description.json"
    if desc_path.exists():
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        (PAPER_FIG_DIR / "f8_caption.txt").write_text(desc["caption"], encoding="utf-8")
        log.info("f8 caption (generated from data): %s", desc["caption"])
    return True



def fig_leak_distributions() -> bool:
    """f3 - per-class distributions of the three features that clear 0.95.

    The decisive panel is URLSimilarityIndex: the legitimate class is a single
    point mass at 100, so the "overlap" between the classes is one atom rather
    than a region. Counts are logarithmic because the classes differ by orders
    of magnitude at most values.
    """
    from src.preprocessing import get_xy
    feats = ["URLSimilarityIndex", "NoOfExternalRef", "LineOfCode"]
    X, y = get_xy("phiusiil")
    if any(f not in X.columns for f in feats):
        return False

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.5))
    for i, (ax, feat) in enumerate(zip(axes, feats)):
        panel_letter(ax, i)
        v = X[feat].to_numpy(dtype=float)
        lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 45)
        ax.hist(np.clip(v[y == 0], lo, hi), bins=bins, color=PALETTE[2],
                edgecolor=PALETTE[1], linewidth=0.4, label="legitimate")
        ax.hist(np.clip(v[y == 1], lo, hi), bins=bins, color=PALETTE[0],
                alpha=0.75, label="phishing")
        ax.set_yscale("log")
        ax.set_xlabel(feat, fontsize=7.5)
        ax.set_ylabel("count (log)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].annotate("legitimate class is a single\npoint mass at 100",
                     xy=(0.04, 0.68), xycoords="axes fraction", fontsize=6.5,
                     color=PALETTE[4])
    fig.tight_layout()
    fig.savefig(PAPER_FIG_DIR / "f3_leak_distributions.png")
    plt.close(fig)
    return True


def fig_within_vs_cross() -> bool:
    """f6 - within-dataset ceiling against cross-dataset agreement.

    Both quantities are Kendall's tau between two SHAP importance rankings, which
    is what makes the comparison legitimate: the only thing that changes is
    whether the two rankings come from two halves of ONE dataset or from two
    DIFFERENT datasets. The gap between the bands is the cost of crossing a
    dataset boundary, and it is wide enough that the axis has to cross zero.
    """
    e2 = _read(RESULTS_DIR / "e2_within_dataset_summary_v2.csv")
    cx = _read(RESULTS_DIR / "cross_dataset_consistency.csv")
    if e2 is None or cx is None:
        return False

    e2s = e2.sort_values("kendall_tau_mean", ascending=False)
    within = [(disp(r["config"]), float(r["kendall_tau_mean"]),
               float(r["kendall_tau_std"])) for _, r in e2s.iterrows()]
    g = cx.groupby("dataset_b")["kendall_tau"].agg(["mean", "std"])
    g = g.sort_values("mean", ascending=False)
    cross = []
    for name, r in g.iterrows():
        sd = float(r["std"]) if pd.notna(r["std"]) else 0.0
        # Every cross pairing is against UCI Phishing Websites, so the tick
        # names only the partner and the band label carries the reference.
        cross.append((disp(name), float(r["mean"]), sd))

    labels = [w[0] for w in within] + [c[0] for c in cross]
    means = [w[1] for w in within] + [c[1] for c in cross]
    errs = [w[2] for w in within] + [c[2] for c in cross]
    n_w = len(within)
    colours = [PALETTE[0]] * n_w + [PALETTE[3]] * len(cross)

    fig, ax = plt.subplots(figsize=(FULL_W, 3.5))
    x = np.arange(len(means))
    ax.bar(x, means, yerr=errs, capsize=4, color=colours, width=0.62,
           error_kw={"elinewidth": 1.1, "ecolor": "#22303f"})
    ax.axhline(0, color="black", lw=0.9)

    ceiling = None
    for lbl, m, _s in within:
        if lbl == disp("uci_full"):
            ceiling = m
    if ceiling is not None:
        ax.axhline(ceiling, ls="--", lw=1.1, color=PALETTE[4])
        ax.annotate("within-dataset ceiling (%s) = %.2f" % (disp("uci_full"), ceiling),
                    xy=(len(means) - 0.45, ceiling), xytext=(0, -12),
                    textcoords="offset points", ha="right", fontsize=7.5,
                    color=PALETTE[4])

    ax.axvline(n_w - 0.5, color=PALETTE[2], lw=1.0, ls=":")
    # Band labels sit INSIDE the axes: at y>1 they collide with the second line
    # of the title.
    ax.text((n_w - 1) / 2.0, 0.975, "WITHIN dataset (two halves)", ha="center",
            va="top", fontsize=8, color=PALETTE[0], fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(n_w + (len(cross) - 1) / 2.0, 0.975,
            "ACROSS datasets (vs UCI Phishing Websites)", ha="center",
            va="top", fontsize=8, color=PALETTE[3], fontweight="bold",
            transform=ax.get_xaxis_transform())

    # Clear the error bar, which is taller than a fixed offset on the noisier
    # cross-dataset bars.
    for xi, m, e in zip(x, means, errs):
        if m >= 0:
            ax.annotate("%.2f" % m, xy=(xi, m + e), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=7)
        else:
            # Below the error bar the label ran into the x tick labels. Place it
            # inside the bar instead, just under the zero line.
            ax.annotate("%.2f" % m, xy=(xi, 0), xytext=(0, -13),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=7, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("Kendall's tau between SHAP rankings")
    # Headroom above the tallest bar has to fit BOTH the value labels and the
    # band labels; 1.12 left them overlapping.
    ax.set_ylim(min(-0.65, min(means) - 0.18), 1.38)
    fig.tight_layout()
    fig.savefig(PAPER_FIG_DIR / "f6_within_vs_cross.png")
    plt.close(fig)
    return True


def fig_group_shap_mass() -> bool:
    """f7 - how attribution mass distributes over the shared concepts.

    Recomputed from the cached cross-dataset SHAP arrays: each feature's share of
    the total mean |SHAP| within each dataset. Where two datasets concentrate
    their explanation weight on different concepts, the rank disagreement in f6
    has a visible cause rather than only a coefficient.
    """
    import glob

    cx = _read(RESULTS_DIR / "cross_dataset_consistency.csv")
    if cx is None:
        return False
    pairings = list(dict.fromkeys([str(v) for v in cx["dataset_b"]]))

    def _profiles(tag):
        """Per-feature mean |SHAP| for every cached run of `tag`, keyed by width.

        The UCI side is cached under one tag across all three pairings, but each
        pairing has a different shared schema, so the arrays differ in width.
        Grouping by width keeps each pairing's runs together instead of averaging
        incompatible shapes.
        """
        out = {}
        for f in sorted(glob.glob("data/cache/shap/cross_%s_seed*__*.npy" % tag)):
            a = np.abs(np.load(f)).mean(axis=0)
            out.setdefault(a.shape[0], []).append(a)
        return out

    def _norm(arrs):
        m = np.mean(arrs, axis=0)
        s = m.sum()
        return m / s if s > 0 else None

    uci_by_width = _profiles("uci")
    panels = []
    for pk in pairings:
        partner = _profiles(pk)
        if not partner:
            continue
        width = max(partner, key=lambda w: len(partner[w]))
        if width not in uci_by_width:
            continue
        a, b = _norm(uci_by_width[width]), _norm(partner[width])
        if a is None or b is None:
            continue
        panels.append((pk, a, b))
    if not panels:
        return False

    from src.cross_dataset import shared_schema, shared_schema_leakfree

    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 3.0), squeeze=False)
    for i, (ax, (pk, a, b)) in enumerate(zip(axes[0], panels)):
        panel_letter(ax, i)
        try:
            if pk == "phiusiil_leakfree":
                names = shared_schema_leakfree()[4]
            else:
                names = shared_schema(pk)[4]
        except Exception:
            names = ["f%d" % i for i in range(a.shape[0])]
        names = [str(n)[:16] for n in list(names)[:a.shape[0]]]
        while len(names) < a.shape[0]:
            names.append("f%d" % len(names))
        idx = np.argsort(-a)
        yy = np.arange(len(idx))
        ax.barh(yy - 0.2, a[idx], height=0.38, color=PALETTE[0],
                label="UCI Phishing Websites (reference)")
        ax.barh(yy + 0.2, b[idx], height=0.38, color=PALETTE[3], label="partner dataset")
        ax.set_yticks(yy)
        ax.set_yticklabels([names[i] for i in idx], fontsize=6.5)
        ax.invert_yaxis()
        # The partner name moves from a panel title into the axis label: it is the
        # only thing distinguishing the panels, and axis labels are permitted.
        ax.set_xlabel("share of total mean |SHAP|\nvs %s" % disp(pk), fontsize=7)
    # One figure-level legend: a per-axes legend overprints the shortest bars, and
    # the partner is already named in each panel title.
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PAPER_FIG_DIR / "f7_group_shap_mass.png")
    plt.close(fig)
    return True


def build_figures() -> dict:
    banner(log, "B6 PAPER FIGURES")
    made, skipped = {}, {}
    # f1 is drawn by hand. Once the real file exists, stop emitting the
    # placeholder beside it - two files for one figure invites picking the wrong one.
    if (PAPER_FIG_DIR / "f1_audit_pipeline.png").exists():
        (PAPER_FIG_DIR / "f1_audit_pipeline_PLACEHOLDER.txt").unlink(missing_ok=True)
        made["f1_audit_pipeline"] = "hand-drawn (left untouched)"
    else:
        (PAPER_FIG_DIR / "f1_audit_pipeline_PLACEHOLDER.txt").write_text(
            "f1_audit_pipeline: to be drawn manually (per the brief).\n",
            encoding="utf-8")
        made["f1_audit_pipeline"] = "placeholder"

    for key, fn, why in (
        ("f2_single_feature_power", fig_single_feature_power,
         "single_feature_predictive_power.csv absent"),
        ("f4_progressive_removal", fig_progressive_removal,
         "A1.3 progressive removal has not completed"),
        ("f3_leak_distributions", fig_leak_distributions,
         "PhiUSIIL prepared frame unavailable"),
        ("f5_transfer_matrix", fig_transfer_matrix, "B1 transfer has not run"),
        ("f6_within_vs_cross", fig_within_vs_cross,
         "needs both the E2 v2 summary and the B5 cross-dataset results"),
        ("f7_group_shap_mass", fig_group_shap_mass,
         "needs cached cross-dataset SHAP arrays"),
        ("f8_explanation_cost_curve", fig_explanation_cost,
         "B5b cost profile has not been built"),
    ):
        try:
            produced = bool(fn())
        except Exception as exc:
            skipped[key] = f"{type(exc).__name__}: {exc}"
            log.error("figure %s failed: %s", key, exc)
            continue
        if not produced:
            skipped[key] = why
            continue
        # Trust the file on disk, not the return value.
        if any(PAPER_FIG_DIR.glob(f"{key}*")):
            made[key] = "ok"
        else:
            skipped[key] = "function reported success but wrote no file"
    log.info("figures written: %s", sorted(made))
    if skipped:
        for k, v in skipped.items():
            log.warning("figure SKIPPED %-28s %s", k, v)
    return {"made": made, "skipped": skipped}


if __name__ == "__main__":
    missing = build_tables()
    figs = build_figures()
    (TABLE_DIR / "_export_report.json").write_text(
        json.dumps({"missing_tables": missing, "figures": figs}, indent=2),
        encoding="utf-8")
    banner(log, "B6 EXPORT COMPLETE", "-")
    log.info("tables  -> %s", TABLE_DIR)
    log.info("figures -> %s", PAPER_FIG_DIR)
