"""Task A1 - characterisation of the PhiUSIIL label leak.

Phase A produced accuracy = 1.0000 +/- 0.0000 for four algorithms across five
seeds on ~35,000 held-out PhiUSIIL instances, while the identical pipeline gave a
wholly plausible 0.9533 on UCI. The pipeline is therefore not at fault and the
problem is a property of PhiUSIIL itself.

This module answers three questions:

  A1.1  How are the suspect features distributed WITHIN each class? A feature
        that is constant inside a class is not a predictor, it is a restatement
        of the label.
  A1.2  Is there a single threshold that separates the classes, and do the two
        classes overlap at all? Disjoint ranges are the decisive evidence: a
        predictor has an overlap region, a label proxy does not.
  A1.3  What is the MINIMAL set of features whose removal destroys perfect
        separation? This determines whether PhiUSIIL is salvageable at all.

Nothing here silently drops or repairs anything. The outputs are evidence.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    FIGURES_DIR,
    LEAK_DIR,
    PROJECT_ROOT,
    SEEDS,
    banner,
    get_logger,
    set_global_seed,
    write_csv_atomic,
)
from src.models import ENSEMBLE_NAME, build_models  # noqa: E402
from src.preprocessing import get_splits, get_xy  # noqa: E402
from src.shap_utils import (  # noqa: E402
    compute_shap_values,
    global_importance,
    make_background,
    select_shap_sample,
)
from src.tuning import get_params_for_models  # noqa: E402

log = get_logger("leak_analysis")

# The three features that cleared 0.95 single-feature accuracy in Task 3.
LEAK_SUSPECTS = ["URLSimilarityIndex", "NoOfExternalRef", "LineOfCode"]

CLASS_NAMES = {0: "legitimate", 1: "phishing"}

# A1.3 stopping rules, quoted from the brief.
DELEAK_ACC_TARGET = 0.99      # keep removing while accuracy stays at/above this
DELEAK_MAX_REMOVED = 15       # give up beyond this many removals

# Sample used to rank features by mean |SHAP| INSIDE the greedy loop. The loop
# needs only the arg-max of the ranking, which is far more stable than the full
# ranking, and it retrains from scratch on every iteration - so the full 10k
# explanation sample would multiply runtime for no change in the decision. The
# final reported feature set is re-validated at full size afterwards.
GREEDY_SHAP_SAMPLE = 2000


# --------------------------------------------------------------------------- #
# A1.1 Per-class value distributions
# --------------------------------------------------------------------------- #
def per_class_distributions(dataset: str = "phiusiil",
                            features: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distribution of each suspect feature separately within each class.

    Returns (summary, top_values). The summary carries one row per
    (feature, class); top_values carries one row per (feature, class, rank) for
    the ten most frequent values.
    """
    features = features or LEAK_SUSPECTS
    banner(log, f"A1.1 PER-CLASS VALUE DISTRIBUTIONS: {dataset}")

    X, y = get_xy(dataset)
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise KeyError(f"[{dataset}] suspect features absent: {missing}")

    summary_rows, top_rows = [], []
    for feat in features:
        col = X[feat]
        log.info("")
        log.info("--- %s ---", feat)
        for cls in (1, 0):
            v = col[y == cls]
            counts = v.value_counts().sort_values(ascending=False)
            n = len(v)
            n_distinct = int(col[y == cls].nunique())
            most_common_value = counts.index[0]
            most_common_share = float(counts.iloc[0] / n)
            is_constant = n_distinct == 1

            summary_rows.append({
                "dataset": dataset, "feature": feat,
                "class": cls, "class_name": CLASS_NAMES[cls], "n": n,
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "median": float(v.median()),
                "std": float(v.std(ddof=1)) if n > 1 else 0.0,
                "n_distinct": n_distinct,
                "is_constant_within_class": bool(is_constant),
                "most_common_value": float(most_common_value),
                "most_common_count": int(counts.iloc[0]),
                "most_common_share": most_common_share,
            })

            log.info("  %-11s n=%-7d  min=%-10.4f max=%-10.4f mean=%-10.4f median=%-10.4f "
                     "std=%-10.4f distinct=%d", CLASS_NAMES[cls], n, v.min(), v.max(),
                     v.mean(), v.median(), v.std(ddof=1) if n > 1 else 0.0, n_distinct)
            if is_constant:
                log.warning("  %-11s CONSTANT WITHIN CLASS: every one of the %d %s "
                            "instances takes the single value %s. This feature does not "
                            "predict the class, it restates it.",
                            CLASS_NAMES[cls], n, CLASS_NAMES[cls], most_common_value)
            log.info("  %-11s most common value %s covers %.4f%% of the class",
                     CLASS_NAMES[cls], most_common_value, 100 * most_common_share)

            head = counts.head(10)
            log.info("  %-11s top-10 values: %s", CLASS_NAMES[cls],
                     ", ".join(f"{val:g}x{cnt}" for val, cnt in head.items()))
            for rank, (val, cnt) in enumerate(head.items(), start=1):
                top_rows.append({
                    "dataset": dataset, "feature": feat, "class": cls,
                    "class_name": CLASS_NAMES[cls], "rank": rank,
                    "value": float(val), "count": int(cnt),
                    "percent_of_class": round(100 * cnt / n, 6),
                })

    summary = pd.DataFrame(summary_rows)
    top_values = pd.DataFrame(top_rows)

    # The spec names one file; the top-10 values are long-form and do not fit one
    # row per (feature, class), so they get a companion file and a compact
    # summary column here.
    compact = (top_values.groupby(["feature", "class"])
               .apply(lambda g: "; ".join(f"{r.value:g}:{r.count}({r.percent_of_class:.2f}%)"
                                          for r in g.itertuples()), include_groups=False)
               .rename("top10_values").reset_index())
    summary = summary.merge(compact, on=["feature", "class"], how="left")

    LEAK_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(LEAK_DIR / "per_class_distributions.csv", index=False)
    top_values.to_csv(LEAK_DIR / "per_class_top_values.csv", index=False)
    log.info("")
    log.info("-> %s", LEAK_DIR / "per_class_distributions.csv")
    log.info("-> %s", LEAK_DIR / "per_class_top_values.csv")
    return summary, top_values


# --------------------------------------------------------------------------- #
# A1.2 Separability thresholds
# --------------------------------------------------------------------------- #
def _best_threshold(values: np.ndarray, y: np.ndarray) -> dict:
    """Threshold maximising balanced accuracy, scanning both decision directions.

    Implemented with cumulative class counts over the sorted unique values rather
    than a Python loop, so it stays exact on 235k rows. For each candidate cut c
    the rule "phishing if x <= c" is evaluated, and its mirror image, and the best
    of the two directions is returned.
    """
    order = np.argsort(values, kind="stable")
    v_sorted = values[order]
    y_sorted = y[order]

    uniq, first_idx = np.unique(v_sorted, return_index=True)
    # cumulative count of positives/negatives at or below each unique value
    cum_pos = np.cumsum(y_sorted == 1)
    cum_neg = np.cumsum(y_sorted == 0)
    last_idx = np.append(first_idx[1:], len(v_sorted)) - 1
    pos_le = cum_pos[last_idx].astype(float)
    neg_le = cum_neg[last_idx].astype(float)

    total_pos = float((y == 1).sum())
    total_neg = float((y == 0).sum())

    # Direction A: predict phishing when x <= c
    tp_a, fp_a = pos_le, neg_le
    fn_a, tn_a = total_pos - pos_le, total_neg - neg_le
    bal_a = 0.5 * (tp_a / total_pos + tn_a / total_neg)

    # Direction B: predict phishing when x > c
    tp_b, fp_b = total_pos - pos_le, total_neg - neg_le
    fn_b, tn_b = pos_le, neg_le
    bal_b = 0.5 * (tp_b / total_pos + tn_b / total_neg)

    if bal_a.max() >= bal_b.max():
        i = int(np.argmax(bal_a))
        tp, fp, fn, tn = tp_a[i], fp_a[i], fn_a[i], tn_a[i]
        direction, bal = "phishing if x <= t", float(bal_a[i])
    else:
        i = int(np.argmax(bal_b))
        tp, fp, fn, tn = tp_b[i], fp_b[i], fn_b[i], tn_b[i]
        direction, bal = "phishing if x > t", float(bal_b[i])

    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": float(uniq[i]), "direction": direction,
        "balanced_accuracy": bal, "accuracy": float(acc),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        # "wrong side" counts: legitimate rows landing in the phishing region
        # and vice versa. Both zero means the threshold separates perfectly.
        "legitimate_on_phishing_side": int(fp),
        "phishing_on_legitimate_side": int(fn),
    }


def separability_thresholds(dataset: str = "phiusiil",
                            features: list[str] | None = None) -> pd.DataFrame:
    """Single-threshold separability plus the decisive class-overlap test."""
    features = features or LEAK_SUSPECTS
    banner(log, f"A1.2 SEPARABILITY THRESHOLDS: {dataset}")

    X, y = get_xy(dataset)
    yv = y.to_numpy()
    rows = []
    for feat in features:
        v = X[feat].to_numpy(dtype=float)
        res = _best_threshold(v, yv)

        vals_p = set(np.unique(v[yv == 1]).tolist())
        vals_l = set(np.unique(v[yv == 0]).tolist())
        shared = sorted(vals_p & vals_l)
        disjoint = len(shared) == 0

        n_rows_in_shared = int(np.isin(v, list(shared)).sum()) if shared else 0

        # A plain disjoint/overlap flag mis-describes the decisive case. When one
        # class collapses to a single value and the other class's support strictly
        # contains it, the "overlap" is a single atom rather than a region, and one
        # side of the threshold is perfectly pure. That is a label proxy just as
        # much as literal disjointness is, so it gets its own category.
        one_sided_pure = (res["legitimate_on_phishing_side"] == 0
                          or res["phishing_on_legitimate_side"] == 0)
        if disjoint:
            sep_type = "disjoint"
        elif len(shared) == 1 and (len(vals_p) == 1 or len(vals_l) == 1) and one_sided_pure:
            sep_type = "degenerate_point_mass"
        elif one_sided_pure:
            sep_type = "one_sided_pure"
        else:
            sep_type = "genuine_overlap"

        rec = {
            "dataset": dataset, "feature": feat, **res,
            "separation_type": sep_type,
            "phishing_min": float(v[yv == 1].min()), "phishing_max": float(v[yv == 1].max()),
            "legitimate_min": float(v[yv == 0].min()), "legitimate_max": float(v[yv == 0].max()),
            "n_distinct_phishing": len(vals_p),
            "n_distinct_legitimate": len(vals_l),
            "n_shared_values": len(shared),
            "classes_disjoint": bool(disjoint),
            "n_rows_at_shared_values": n_rows_in_shared,
            "shared_value_examples": "; ".join(f"{s:g}" for s in shared[:10]),
        }
        rows.append(rec)

        log.info("")
        log.info("--- %s ---", feat)
        log.info("  best threshold t=%g  (%s)", res["threshold"], res["direction"])
        log.info("  accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f (balanced acc %.4f)",
                 res["accuracy"], res["precision"], res["recall"], res["f1"],
                 res["balanced_accuracy"])
        log.info("  legitimate on the phishing side: %d", res["legitimate_on_phishing_side"])
        log.info("  phishing on the legitimate side: %d", res["phishing_on_legitimate_side"])
        log.info("  phishing range   [%g, %g] over %d distinct values",
                 rec["phishing_min"], rec["phishing_max"], rec["n_distinct_phishing"])
        log.info("  legitimate range [%g, %g] over %d distinct values",
                 rec["legitimate_min"], rec["legitimate_max"], rec["n_distinct_legitimate"])
        if sep_type == "disjoint":
            log.warning("  DISJOINT: the two classes share NO value of %s. There is no "
                        "overlap region at all, so this feature is a label proxy rather "
                        "than a predictor.", feat)
        elif sep_type == "degenerate_point_mass":
            constant_cls = "legitimate" if len(vals_l) == 1 else "phishing"
            log.warning("  DEGENERATE POINT MASS: the classes are not literally disjoint, "
                        "but they share exactly ONE value (%s) and the %s class consists of "
                        "nothing else. The overlap is a single atom, not a region.",
                        rec["shared_value_examples"], constant_cls)
            log.warning("  Every value other than %s belongs to one class only, and %d "
                        "instances sit on the wrong side of the threshold in total. This is "
                        "a label proxy, not a predictor.", rec["shared_value_examples"],
                        res["legitimate_on_phishing_side"] + res["phishing_on_legitimate_side"])
        elif sep_type == "one_sided_pure":
            log.warning("  ONE-SIDED PURE: %d values are shared, but one side of the best "
                        "threshold contains no contamination at all.", len(shared))
        else:
            log.info("  GENUINE OVERLAP: %d values (%d rows) occur in both classes, with "
                     "contamination on BOTH sides of the threshold (%d legitimate, %d "
                     "phishing). This is a strong predictor, not a label proxy.",
                     len(shared), n_rows_in_shared, res["legitimate_on_phishing_side"],
                     res["phishing_on_legitimate_side"])

    df = pd.DataFrame(rows)
    df.to_csv(LEAK_DIR / "separability_thresholds.csv", index=False)
    log.info("")
    log.info("-> %s", LEAK_DIR / "separability_thresholds.csv")
    return df


def plot_suspect_distributions(dataset: str = "phiusiil",
                               features: list[str] | None = None) -> Path:
    """Per-class histograms for the suspect features, on a shared log count axis."""
    features = features or LEAK_SUSPECTS
    X, y = get_xy(dataset)
    fig, axes = plt.subplots(1, len(features), figsize=(5.0 * len(features), 4.2))
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, features):
        v = X[feat].to_numpy(dtype=float)
        bins = np.histogram_bin_edges(v, bins=60)
        ax.hist(v[y == 0], bins=bins, alpha=0.65, label="legitimate", color="#4C72B0")
        ax.hist(v[y == 1], bins=bins, alpha=0.65, label="phishing", color="#C44E52")
        ax.set_yscale("log")
        ax.set_xlabel(feat)
        ax.set_ylabel("count (log)")
        ax.legend()
    fig.suptitle(f"A1.1/A1.2 - per-class distributions of the leak suspects ({dataset})")
    fig.tight_layout()
    out = FIGURES_DIR / "leak_suspect_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("-> %s", out)
    return out


# --------------------------------------------------------------------------- #
# A1.3 Progressive leak removal
# --------------------------------------------------------------------------- #
def _fit_score_rank(dataset: str, seed: int, keep: list[str], params: dict,
                    shap_sample: int) -> tuple[dict, pd.Series]:
    """Train the tuned ensemble on `keep`, score on test, rank `keep` by mean |SHAP|."""
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    splits = get_splits(dataset, seed, features=keep)
    ens = build_models(seed, params)[ENSEMBLE_NAME]
    t0 = time.perf_counter()
    ens.fit(splits.X_train, splits.y_train)
    fit_s = time.perf_counter() - t0

    y_pred = ens.predict(splits.X_test)
    scores = {
        "accuracy": float(accuracy_score(splits.y_test, y_pred)),
        "f1": float(f1_score(splits.y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(splits.y_test, y_pred)),
        "fit_seconds": round(fit_s, 2),
    }

    background = make_background(splits.X_train, splits.y_train, seed)
    X_explain, _ = select_shap_sample(splits.X_test, splits.y_test, seed=seed, n=shap_sample)
    t0 = time.perf_counter()
    values = compute_shap_values(
        ens, X_explain, seed, background=background,
        cache_key=f"leak_{dataset}_seed{seed}_k{len(keep)}")
    scores["shap_seconds"] = round(time.perf_counter() - t0, 2)
    imp = global_importance(values, splits.feature_names).sort_values(ascending=False)
    return scores, imp


PROGRESSIVE_COLS = ["iteration", "n_features_remaining", "removed_feature",
                    "accuracy", "f1", "mcc", "top_feature_after"]


def _write_progressive(rows: list[dict]) -> pd.DataFrame:
    """Persist the table built so far, after EVERY iteration.

    Each iteration costs 10-20 minutes, so a crash near the cap would otherwise
    discard hours of completed work. `top_feature_after` is derived here rather
    than in a post-loop pass: row i's value is row i+1's `top_feature_before`,
    and the final row's is left blank until the next iteration supplies it.
    """
    out = [dict(r) for r in rows]
    for i in range(len(out) - 1):
        out[i]["top_feature_after"] = out[i + 1]["top_feature_before"]
    if out:
        out[-1].setdefault("top_feature_after", "")
    df = pd.DataFrame(out)
    df = df[PROGRESSIVE_COLS + [c for c in df.columns if c not in PROGRESSIVE_COLS]]
    write_csv_atomic(df, LEAK_DIR / "progressive_removal.csv")
    return df


def progressive_removal(dataset: str = "phiusiil", seed: int = 42,
                        max_removed: int = DELEAK_MAX_REMOVED,
                        acc_target: float = DELEAK_ACC_TARGET,
                        shap_sample: int = GREEDY_SHAP_SAMPLE) -> pd.DataFrame:
    """Greedily remove the top-|SHAP| feature until perfect separation collapses.

    Each iteration trains the tuned ensemble on the surviving features, records
    held-out performance, then removes whichever surviving feature the model leans
    on hardest. The loop stops as soon as accuracy falls below `acc_target`, or
    after `max_removed` removals (which would mean the leakage is diffuse and the
    dataset cannot be repaired by feature removal).
    """
    banner(log, f"A1.3 PROGRESSIVE LEAK REMOVAL: {dataset} (seed {seed})")
    set_global_seed(seed)
    params = get_params_for_models(dataset)
    if not params:
        log.warning("[%s] no tuned hyperparameters found - the greedy loop will use "
                    "library defaults, which is not the configuration in the paper.", dataset)

    X, _ = get_xy(dataset)
    all_features = list(X.columns)
    removed: list[str] = []
    rows: list[dict] = []
    outcome = "unresolved"

    log.info("[%s] starting from %d features, target accuracy < %.2f, cap %d removals",
             dataset, len(all_features), acc_target, max_removed)
    log.info("[%s] in-loop SHAP ranking uses %d instances (see GREEDY_SHAP_SAMPLE)",
             dataset, shap_sample)

    for iteration in range(max_removed + 1):
        keep = [f for f in all_features if f not in removed]
        if len(keep) < 2:
            outcome = "exhausted_features"
            break

        scores, imp = _fit_score_rank(dataset, seed, keep, params, shap_sample)
        top_feature = str(imp.index[0])

        row = {
            "iteration": iteration,
            "n_features_remaining": len(keep),
            "removed_so_far": len(removed),
            "removed_features_so_far": "; ".join(removed),
            "accuracy": scores["accuracy"], "f1": scores["f1"], "mcc": scores["mcc"],
            "top_feature_before": top_feature,
            "top_shap_before": float(imp.iloc[0]),
            "second_feature": str(imp.index[1]),
            "second_shap": float(imp.iloc[1]),
            "fit_seconds": scores["fit_seconds"], "shap_seconds": scores["shap_seconds"],
        }

        log.info("[iter %2d] n_features=%2d acc=%.6f f1=%.6f mcc=%.6f | top=%s (%.5f) "
                 "| fit %.1fs shap %.1fs", iteration, len(keep), scores["accuracy"],
                 scores["f1"], scores["mcc"], top_feature, imp.iloc[0],
                 scores["fit_seconds"], scores["shap_seconds"])

        if scores["accuracy"] < acc_target:
            row["removed_feature"] = ""
            row["top_feature_after"] = ""
            rows.append(row)
            _write_progressive(rows)
            outcome = "deleaked"
            log.info("[%s] STOP: accuracy %.6f fell below %.2f after removing %d "
                     "feature(s): %s", dataset, scores["accuracy"], acc_target,
                     len(removed), removed or "none")
            break

        if len(removed) >= max_removed:
            row["removed_feature"] = ""
            row["top_feature_after"] = ""
            rows.append(row)
            _write_progressive(rows)
            outcome = "failed_to_deleak"
            log.warning("[%s] STOP: %d features removed and accuracy is still %.6f "
                        ">= %.2f.", dataset, len(removed), scores["accuracy"], acc_target)
            break

        row["removed_feature"] = top_feature
        rows.append(row)
        removed.append(top_feature)
        # Written after every iteration, not once at the end: each iteration costs
        # 10-20 minutes and a late crash would otherwise lose the whole run.
        _write_progressive(rows)
        log.info("[iter %2d] progressive_removal.csv updated (%d rows)", iteration, len(rows))

    df = _write_progressive(rows)
    log.info("-> %s", LEAK_DIR / "progressive_removal.csv")

    final_acc = float(df["accuracy"].iloc[-1])
    _report_outcome(dataset, outcome, removed, final_acc, df)

    (LEAK_DIR / "progressive_removal_outcome.json").write_text(json.dumps({
        "dataset": dataset, "seed": seed, "outcome": outcome,
        "removed_features": removed, "n_removed": len(removed),
        "final_accuracy": final_acc,
        "final_f1": float(df["f1"].iloc[-1]), "final_mcc": float(df["mcc"].iloc[-1]),
        "n_features_remaining": int(df["n_features_remaining"].iloc[-1]),
        "acc_target": acc_target, "max_removed": max_removed,
        "greedy_shap_sample": shap_sample,
    }, indent=2), encoding="utf-8")
    return df


def _report_outcome(dataset: str, outcome: str, removed: list[str], final_acc: float,
                    df: pd.DataFrame) -> None:
    """Print which of the brief's three outcomes occurred, in plain terms."""
    banner(log, "A1.3 OUTCOME")
    accs = df["accuracy"].to_numpy(dtype=float)
    drop = float(accs[0] - accs[-1])

    if outcome == "failed_to_deleak":
        log.warning("OUTCOME 2 - DIFFUSE LEAKAGE, NOT REPAIRABLE BY FEATURE REMOVAL.")
        log.warning("%d features were removed and held-out accuracy never fell below "
                    "%.2f (final %.6f). PhiUSIIL cannot be de-leaked this way. This is a "
                    "significant finding about the benchmark and must be reported as such; "
                    "PHIUSIIL_LEAKFREE_FEATURES stays None and A4/A5 must not be run for "
                    "the leak-free config.", len(removed), DELEAK_ACC_TARGET, final_acc)
    elif final_acc < 0.80:
        log.warning("OUTCOME 3 - ACCURACY COLLAPSE.")
        log.warning("After removing %d feature(s) %s accuracy fell to %.6f, below 0.80. "
                    "The surviving features carry little signal; the cliff occurs at "
                    "iteration %d (%.6f -> %.6f).", len(removed), removed,
                    final_acc, len(accs) - 1, accs[-2] if len(accs) > 1 else accs[-1],
                    accs[-1])
    elif 0.95 <= final_acc <= 0.98:
        log.info("OUTCOME 1 - USABLE LEAK-CONTROLLED VARIANT (the good outcome).")
        log.info("Removing %d feature(s) %s brought accuracy from %.6f to %.6f, inside the "
                 "plausible 0.95-0.98 band. PhiUSIIL is usable in a leak-controlled "
                 "variant.", len(removed), removed, accs[0], final_acc)
    else:
        log.warning("OUTCOME: BETWEEN THE BRIEF'S CASES.")
        log.warning("Removing %d feature(s) %s took accuracy from %.6f to %.6f (drop %.4f), "
                    "which is below the 0.99 target but outside the 0.95-0.98 band the "
                    "brief anticipated. Reported as-is rather than forced into a category.",
                    len(removed), removed, accs[0], final_acc, drop)


# --------------------------------------------------------------------------- #
# A1.4 Persist the leak-free feature set into config.py
# --------------------------------------------------------------------------- #
BEGIN_MARK = "# >>>BEGIN GENERATED: PHIUSIIL_LEAKFREE (Task A1.4)>>>"
END_MARK = "# <<<END GENERATED<<<"


def write_leakfree_config(removed: list[str], final_acc: float, outcome: str,
                          dataset: str = "phiusiil", seed: int = 42) -> bool:
    """Rewrite the generated block in config.py with the leak-free feature set.

    Returns True when a feature set was written, False when the analysis declined
    to define one (outcome 2), in which case the constant stays None by design.
    """
    banner(log, "A1.4 LEAK-FREE FEATURE SET")
    cfg_path = PROJECT_ROOT / "config.py"
    text = cfg_path.read_text(encoding="utf-8")
    if BEGIN_MARK not in text or END_MARK not in text:
        raise RuntimeError(f"generated-block markers not found in {cfg_path}")

    if outcome == "failed_to_deleak":
        log.warning("Leakage was not eliminated by removing %d features, so "
                    "PHIUSIIL_LEAKFREE_FEATURES stays None and the phiusiil_leakfree "
                    "config remains unavailable. Stopping before Task A4 as instructed.",
                    len(removed))
        body = (
            f"{BEGIN_MARK}\n"
            f"# A1.3 outcome: {outcome}. Removing {len(removed)} features left held-out\n"
            f"# accuracy at {final_acc:.6f}, so no leak-free feature set could be defined.\n"
            f"PHIUSIIL_LEAKFREE_FEATURES: list[str] | None = None\n"
            f"PHIUSIIL_LEAKFREE_EVIDENCE: str | None = (\n"
            f'    "A1.3 progressive removal failed to de-leak {dataset}: {len(removed)} "\n'
            f'    "features removed, final accuracy {final_acc:.6f}."\n'
            f")\n"
            f"{END_MARK}"
        )
        written = False
    else:
        X, _ = get_xy(dataset)
        keep = [c for c in X.columns if c not in removed]
        # Each entry already carries its trailing comma, so the separator must be a
        # bare newline. Joining with ",\n" produced `"Feature",,` and a SyntaxError
        # that broke every import of config.py.
        feats = "\n".join(f'    "{f}",' for f in keep)
        evidence = (f"A1.3 progressive removal (seed {seed}): removing "
                    f"{', '.join(removed)} dropped held-out ensemble accuracy from "
                    f"1.000000 to {final_acc:.6f}.")
        log.info("Leak-free set: %d features (removed %d: %s), accuracy %.6f",
                 len(keep), len(removed), ", ".join(removed), final_acc)
        body = (
            f"{BEGIN_MARK}\n"
            f"# Evidence: {evidence}\n"
            f"# Removed as leaking: {', '.join(removed)}\n"
            f"# Retained: {len(keep)} of {len(X.columns)} features.\n"
            f"PHIUSIIL_LEAKFREE_FEATURES: list[str] | None = [\n{feats}\n]\n"
            f"PHIUSIIL_LEAKFREE_EVIDENCE: str | None = (\n"
            f'    "{evidence}"\n'
            f")\n"
            f"{END_MARK}"
        )
        written = True

    start = text.index(BEGIN_MARK)
    end = text.index(END_MARK) + len(END_MARK)
    cfg_path.write_text(text[:start] + body + text[end:], encoding="utf-8")
    log.info("config.py updated (generated block rewritten)")
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_leak_analysis(dataset: str = "phiusiil", seed: int = 42,
                      skip_progressive: bool = False) -> dict:
    banner(log, f"TASK A1: LEAK CHARACTERISATION - {dataset}")
    summary, _ = per_class_distributions(dataset)
    thresholds = separability_thresholds(dataset)
    plot_suspect_distributions(dataset)

    out = {"per_class": summary.to_dict("records"),
           "thresholds": thresholds.to_dict("records")}
    if skip_progressive:
        log.warning("A1.3 skipped by request - no leak-free feature set will be defined.")
        return out

    df = progressive_removal(dataset, seed=seed)
    meta = json.loads((LEAK_DIR / "progressive_removal_outcome.json").read_text())
    write_leakfree_config(meta["removed_features"], meta["final_accuracy"],
                          meta["outcome"], dataset=dataset, seed=seed)
    out["progressive_removal"] = df.to_dict("records")
    out["outcome"] = meta
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task A1: PhiUSIIL leak characterisation")
    ap.add_argument("--dataset", default="phiusiil")
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    ap.add_argument("--stage", default="all",
                    choices=["all", "distributions", "thresholds", "progressive"])
    ap.add_argument("--max-removed", type=int, default=DELEAK_MAX_REMOVED)
    ap.add_argument("--shap-sample", type=int, default=GREEDY_SHAP_SAMPLE)
    args = ap.parse_args()

    if args.stage in ("all", "distributions"):
        per_class_distributions(args.dataset)
    if args.stage in ("all", "thresholds"):
        separability_thresholds(args.dataset)
        plot_suspect_distributions(args.dataset)
    if args.stage in ("all", "progressive"):
        d = progressive_removal(args.dataset, seed=args.seed,
                                max_removed=args.max_removed,
                                shap_sample=args.shap_sample)
        m = json.loads((LEAK_DIR / "progressive_removal_outcome.json").read_text())
        write_leakfree_config(m["removed_features"], m["final_accuracy"], m["outcome"],
                              dataset=args.dataset, seed=args.seed)
