"""Task 3 - single-feature leakage investigation.

PhiUSIIL's URLSimilarityIndex is suspected of being near-deterministic with the
label. If a single feature can reproduce the label almost perfectly, then a 99%
ensemble accuracy says nothing about phishing detection and every SHAP ranking is
dominated by that one column - which would make a consistency study meaningless.

This module quantifies the effect for every feature in both datasets and reports
it. It deliberately does NOT drop anything: the decision about a with/without
variant belongs to the analysis, and silent removal would hide the finding.
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
import seaborn as sns  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    FIGURES_DIR,
    LEAKAGE_ACC_THRESHOLD,
    RESULTS_DIR,
    SEEDS,
    SUSPECTED_LEAKY_FEATURE,
    banner,
    get_logger,
    set_global_seed,
)
from src.preprocessing import get_splits  # noqa: E402

log = get_logger("leakage_check")

# Mutual information is estimated on a subsample: the kNN estimator is O(n^2)-ish
# and 165k rows x 50 features is needlessly slow for a diagnostic.
MI_SAMPLE_SIZE = 20000


def single_feature_tree(splits, feature_idx: int, seed: int, max_depth: int = 3) -> dict:
    """Train a depth-limited tree on ONE feature and score it on the test split."""
    Xtr = splits.X_train[:, [feature_idx]]
    Xte = splits.X_test[:, [feature_idx]]
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree.fit(Xtr, splits.y_train)
    pred = tree.predict(Xte)
    return {
        "accuracy": accuracy_score(splits.y_test, pred),
        "precision": precision_score(splits.y_test, pred, zero_division=0),
        "recall": recall_score(splits.y_test, pred, zero_division=0),
        "f1": f1_score(splits.y_test, pred, zero_division=0),
        "_tree": tree,
    }


def investigate_suspected_feature(dataset: str, seed: int) -> dict | None:
    """Steps 1-3 of Task 3 for URLSimilarityIndex specifically."""
    splits = get_splits(dataset, seed)
    if SUSPECTED_LEAKY_FEATURE not in splits.feature_names:
        log.info("[%s] %s not present - skipping the targeted investigation",
                 dataset, SUSPECTED_LEAKY_FEATURE)
        return None

    banner(log, f"TARGETED LEAKAGE INVESTIGATION: {dataset}.{SUSPECTED_LEAKY_FEATURE}", "-")
    idx = splits.feature_names.index(SUSPECTED_LEAKY_FEATURE)
    res = single_feature_tree(splits, idx, seed)
    tree = res.pop("_tree")

    log.info("[%s] single-feature DecisionTree(max_depth=3) on %s, test-set scores:",
             dataset, SUSPECTED_LEAKY_FEATURE)
    for k in ("accuracy", "precision", "recall", "f1"):
        log.info("    %-10s %.4f", k, res[k])
    log.info("[%s] learned decision rule (thresholds are on STANDARDISED values):\n%s",
             dataset, export_text(tree, feature_names=[SUSPECTED_LEAKY_FEATURE]))

    mi = mutual_information(splits, [idx])[0]
    res["mutual_information"] = mi
    log.info("[%s] mutual information with the label: %.4f nats", dataset, mi)

    _plot_distribution(dataset, splits, idx)
    res.update({"dataset": dataset, "feature": SUSPECTED_LEAKY_FEATURE, "seed": seed})
    return res


def _plot_distribution(dataset: str, splits, feature_idx: int) -> Path:
    """Distribution of the suspected feature split by class."""
    # Plot on the ORIGINAL scale, which is what a reader can interpret.
    raw = splits.scaler.inverse_transform(splits.X_test)[:, feature_idx]
    frame = pd.DataFrame({
        SUSPECTED_LEAKY_FEATURE: raw,
        "class": np.where(splits.y_test == 1, "phishing", "legitimate"),
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(data=frame, x=SUSPECTED_LEAKY_FEATURE, hue="class", bins=50,
                 element="step", stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title(f"{dataset}: {SUSPECTED_LEAKY_FEATURE} by class")
    sns.boxplot(data=frame, x="class", y=SUSPECTED_LEAKY_FEATURE, ax=axes[1])
    axes[1].set_title("Class-conditional distribution")
    fig.suptitle(f"Leakage inspection - {SUSPECTED_LEAKY_FEATURE} ({dataset}, test split)")
    fig.tight_layout()
    out = FIGURES_DIR / "leakage_urlsimilarity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("[%s] distribution figure -> %s", dataset, out)
    return out


def mutual_information(splits, feature_indices: list[int] | None = None) -> np.ndarray:
    """MI between each feature and the label, estimated on the training split."""
    cols = feature_indices if feature_indices is not None else range(splits.X_train.shape[1])
    X = splits.X_train[:, list(cols)]
    y = splits.y_train
    if len(y) > MI_SAMPLE_SIZE:
        rng = np.random.default_rng(splits.seed)
        sel = rng.choice(len(y), size=MI_SAMPLE_SIZE, replace=False)
        X, y = X[sel], y[sel]
    return mutual_info_classif(X, y, random_state=splits.seed)


def rank_all_features(dataset: str, seed: int) -> pd.DataFrame:
    """Step 4: single-feature predictive power for EVERY feature."""
    splits = get_splits(dataset, seed)
    banner(log, f"SINGLE-FEATURE PREDICTIVE POWER: {dataset} (seed {seed})", "-")

    mi_all = mutual_information(splits)
    rows = []
    for i, fname in enumerate(splits.feature_names):
        res = single_feature_tree(splits, i, seed)
        res.pop("_tree")
        rows.append({
            "dataset": dataset,
            "feature": fname,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "mutual_information": mi_all[i],
        })
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    log.info("[%s] top 10 features by single-feature test accuracy:", dataset)
    for _, r in df.head(10).iterrows():
        log.info("    %-30s acc=%.4f f1=%.4f mi=%.4f",
                 r["feature"], r["accuracy"], r["f1"], r["mutual_information"])
    return df


def run_leakage_check(datasets: list[str], seed: int | None = None) -> dict:
    """Full Task 3: targeted investigation plus the ranked table for all features."""
    seed = seed if seed is not None else SEEDS[0]
    set_global_seed(seed)
    banner(log, "TASK 3: LEAKAGE CHECK")

    targeted = {}
    tables = []
    for ds in datasets:
        r = investigate_suspected_feature(ds, seed)
        if r:
            targeted[ds] = r
        tables.append(rank_all_features(ds, seed))

    table = pd.concat(tables, ignore_index=True)
    out_csv = RESULTS_DIR / "single_feature_predictive_power.csv"
    table.to_csv(out_csv, index=False)
    log.info("ranked table -> %s (%d rows)", out_csv, len(table))

    flagged = table[table["accuracy"] > LEAKAGE_ACC_THRESHOLD].copy()
    banner(log, "LEAKAGE VERDICT")
    if len(flagged):
        log.warning("*" * 78)
        log.warning("LEAKAGE WARNING: %d feature(s) exceed %.2f single-feature test accuracy.",
                    len(flagged), LEAKAGE_ACC_THRESHOLD)
        log.warning("A depth-3 tree on ONE of these columns nearly reproduces the label,")
        log.warning("so headline ensemble accuracy is not evidence of phishing detection")
        log.warning("and SHAP rankings will be dominated by these features.")
        for _, r in flagged.iterrows():
            log.warning("    %-10s %-30s acc=%.4f  f1=%.4f  mi=%.4f",
                        r["dataset"], r["feature"], r["accuracy"], r["f1"],
                        r["mutual_information"])
        log.warning("NOTHING HAS BEEN DROPPED. Whether the main analysis needs a")
        log.warning("with/without variant is a reporting decision, not a silent fix.")
        log.warning("*" * 78)
    else:
        log.info("No feature exceeds %.2f single-feature accuracy in either dataset.",
                 LEAKAGE_ACC_THRESHOLD)

    summary = {
        "seed": seed,
        "threshold": LEAKAGE_ACC_THRESHOLD,
        "targeted": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                         for kk, vv in v.items()} for k, v in targeted.items()},
        "flagged_features": flagged[["dataset", "feature", "accuracy", "f1",
                                     "mutual_information"]].to_dict("records"),
        "n_flagged": int(len(flagged)),
    }
    (RESULTS_DIR / "leakage_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task 3: single-feature leakage check")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    ds = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
    run_leakage_check(ds, seed=args.seed)
