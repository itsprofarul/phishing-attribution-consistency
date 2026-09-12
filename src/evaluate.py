"""Task 6 (E1) - performance evaluation across datasets, models and seeds.

Every (dataset, model, seed) cell is trained on its own training split and scored
once on the held-out test split. The five seeds vary both the split and the model
randomness, so the reported std is the honest run-to-run spread rather than a
single lucky split.
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
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
    SEEDS,
    available_configs,
    banner,
    get_logger,
    resolve_config,
    set_global_seed,
    write_csv_atomic,
)
from src.models import ALL_MODEL_NAMES, DISPLAY_NAMES, ENSEMBLE_NAME, build_models  # noqa: E402
from src.preprocessing import get_splits, get_splits_for_config  # noqa: E402
from src.tuning import get_params_for_config, get_params_for_models  # noqa: E402

log = get_logger("evaluate")

METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]

# Published work reports ~99% accuracy on both datasets; anything far below that
# means something is wrong upstream rather than something interesting downstream.
SANITY_ACCURACY_FLOOR = 0.90


def score_predictions(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_evaluate(dataset: str, seed: int, params: dict | None = None,
                       keep_models: bool = False) -> tuple[list[dict], dict, dict]:
    """Train all four estimators for one (dataset, seed) and score them on test.

    Returns (rows, confusion_matrices, fitted_models). `fitted_models` is empty
    unless keep_models=True - the PhiUSIIL forests are large and holding five
    seeds' worth in memory is pointless when only one seed feeds the SHAP checks.
    """
    set_global_seed(seed)
    splits = get_splits(dataset, seed)
    params = params if params is not None else get_params_for_models(dataset)
    models = build_models(seed, params)

    rows, cms, fitted = [], {}, {}
    for name in ALL_MODEL_NAMES:
        est = models[name]
        t0 = time.perf_counter()
        est.fit(splits.X_train, splits.y_train)
        fit_s = time.perf_counter() - t0

        y_pred = est.predict(splits.X_test)
        y_proba = est.predict_proba(splits.X_test)[:, 1]
        scores = score_predictions(splits.y_test, y_pred, y_proba)

        rows.append({"dataset": dataset, "model": name, "seed": seed,
                     "fit_seconds": round(fit_s, 2), "n_train": len(splits.y_train),
                     "n_test": len(splits.y_test), **scores})
        cms[name] = confusion_matrix(splits.y_test, y_pred, labels=[0, 1])
        if keep_models:
            fitted[name] = est

        log.info("[%s seed=%d] %-18s acc=%.4f prec=%.4f rec=%.4f f1=%.4f mcc=%.4f auc=%.4f "
                 "(fit %.1fs)", dataset, seed, DISPLAY_NAMES[name], scores["accuracy"],
                 scores["precision"], scores["recall"], scores["f1"], scores["mcc"],
                 scores["roc_auc"], fit_s)

        if scores["accuracy"] < SANITY_ACCURACY_FLOOR:
            log.warning("[%s seed=%d] %s accuracy %.4f is far below the ~0.99 reported in "
                        "published work - investigate upstream (labels, splits, scaling) "
                        "before trusting this number.", dataset, seed,
                        DISPLAY_NAMES[name], scores["accuracy"])
    return rows, cms, fitted


def plot_confusion(dataset: str, model: str, cm: np.ndarray, n_seeds: int) -> Path:
    """Confusion matrix summed over seeds, annotated with counts and row shares."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    total_per_row = cm.sum(axis=1, keepdims=True)
    shares = np.divide(cm, np.maximum(total_per_row, 1))
    labels = np.array([[f"{cm[i, j]:,}\n({shares[i, j]:.2%})" for j in range(2)]
                       for i in range(2)])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["legitimate", "phishing"],
                yticklabels=["legitimate", "phishing"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{dataset} - {DISPLAY_NAMES[model]}\n(summed over {n_seeds} seeds)")
    fig.tight_layout()
    out = FIGURES_DIR / f"confusion_{dataset}_{model}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std across seeds for every (dataset, model)."""
    g = df.groupby(["dataset", "model"])[METRICS]
    summary = g.agg(["mean", "std"]).round(5)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    # Present base learners in their declared order with the ensemble last, rather
    # than groupby's alphabetical order - this table goes into the paper.
    order = {name: i for i, name in enumerate(ALL_MODEL_NAMES)}
    summary["_o"] = summary["model"].map(order)
    return (summary.sort_values(["dataset", "_o"])
                   .drop(columns="_o")
                   .reset_index(drop=True))


def format_summary_table(summary: pd.DataFrame) -> str:
    """Fixed-width mean +/- std table, ready to paste into the paper."""
    lines = []
    header = f"{'dataset':<10} {'model':<19} " + " ".join(f"{m:^17}" for m in METRICS)
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in summary.iterrows():
        cells = [f"{r[f'{m}_mean']:.4f}+/-{r[f'{m}_std']:.4f}" for m in METRICS]
        lines.append(f"{r['dataset']:<10} {DISPLAY_NAMES[r['model']]:<19} " +
                     " ".join(f"{c:^17}" for c in cells))
    return "\n".join(lines)


def run_e1(datasets: list[str], seeds: list[int] | None = None) -> pd.DataFrame:
    """Full E1: all datasets x all models x all seeds."""
    seeds = seeds or SEEDS
    banner(log, "TASK 6 (E1): PERFORMANCE EVALUATION")

    all_rows = []
    for ds in datasets:
        params = get_params_for_models(ds)
        if params:
            log.info("[%s] using tuned hyperparameters for %s", ds, sorted(params))
        else:
            log.warning("[%s] NO tuned hyperparameters found - falling back to library "
                        "defaults. Run tuning.py for the numbers that go in the paper.", ds)
        cms_total = {m: np.zeros((2, 2), dtype=int) for m in ALL_MODEL_NAMES}
        for seed in seeds:
            rows, cms, _ = train_and_evaluate(ds, seed, params)
            all_rows.extend(rows)
            for m, cm in cms.items():
                cms_total[m] += cm
        for m, cm in cms_total.items():
            out = plot_confusion(ds, m, cm, len(seeds))
            log.info("[%s] confusion matrix -> %s", ds, out.name)

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "e1_performance.csv"
    df.to_csv(out_csv, index=False)
    log.info("per-run results -> %s (%d rows)", out_csv, len(df))

    summary = summarise(df)
    summary.to_csv(RESULTS_DIR / "e1_performance_summary.csv", index=False)
    banner(log, "E1 SUMMARY (mean +/- std across seeds)", "-")
    for line in format_summary_table(summary).splitlines():
        log.info(line)

    worst = df.loc[df["accuracy"].idxmin()]
    if worst["accuracy"] < SANITY_ACCURACY_FLOOR:
        log.warning("Lowest accuracy in E1 is %.4f (%s/%s seed %d) - below the %.2f "
                    "sanity floor.", worst["accuracy"], worst["dataset"], worst["model"],
                    worst["seed"], SANITY_ACCURACY_FLOOR)

    (RESULTS_DIR / "e1_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8")
    return df


# --------------------------------------------------------------------------- #
# Task A4 - E1 re-run across experiment configs
# --------------------------------------------------------------------------- #
def train_and_evaluate_config(config, seed: int, params: dict | None = None,
                              keep_models: bool = False) -> tuple[list[dict], dict, dict]:
    """`train_and_evaluate` addressed by experiment config rather than dataset."""
    cfg = resolve_config(config) if isinstance(config, str) else config
    set_global_seed(seed)
    splits = get_splits_for_config(cfg, seed)
    params = params if params is not None else get_params_for_config(cfg)
    models = build_models(seed, params)

    rows, cms, fitted = [], {}, {}
    for name in ALL_MODEL_NAMES:
        est = models[name]
        t0 = time.perf_counter()
        est.fit(splits.X_train, splits.y_train)
        fit_s = time.perf_counter() - t0

        y_pred = est.predict(splits.X_test)
        y_proba = est.predict_proba(splits.X_test)[:, 1]
        scores = score_predictions(splits.y_test, y_pred, y_proba)

        rows.append({"config": cfg.name, "dataset": cfg.dataset, "model": name, "seed": seed,
                     "dedup": cfg.dedup, "n_features": len(splits.feature_names),
                     "fit_seconds": round(fit_s, 2), "n_train": len(splits.y_train),
                     "n_test": len(splits.y_test), **scores})
        cms[name] = confusion_matrix(splits.y_test, y_pred, labels=[0, 1])
        if keep_models:
            fitted[name] = est

        log.info("[%s seed=%d] %-18s acc=%.4f prec=%.4f rec=%.4f f1=%.4f mcc=%.4f auc=%.4f "
                 "(fit %.1fs)", cfg.name, seed, DISPLAY_NAMES[name], scores["accuracy"],
                 scores["precision"], scores["recall"], scores["f1"], scores["mcc"],
                 scores["roc_auc"], fit_s)
    return rows, cms, fitted


def _check_a4_expectations(df: pd.DataFrame) -> list[str]:
    """The brief's sanity expectations for A4, checked rather than assumed."""
    notes = []
    for cfg_name, g in df.groupby("config"):
        acc = g.loc[g["model"] == ENSEMBLE_NAME, "accuracy"]
        if acc.empty:
            continue
        mean_acc = float(acc.mean())
        if cfg_name == "uci_full":
            if 0.95 <= mean_acc <= 0.97:
                notes.append(f"uci_full ensemble accuracy {mean_acc:.4f} is inside the "
                             f"expected 0.95-0.97 band.")
            else:
                notes.append(f"NOTE: uci_full ensemble accuracy {mean_acc:.4f} is OUTSIDE "
                             f"the expected 0.95-0.97 band.")
        if cfg_name == "phiusiil_leakfree":
            if mean_acc >= 1.0 - 1e-12:
                notes.append(
                    "STOP: phiusiil_leakfree is STILL exactly 1.0000. A1.3 did not identify "
                    "the full leaking set, so the leak-free variant is not leak-free. This "
                    "must be reported rather than worked around.")
            else:
                notes.append(f"phiusiil_leakfree ensemble accuracy {mean_acc:.6f} is below "
                             f"1.0, as required.")
    return notes


def _write_e1_v2(all_rows: list[dict]) -> pd.DataFrame:
    """Persist the A4 table built so far, after every seed."""
    df = pd.DataFrame(all_rows)
    lead = ["config", "dataset", "model", "seed"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    write_csv_atomic(df, RESULTS_DIR / "e1_performance_v2.csv")
    return df


def run_e1_v2(configs: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    """Task A4: E1 across every experiment config, 5 seeds each."""
    seeds = seeds or SEEDS
    configs = configs or available_configs()
    banner(log, "TASK A4 (E1 v2): PERFORMANCE ACROSS CONFIGS")

    all_rows = []
    for name in configs:
        cfg = resolve_config(name)
        log.info("")
        log.info("=== config %s ===", cfg.describe())
        params = get_params_for_config(cfg)
        if params:
            log.info("[%s] tuned hyperparameters for %s", cfg.name, sorted(params))
        else:
            log.warning("[%s] NO tuned hyperparameters found - using library defaults.",
                        cfg.name)
        cms_total = {m: np.zeros((2, 2), dtype=int) for m in ALL_MODEL_NAMES}
        for seed in seeds:
            rows, cms, _ = train_and_evaluate_config(cfg, seed, params)
            all_rows.extend(rows)
            for m, cm in cms.items():
                cms_total[m] += cm
            # Written after every seed rather than once at the end, so an
            # interrupted sweep keeps the cells it already paid for.
            _write_e1_v2(all_rows)
        for m, cm in cms_total.items():
            plot_confusion(cfg.name, m, cm, len(seeds))

    df = _write_e1_v2(all_rows)
    log.info("per-run results -> %s (%d rows)", RESULTS_DIR / "e1_performance_v2.csv", len(df))

    summary = summarise_v2(df)
    summary.to_csv(RESULTS_DIR / "e1_performance_summary_v2.csv", index=False)
    banner(log, "E1 v2 SUMMARY (mean +/- std across seeds)", "-")
    for line in format_summary_table_v2(summary).splitlines():
        log.info(line)

    banner(log, "A4 SANITY CHECKS", "-")
    notes = _check_a4_expectations(df)
    for n in notes:
        (log.warning if n.startswith(("STOP", "NOTE")) else log.info)(n)

    (RESULTS_DIR / "e1_summary_v2.json").write_text(
        json.dumps({"summary": summary.to_dict("records"), "sanity_notes": notes},
                   indent=2, default=float), encoding="utf-8")
    return df


def summarise_v2(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std across seeds for every (config, model)."""
    g = df.groupby(["config", "model"])[METRICS]
    summary = g.agg(["mean", "std"]).round(6)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    order = {name: i for i, name in enumerate(ALL_MODEL_NAMES)}
    summary["_o"] = summary["model"].map(order)
    return (summary.sort_values(["config", "_o"]).drop(columns="_o").reset_index(drop=True))


def format_summary_table_v2(summary: pd.DataFrame) -> str:
    lines = []
    header = f"{'config':<19} {'model':<19} " + " ".join(f"{m:^19}" for m in METRICS)
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in summary.iterrows():
        cells = [f"{r[f'{m}_mean']:.4f}+/-{r[f'{m}_std']:.4f}" for m in METRICS]
        lines.append(f"{r['config']:<19} {DISPLAY_NAMES[r['model']]:<19} " +
                     " ".join(f"{c:^19}" for c in cells))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from config import CONFIG_NAMES

    ap = argparse.ArgumentParser(description="Task 6 / A4: E1 performance evaluation")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--configs", nargs="*", default=None, choices=CONFIG_NAMES,
                    help="run the Task A4 config sweep instead of the Phase A datasets")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.configs is not None:
        run_e1_v2(args.configs or None, seeds=args.seeds)
    else:
        ds = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
        run_e1(ds, seeds=args.seeds)
