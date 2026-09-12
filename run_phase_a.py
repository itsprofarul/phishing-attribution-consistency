"""Task 10 - Phase A orchestration.

Runs Tasks 1 -> 9 in order and prints a single summary block at the end covering
everything a reader needs to judge the run: class balance after harmonisation,
leakage warnings, E1 performance, the E2 baseline, wall-clock time, and every
warning or assertion failure raised along the way.

Usage
    python run_phase_a.py                          # everything, both datasets
    python run_phase_a.py --skip-tuning            # reuse cached best params
    python run_phase_a.py --dataset uci            # one dataset only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    RESULTS_DIR,
    SEEDS,
    SHAP_CONVERGENCE_TAU_THRESHOLD,
    TAU_STABLE_THRESHOLD,
    TAU_UNSTABLE_THRESHOLD,
    banner,
    get_logger,
    library_versions,
    set_global_seed,
)
from src.data_loader import load_all  # noqa: E402
from src.evaluate import format_summary_table, run_e1, summarise  # noqa: E402
from src.experiment_e2 import interpret, run_e2  # noqa: E402
from src.experiment_e2 import aggregate as e2_aggregate  # noqa: E402
from src.leakage_check import run_leakage_check  # noqa: E402
from src.models import ENSEMBLE_NAME  # noqa: E402
from src.preprocessing import prepare_dataset  # noqa: E402
from src.shap_utils import make_background, run_convergence_checks  # noqa: E402
from src.tuning import get_params_for_models, tune_dataset  # noqa: E402

log = get_logger("run_phase_a")


class WarningCollector(logging.Handler):
    """Captures every WARNING+ record so the final block can replay them."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"{record.levelname:<8} {record.name}: {record.getMessage()}")


def _section(title: str) -> None:
    banner(log, title)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase A pipeline")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--skip-tuning", action="store_true",
                    help="reuse cached best params instead of re-running the searches")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--skip-convergence", action="store_true",
                    help="skip the SHAP sample-size convergence check (Task 7)")
    args = ap.parse_args()

    datasets = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
    seeds = args.seeds or SEEDS

    collector = WarningCollector()
    logging.getLogger("phase_a").addHandler(collector)

    set_global_seed(seeds[0])
    t_start = time.perf_counter()
    failures: list[str] = []
    summary_state: dict = {"datasets": datasets, "seeds": seeds}

    _section("PHASE A - SHAP CROSS-DATASET CONSISTENCY STUDY")
    log.info("datasets=%s seeds=%s skip_tuning=%s", datasets, seeds, args.skip_tuning)
    for k, v in library_versions().items():
        log.info("    %-12s %s", k, v)

    # ---- Tasks 1-2 -------------------------------------------------------- #
    _section("TASKS 1-2: DATA LOADING AND PREPROCESSING")
    load_all(datasets=datasets)
    balance = {}
    for ds in datasets:
        df = prepare_dataset(ds)
        report = json.loads((RESULTS_DIR / f"preprocessing_{ds}.json").read_text(encoding="utf-8"))
        balance[ds] = {
            "shape": list(df.shape),
            "n_features": int(df.shape[1] - 1),
            "phishing_rate": report["phishing_rate"],
            "phishing_rate_before_dedup": report["phishing_rate_before_dedup"],
            "published_phishing_rate": report["published_phishing_rate"],
            "duplicates_removed": report["duplicates"]["exact_duplicates_removed"],
            "rows_before": report["duplicates"]["rows_before"],
            "dropped_columns": report["dropped_columns"],
        }
    summary_state["class_balance"] = balance

    # ---- Task 3 ----------------------------------------------------------- #
    _section("TASK 3: LEAKAGE CHECK")
    try:
        leakage = run_leakage_check(datasets, seed=seeds[0])
        summary_state["leakage"] = leakage
    except Exception as exc:
        failures.append(f"leakage_check: {exc}")
        log.error("leakage check failed:\n%s", traceback.format_exc())
        summary_state["leakage"] = {"error": str(exc)}

    # ---- Task 5 ----------------------------------------------------------- #
    _section("TASK 5: HYPERPARAMETER TUNING")
    for ds in datasets:
        cached = get_params_for_models(ds)
        if args.skip_tuning:
            if cached:
                log.info("[%s] --skip-tuning: reusing cached best params %s", ds, cached)
            else:
                log.warning("[%s] --skip-tuning was requested but NO cached parameters "
                            "exist; library defaults will be used.", ds)
        else:
            try:
                tune_dataset(ds)
            except Exception as exc:
                failures.append(f"tuning[{ds}]: {exc}")
                log.error("tuning failed for %s:\n%s", ds, traceback.format_exc())
    summary_state["best_params"] = {ds: get_params_for_models(ds) for ds in datasets}

    # ---- Task 6 (E1) ------------------------------------------------------ #
    _section("TASK 6 (E1): PERFORMANCE EVALUATION")
    e1 = None
    try:
        e1 = run_e1(datasets, seeds=seeds)
        summary_state["e1_summary"] = summarise(e1).to_dict("records")
    except Exception as exc:
        failures.append(f"e1: {exc}")
        log.error("E1 failed:\n%s", traceback.format_exc())

    # ---- Task 7 (SHAP convergence) ---------------------------------------- #
    _section("TASK 7: SHAP CONVERGENCE CHECK")
    convergence = None
    if args.skip_convergence:
        log.info("--skip-convergence: skipped")
    else:
        try:
            from src.evaluate import train_and_evaluate
            from src.preprocessing import get_splits

            payloads = {}
            for ds in datasets:
                splits = get_splits(ds, seeds[0])
                _, _, fitted = train_and_evaluate(ds, seeds[0], get_params_for_models(ds),
                                                  keep_models=True)
                payloads[ds] = {
                    "model": fitted[ENSEMBLE_NAME],
                    "X_pool": splits.X_test,
                    "y_pool": splits.y_test,
                    "feature_names": splits.feature_names,
                    "seed": seeds[0],
                    "background": make_background(splits.X_train, splits.y_train, seeds[0]),
                    "model_name": ENSEMBLE_NAME,
                }
            convergence = run_convergence_checks(payloads)
            summary_state["convergence"] = convergence.to_dict("records")
        except Exception as exc:
            failures.append(f"convergence: {exc}")
            log.error("convergence check failed:\n%s", traceback.format_exc())

    # ---- Task 9 (E2) ------------------------------------------------------ #
    _section("TASK 9 (E2): WITHIN-DATASET CONSISTENCY BASELINE")
    e2 = None
    try:
        e2 = run_e2(datasets, seeds=seeds)
        summary_state["e2_summary"] = e2_aggregate(e2).to_dict("records")
    except Exception as exc:
        failures.append(f"e2: {exc}")
        log.error("E2 failed:\n%s", traceback.format_exc())

    # ---- Final summary ---------------------------------------------------- #
    elapsed = time.perf_counter() - t_start
    _section("PHASE A FINAL SUMMARY")

    log.info("")
    log.info("1. DATASET SHAPES AND CLASS BALANCE AFTER HARMONISATION (1 = phishing)")
    for ds, b in balance.items():
        log.info("   %-9s %d rows x %d features", ds, b["shape"][0], b["n_features"])
        log.info("             phishing rate %.4f  (published %.4f, pre-dedup %.4f -> "
                 "ASSERTION PASSED)", b["phishing_rate"], b["published_phishing_rate"],
                 b["phishing_rate_before_dedup"])
        log.info("             %d of %d rows removed as exact duplicates",
                 b["duplicates_removed"], b["rows_before"])
        if b["dropped_columns"]:
            log.info("             dropped identifier columns: %s", b["dropped_columns"])

    log.info("")
    log.info("2. LEAKAGE WARNINGS")
    lk = summary_state.get("leakage", {})
    if lk.get("n_flagged"):
        log.info("   %d feature(s) exceed %.2f single-feature test accuracy:",
                 lk["n_flagged"], lk.get("threshold"))
        for f in lk["flagged_features"]:
            log.info("     %-9s %-28s acc=%.4f f1=%.4f mi=%.4f", f["dataset"], f["feature"],
                     f["accuracy"], f["f1"], f["mutual_information"])
        log.info("   Nothing was dropped; see results/single_feature_predictive_power.csv.")
    elif "error" in lk:
        log.info("   leakage check did not complete: %s", lk["error"])
    else:
        log.info("   none")

    log.info("")
    log.info("3. E1 PERFORMANCE (mean +/- std across %d seeds)", len(seeds))
    if e1 is not None:
        for line in format_summary_table(summarise(e1)).splitlines():
            log.info("   %s", line)
    else:
        log.info("   E1 did not complete")

    log.info("")
    log.info("4. E2 WITHIN-DATASET BASELINE")
    if e2 is not None:
        summ = e2_aggregate(e2)
        for _, r in summ.iterrows():
            log.info("   %-9s tau=%.4f+/-%.4f  J@5=%.3f  J@10=%.3f  J@15=%.3f  sign=%.1f%%",
                     r["dataset"], r["kendall_tau_mean"], r["kendall_tau_std"],
                     r["jaccard_5_mean"], r["jaccard_10_mean"], r["jaccard_15_mean"],
                     r["sign_agreement_mean"])
        log.info("")
        for line in interpret(summ):
            log.info("   %s", line)
    else:
        log.info("   E2 did not complete")

    log.info("")
    log.info("5. SHAP CONVERGENCE")
    if convergence is not None and len(convergence):
        for _, r in convergence.iterrows():
            flag = " [DEGENERATE]" if r.get("degenerate") else ""
            log.info("   %-9s tau(%d vs %d) = %.4f%s", r["dataset"], r["size_a"],
                     r["size_b"], r["kendall_tau"], flag)
        decisive = convergence[(convergence["size_a"] == 10000) &
                               (convergence["size_b"] == 20000)]
        for _, r in decisive.iterrows():
            verdict = ("CONVERGED" if r["kendall_tau"] >= SHAP_CONVERGENCE_TAU_THRESHOLD
                       else "NOT CONVERGED - sample too small")
            if r.get("degenerate"):
                verdict = "NOT EVALUABLE - pool smaller than 20k"
            log.info("   %-9s 10k vs 20k: %s", r["dataset"], verdict)
    else:
        log.info("   not run")

    log.info("")
    log.info("6. WALL-CLOCK TIME: %.1f s (%.2f h)", elapsed, elapsed / 3600)

    log.info("")
    log.info("7. WARNINGS AND ASSERTION FAILURES")
    if failures:
        log.info("   FAILURES (%d):", len(failures))
        for f in failures:
            log.info("     - %s", f)
    else:
        log.info("   no stage raised an exception")
    unique_warnings = list(dict.fromkeys(collector.records))
    if unique_warnings:
        log.info("   warnings raised (%d unique):", len(unique_warnings))
        for w in unique_warnings:
            log.info("     - %s", w)
    else:
        log.info("   no warnings raised")

    summary_state["wall_clock_seconds"] = round(elapsed, 2)
    summary_state["failures"] = failures
    summary_state["warnings"] = unique_warnings
    summary_state["library_versions"] = library_versions()
    (RESULTS_DIR / "phase_a_summary.json").write_text(
        json.dumps(summary_state, indent=2, default=str), encoding="utf-8")
    log.info("")
    log.info("machine-readable summary -> %s", RESULTS_DIR / "phase_a_summary.json")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
