"""Phase A Addendum orchestration - Tasks A1 to A6.

Runs the addendum in the order the brief specifies, because A1 determines the
inputs to A4: the leak-free feature set does not exist until the progressive
removal experiment has run, and A4/A5 must not run for the leak-free config if
that experiment failed to de-leak the dataset.

Usage
    python run_addendum.py                     # A1 -> A6, everything
    python run_addendum.py --tasks A2 A3       # a subset, in the given order
    python run_addendum.py --skip-tuning       # reuse cached best params for A4
    python run_addendum.py --configs uci_full  # restrict the A4/A5 sweep

The Phase A pipeline (run_phase_a.py) is left untouched; this is a separate
entry point so the original results stay reproducible.
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
    CONFIG_NAMES,
    LEAK_DIR,
    RESULTS_DIR,
    SEEDS,
    ConfigUnavailableError,
    available_configs,
    banner,
    get_logger,
    library_versions,
)

log = get_logger("run_addendum")

ALL_TASKS = ["A1", "A2", "A3", "A4", "A5", "A6"]


class WarningCollector(logging.Handler):
    """Captures every WARNING+ record so the final block can replay them."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"{record.levelname:<8} {record.name}: {record.getMessage()}")


def _reload_config():
    """Re-import config after A1.4 rewrites PHIUSIIL_LEAKFREE_FEATURES into it.

    A1 writes the leak-free feature set into config.py as source. Within a single
    process the module is already imported, so the new value is only visible after
    an explicit reload - without this, A4 in the same run would still see None.
    """
    import importlib

    import config as _config
    importlib.reload(_config)
    return _config


def task_a1(args) -> dict:
    banner(log, "TASK A1 - CHARACTERISE THE PHIUSIIL LEAK")
    from src.leak_analysis import run_leak_analysis

    out = run_leak_analysis("phiusiil", seed=SEEDS[0],
                            skip_progressive=args.skip_progressive)
    _reload_config()
    return {"outcome": out.get("outcome", {}).get("outcome", "skipped")}


def task_a2(args) -> dict:
    banner(log, "TASK A2 - SIGN-AGREEMENT METRIC")
    import subprocess

    # The corrected metric's guarantees are asserted by the test suite, so the
    # orchestrator runs the tests rather than re-deriving them here.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_consistency.py", "-q"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        log.error("consistency tests FAILED:\n%s", proc.stdout[-3000:])
        raise RuntimeError("Task A2 consistency tests failed")
    log.info("consistency tests pass: %s", tail)
    return {"pytest": tail}


def task_a3(args) -> dict:
    banner(log, "TASK A3 - DE-DUPLICATION POLICY")
    from config import DEDUPLICATE, EXPECTED_PHISHING_RATE, EXPECTED_RATE_TOLERANCE
    from src.preprocessing import LABEL, prepare_dataset

    out = {}
    for ds in ("phiusiil", "uci"):
        for dedup in (True, False):
            df = prepare_dataset(ds, dedup=dedup)
            rate = float(df[LABEL].mean())
            out[f"{ds}_{'dedup' if dedup else 'full'}"] = {
                "rows": int(len(df)), "phishing_rate": rate,
                "is_primary": dedup == DEDUPLICATE[ds],
            }
    # The brief's explicit check: UCI's primary analysis must now carry the
    # published balance without any adjustment.
    prim = out["uci_full"]
    delta = abs(prim["phishing_rate"] - EXPECTED_PHISHING_RATE["uci"])
    if delta > EXPECTED_RATE_TOLERANCE:
        raise AssertionError(
            f"uci_full phishing rate {prim['phishing_rate']:.4f} does not match the "
            f"published {EXPECTED_PHISHING_RATE['uci']:.4f}")
    log.info("CONFIRMED: uci primary analysis = %d rows, phishing rate %.4f vs published "
             "%.4f (|delta| = %.4f) - the class-balance assertion passes unadjusted.",
             prim["rows"], prim["phishing_rate"], EXPECTED_PHISHING_RATE["uci"], delta)
    return out


def _selected_configs(args) -> list[str]:
    wanted = args.configs or CONFIG_NAMES
    usable = available_configs()
    out, skipped = [], []
    for c in wanted:
        (out if c in usable else skipped).append(c)
    for c in skipped:
        log.warning("config %r is unavailable (its feature set has not been determined by "
                    "A1) and will be skipped.", c)
    return out


def task_a4(args) -> dict:
    banner(log, "TASK A4 - RE-RUN E1 ACROSS CONFIGS")
    _reload_config()
    from src.evaluate import run_e1_v2
    from src.tuning import tune_config

    configs = _selected_configs(args)
    if not args.skip_tuning:
        for c in configs:
            tune_config(c)
    else:
        log.warning("--skip-tuning: reusing whatever tuned parameters are cached. For "
                    "phiusiil_leakfree this means parameters tuned on a DIFFERENT feature "
                    "space unless a leak-free search has already been cached.")
    df = run_e1_v2(configs, seeds=args.seeds)
    return {"rows": int(len(df)), "configs": configs}


def task_a5(args) -> dict:
    banner(log, "TASK A5 - RE-RUN E2 ACROSS CONFIGS")
    _reload_config()
    from src.experiment_e2 import run_e2_v2

    configs = _selected_configs(args)
    df = run_e2_v2(configs, seeds=args.seeds)
    return {"rows": int(len(df)), "configs": configs}


def task_a6(args) -> dict:
    banner(log, "TASK A6 - FEATURE INVENTORY RECONCILIATION")
    from src.feature_inventory import run_inventory

    inv = run_inventory()
    return {"rows": int(len(inv))}


TASK_FNS = {"A1": task_a1, "A2": task_a2, "A3": task_a3,
            "A4": task_a4, "A5": task_a5, "A6": task_a6}


def final_summary(results: dict, warnings: list[str], elapsed: float) -> None:
    banner(log, "ADDENDUM SUMMARY")

    leak_outcome = LEAK_DIR / "progressive_removal_outcome.json"
    if leak_outcome.exists():
        m = json.loads(leak_outcome.read_text(encoding="utf-8"))
        log.info("A1.3 outcome        : %s", m["outcome"])
        log.info("     removed        : %d %s", m["n_removed"], m["removed_features"])
        log.info("     final accuracy : %.6f on %d features",
                 m["final_accuracy"], m["n_features_remaining"])

    e1 = RESULTS_DIR / "e1_performance_v2.csv"
    if e1.exists():
        df = pd.read_csv(e1)
        ens = df[df["model"] == "ensemble"]
        log.info("")
        log.info("A4 E1 (ensemble, mean +/- std over seeds):")
        for cfg_name, g in ens.groupby("config"):
            log.info("     %-19s acc=%.4f+/-%.4f  f1=%.4f+/-%.4f  mcc=%.4f+/-%.4f",
                     cfg_name, g["accuracy"].mean(), g["accuracy"].std(),
                     g["f1"].mean(), g["f1"].std(), g["mcc"].mean(), g["mcc"].std())

    e2 = RESULTS_DIR / "e2_within_dataset_baseline_v2.csv"
    if e2.exists():
        df = pd.read_csv(e2)
        log.info("")
        log.info("A5 E2 (within-dataset consistency):")
        for cfg_name, g in df.groupby("config"):
            log.info("     %-19s tau=%.4f+/-%.4f  J@10=%.3f  sign10=%.1f%%  "
                     "unfiltered=%.1f%%  top1_agree=%d/%d",
                     cfg_name, g["kendall_tau"].mean(), g["kendall_tau"].std(),
                     g["jaccard_10"].mean(), g["sign_agreement_top10"].mean(),
                     g["sign_agreement_unfiltered"].mean(),
                     int(g["top1_agrees"].sum()), len(g))

    inv = RESULTS_DIR / "feature_inventory_reconciliation.json"
    if inv.exists():
        log.info("")
        log.info("A6 feature inventory:")
        for r in json.loads(inv.read_text(encoding="utf-8")):
            log.info("     %-10s delivered=%d dropped=%d retained=%d not_delivered=%d",
                     r["dataset"], r["features_delivered"], r["columns_dropped"],
                     r["features_retained"], r["columns_not_delivered"])

    log.info("")
    log.info("Tasks completed     : %s", ", ".join(results))
    log.info("Total wall clock    : %.1f min", elapsed / 60)
    log.info("Warnings raised     : %d", len(warnings))
    for w in warnings:
        log.info("    %s", w)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase A addendum (Tasks A1-A6)")
    ap.add_argument("--tasks", nargs="*", default=ALL_TASKS, choices=ALL_TASKS)
    ap.add_argument("--configs", nargs="*", default=None, choices=CONFIG_NAMES)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--skip-progressive", action="store_true",
                    help="skip the expensive A1.3 greedy loop (no leak-free set defined)")
    args = ap.parse_args()

    collector = WarningCollector()
    logging.getLogger("phase_a").addHandler(collector)

    banner(log, "PHASE A ADDENDUM")
    log.info("tasks: %s", ", ".join(args.tasks))
    for k, v in library_versions().items():
        log.info("  %-12s %s", k, v)

    t0 = time.perf_counter()
    results: dict = {}
    status = 0
    for task in ALL_TASKS:
        if task not in args.tasks:
            continue
        try:
            results[task] = TASK_FNS[task](args)
        except ConfigUnavailableError as exc:
            log.error("%s stopped: %s", task, exc)
            results[task] = {"error": str(exc)}
            status = 1
            break
        except Exception:
            log.error("%s FAILED:\n%s", task, traceback.format_exc())
            results[task] = {"error": traceback.format_exc(limit=3)}
            status = 1
            break

    elapsed = time.perf_counter() - t0
    final_summary(results, collector.records, elapsed)
    (RESULTS_DIR / "addendum_summary.json").write_text(
        json.dumps({"results": results, "warnings": collector.records,
                    "elapsed_seconds": round(elapsed, 1),
                    "versions": library_versions()}, indent=2, default=str),
        encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
