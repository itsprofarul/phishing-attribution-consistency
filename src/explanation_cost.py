"""Task B5b - explanation cost as a leakage diagnostic, with contention control.

During A1.3 the ensemble fit time stayed flat while TreeSHAP cost grew ~7x, then
plateaued. The mechanism is that interventional TreeSHAP walks tree nodes: while
a leaking feature separates the classes in one split the forest stays shallow, and
once the leak features are stripped the boundary becomes genuinely hard, the trees
deepen, and explanation cost rises with them. That makes the cost of explaining a
model a free leakage signal.

BUT: this only holds as a published measurement if the timings are clean. Several
A1.3 iterations ran while other experiments competed for the same cores, and a
contended wall-clock reading is not a measurement of explanation cost. So every
iteration is annotated with whether other jobs were running, reconstructed from
two independent sources:

  * results/phase_a.log - every module logs there with a timestamp and a logger
    name, so a window containing `phase_a.experiment_e2`, `phase_a.transfer`,
    `phase_a.benchmark_screen`, `phase_a.evaluate` or `phase_a.tuning` records
    demonstrably had another experiment running. leak_analysis never emits under
    those logger names, so their presence is positive evidence, not inference.
  * results/leak_analysis/load_samples.csv - direct CPU sampling, available from
    the point the sampler was started onward.

Iterations flagged `contended` must be excluded from, or annotated in, the cost
analysis. `cost_analysis_eligible` encodes that decision in the table itself so a
downstream plot cannot silently mix clean and contended timings.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LEAK_DIR, LOG_PATH, RESULTS_DIR, banner, get_logger, write_csv_atomic  # noqa: E402

log = get_logger("explanation_cost")

COST_COLUMNS = ["dataset", "iteration", "n_features", "accuracy", "fit_seconds",
                "shap_seconds", "mean_tree_depth", "total_nodes"]

# Loggers that only ever appear when a DIFFERENT experiment is running.
COMPETING_LOGGERS = {
    "phase_a.experiment_e2": "E2",
    "phase_a.evaluate": "E1",
    "phase_a.tuning": "TUNE",
    "phase_a.transfer": "B1",
    "phase_a.benchmark_screen": "B2",
    "phase_a.external_datasets": "B2acq",
    "phase_a.feature_inventory": "A6",
    "phase_a.run_addendum": "A4A5",
}

_LOG_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}) \| \w+\s+\| ([\w.]+) \|")
_ITER_RE = re.compile(r"\[iter\s+(\d+)\] n_features=(\d+) acc=([\d.]+) .*?"
                      r"fit ([\d.]+)s shap ([\d.]+)s")


def _parse_log_events(path: Path) -> list[tuple[str, str, int]]:
    """(HH:MM:SS, logger, day_index) for every line, in file order.

    phase_a.log records only a clock time, no date, and it accumulates across
    days. Matching a window purely on HH:MM:SS therefore lets a job that ran at
    11:30 TODAY be scored as contending with an iteration that ran at 11:30
    YESTERDAY - which silently reclassified seven clean iterations as contended.

    Day boundaries are recovered from the only signal available: the clock going
    backwards means a new day started.
    """
    out = []
    if not path.exists():
        return out
    day, prev = 0, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LOG_RE.match(line)
        if not m:
            continue
        ts = m.group(1)
        if prev is not None and _t(ts) < _t(prev):
            day += 1
        prev = ts
        out.append((ts, m.group(2), day))
    return out


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%H:%M:%S")


def competing_jobs_in_window(events, start: str, end: str,
                             day: int | None = None) -> list[str]:
    """Distinct competing experiments with log activity inside [start, end].

    `day` restricts matching to one day-block; without it a later day's activity
    at the same clock time would be counted as contention.
    """
    t0, t1 = _t(start), _t(end)
    if t1 < t0:                      # window crosses midnight; ignore such windows
        return []
    found = set()
    for ts, logger, d in events:
        if day is not None and d != day:
            continue
        if t0 <= _t(ts) <= t1 and logger in COMPETING_LOGGERS:
            found.add(COMPETING_LOGGERS[logger])
    return sorted(found)


def _a13_day(events, iter_times: list[str]) -> int | None:
    """Which day-block the A1.3 run itself lives in.

    Identified by the day containing leak_analysis activity closest to the
    iteration timestamps, so contention is only ever scored against jobs that
    genuinely ran alongside it.
    """
    if not iter_times:
        return None
    want = set(iter_times)
    for ts, logger, d in events:
        if logger == "phase_a.leak_analysis" and ts in want:
            return d
    return None


def load_samples_in_window(start: str, end: str, date: str | None = None) -> dict:
    """Mean CPU and concurrent-job tags from the sampler, if it covers the window.

    `date` (YYYY-MM-DD) is REQUIRED to be meaningful: the sampler stores full ISO
    timestamps but the iteration windows carry only a clock time, so matching on
    HH:MM:SS alone let samples from a LATER DAY be attributed to an earlier run.
    That spuriously flagged seven clean iterations as contended by a job that had
    not yet started when they ran. Without a date, sampler evidence is declined
    rather than guessed.
    """
    path = LEAK_DIR / "load_samples.csv"
    if not path.exists() or date is None:
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    ts = pd.to_datetime(df["timestamp"])
    df = df[ts.dt.strftime("%Y-%m-%d") == date]
    if df.empty:
        return {}
    df = df.assign(t=pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S"))
    t0, t1 = _t(start), _t(end)
    sel = df[df["t"].map(lambda s: t0 <= _t(s) <= t1)]
    if sel.empty:
        return {}
    tags = set()
    for s in sel["job_tags"].dropna():
        tags |= {x for x in str(s).split(";") if x not in ("", "other", "worker", "A1.3")}
    return {"sampled_cpu_percent_mean": round(float(sel["cpu_percent"].mean()), 2),
            "sampled_cpu_percent_max": round(float(sel["cpu_percent"].max()), 2),
            "sampled_n_python_procs_max": int(sel["n_python_procs"].max()),
            "sampled_job_tags": ";".join(sorted(tags)),
            "n_load_samples": int(len(sel))}


def parse_a13_log(log_file: Path, dataset: str = "phiusiil") -> pd.DataFrame:
    """Recover the A1.3 curve from a run log, with each iteration's time window.

    The brief asks for these to be recovered from the log rather than re-run, so
    they are parsed rather than recomputed. Each iteration's window runs from the
    previous iteration's completion to its own.
    """
    if not log_file.exists():
        return pd.DataFrame()
    text = log_file.read_text(encoding="utf-8", errors="replace")
    rows, prev_ts = [], None
    for line in text.splitlines():
        m = _ITER_RE.search(line)
        if not m:
            continue
        tm = _LOG_RE.match(line)
        ts = tm.group(1) if tm else None
        it, nfeat, acc, fit_s, shap_s = m.groups()
        rows.append({"dataset": dataset, "iteration": int(it),
                     "n_features": int(nfeat), "accuracy": float(acc),
                     "fit_seconds": float(fit_s), "shap_seconds": float(shap_s),
                     "window_start": prev_ts, "window_end": ts})
        prev_ts = ts
    return pd.DataFrame(rows)


def build_cost_profile(a13_logs: list[Path], include_screening: bool = True) -> pd.DataFrame:
    """Assemble results/leak_analysis/explanation_cost_profile.csv."""
    banner(log, "B5b EXPLANATION COST PROFILE")
    events = _parse_log_events(LOG_PATH)

    frames = []
    for lf in a13_logs:
        df = parse_a13_log(lf)
        if df.empty:
            log.warning("no iterations parsed from %s", lf)
            continue
        # A resumed run replays completed iterations from cache; shap_seconds of
        # ~0 is a cache hit, not a measurement, and must never enter the curve.
        df["cache_replay"] = df["shap_seconds"] < 1.0
        df["source_log"] = lf.name
        # The run's calendar date, taken from the log file itself - the iteration
        # rows carry only a clock time.
        df["run_date"] = datetime.fromtimestamp(lf.stat().st_mtime).strftime("%Y-%m-%d")
        frames.append(df)
    if not frames:
        raise RuntimeError("no A1.3 iterations could be recovered from any log")

    prof = pd.concat(frames, ignore_index=True)

    # Prefer a real measurement over a cache replay for the same iteration.
    prof = (prof.sort_values(["iteration", "cache_replay"])
                .drop_duplicates(subset=["dataset", "iteration"], keep="first")
                .sort_values("iteration").reset_index(drop=True))

    # Iteration 0 has no predecessor, so its window start is reconstructed from
    # its own measured duration rather than left unassessable - it is the
    # "flat and fast while leaking" anchor of the whole curve and cannot be the
    # one point with no contention check.
    first = prof.index[prof["iteration"] == prof["iteration"].min()]
    for i in first:
        if not isinstance(prof.at[i, "window_start"], str) and isinstance(
                prof.at[i, "window_end"], str):
            dur = float(prof.at[i, "fit_seconds"]) + float(prof.at[i, "shap_seconds"])
            start = _t(prof.at[i, "window_end"]) - timedelta(seconds=dur + 30)
            prof.at[i, "window_start"] = start.strftime("%H:%M:%S")
            prof.at[i, "window_reconstructed"] = True

    # Restrict contention scoring to the day-block A1.3 actually ran in.
    a13_day = _a13_day(events, [s for s in prof["window_end"] if isinstance(s, str)])
    if a13_day is not None:
        log.info("scoring contention against day-block %d of phase_a.log only", a13_day)

    annotations = []
    for _, r in prof.iterrows():
        ann = {"competing_jobs": "", "contention_evidence": "none"}
        # NaN is truthy, so a missing window must be tested explicitly: iteration 0
        # has no predecessor and therefore no start time.
        has_window = (isinstance(r["window_start"], str)
                      and isinstance(r["window_end"], str))
        if not has_window:
            ann["contention_evidence"] = "no_window"
        if has_window:
            jobs = competing_jobs_in_window(events, r["window_start"], r["window_end"],
                                            day=a13_day)
            ann["competing_jobs"] = ";".join(jobs)
            if jobs:
                ann["contention_evidence"] = "phase_a.log"
            ann.update(load_samples_in_window(r["window_start"], r["window_end"],
                                              date=r.get("run_date")))
            if ann.get("sampled_job_tags"):
                ann["contention_evidence"] = ("phase_a.log+sampler"
                                              if jobs else "sampler")
                merged = sorted(set(filter(None, jobs + ann["sampled_job_tags"].split(";"))))
                ann["competing_jobs"] = ";".join(merged)
        annotations.append(ann)

    prof = pd.concat([prof, pd.DataFrame(annotations)], axis=1)
    prof["contended"] = prof["competing_jobs"].fillna("").str.len() > 0
    _MERGE_MARKER = True
    # A cache replay is not a timing at all; a contended run is a timing of the
    # wrong thing. Both are ineligible for the published cost curve.
    prof["cost_analysis_eligible"] = ~(prof["contended"].astype(bool)
                                      | prof["cache_replay"].astype(bool))

    # Merge the clean re-measurement. It is AUTHORITATIVE where present: it was
    # produced with no other experiment running (the chain runs it last and alone),
    # so its timings supersede any log-derived judgement, and it is the only source
    # of tree structure, which A1.3 never logged.
    for c in ("mean_tree_depth", "total_nodes"):
        if c not in prof.columns:
            prof[c] = pd.NA
    remeas_path = LEAK_DIR / "explanation_cost_remeasured.csv"
    if remeas_path.exists():
        rm = pd.read_csv(remeas_path).set_index("iteration")
        prof["remeasured_clean"] = False
        for i, r in prof.iterrows():
            it = int(r["iteration"])
            if it not in rm.index:
                continue
            row = rm.loc[it]
            prof.at[i, "mean_tree_depth"] = row["mean_tree_depth"]
            prof.at[i, "total_nodes"] = row["total_nodes"]
            if bool(row.get("shap_remeasured", False)) and pd.notna(row["shap_seconds"]):
                # Clean timing replaces the contended one outright.
                prof.at[i, "shap_seconds_original_contended"] = r["shap_seconds"]
                prof.at[i, "shap_seconds"] = float(row["shap_seconds"])
                prof.at[i, "fit_seconds"] = float(row["fit_seconds"])
                prof.at[i, "contended"] = False
                prof.at[i, "competing_jobs"] = ""
                prof.at[i, "contention_evidence"] = "remeasured_uncontended"
                prof.at[i, "remeasured_clean"] = True
        n_rm = int(prof["remeasured_clean"].sum())
        log.info("merged %d clean re-measured timing(s) and tree structure for %d "
                 "iteration(s) from %s", n_rm, len(rm), remeas_path.name)
        # Eligibility was computed before this merge, so it must be recomputed:
        # the re-measured iterations are now clean and belong back in the curve.
        prof["cost_analysis_eligible"] = ~(prof["contended"].astype(bool)
                                          | prof["cache_replay"].astype(bool))

    if include_screening:
        scr = RESULTS_DIR / "screening" / "screening_progressive_removal.csv"
        if scr.exists():
            s = pd.read_csv(scr)
            s["cache_replay"] = False
            s["contended"] = pd.NA
            s["competing_jobs"] = ""
            s["contention_evidence"] = "not_assessed"
            s["cost_analysis_eligible"] = pd.NA
            s["source_log"] = "screening"
            prof = pd.concat([prof, s], ignore_index=True)

    ordered = COST_COLUMNS + [c for c in prof.columns if c not in COST_COLUMNS]
    prof = prof[ordered]
    write_csv_atomic(prof, LEAK_DIR / "explanation_cost_profile.csv")
    desc = describe_cost_curve(prof)
    (LEAK_DIR / "explanation_cost_description.json").write_text(
        json.dumps(desc, indent=2, default=float), encoding="utf-8")
    _report(prof)
    log.info("")
    log.info("CURVE DESCRIPTION (generated from the eligible data):")
    log.info("  %s", desc.get("caption", ""))
    return prof


def describe_cost_curve(prof: pd.DataFrame) -> dict:
    """Generate the B5b curve description FROM THE DATA, not from memory.

    The eligible timings are not monotonic - they rise overall but fall back
    several times along the way - so the curve must be described as an overall
    increase with local variation followed by a plateau, never as "steady" or
    "monotonic" growth. A reviewer will check a figure caption against the
    numbers, so the caption is derived here and the monotonicity claim is tested
    rather than asserted.
    """
    a13 = prof[(prof["source_log"] != "screening")
               & (prof["cost_analysis_eligible"].fillna(False))].sort_values("iteration")
    if len(a13) < 3:
        return {"n_eligible": int(len(a13)), "caption": "insufficient clean iterations"}

    s = a13["shap_seconds"].to_numpy(dtype=float)
    diffs = np.diff(s)
    n_down = int((diffs < 0).sum())
    monotonic = n_down == 0
    fold = float(s.max() / s.min()) if s.min() > 0 else float("nan")

    # Plateau test on the last three eligible points: a small relative spread
    # means the curve has flattened rather than still climbing.
    tail = s[-3:]
    tail_spread = float((tail.max() - tail.min()) / tail.max()) if tail.max() else 0.0
    plateau = tail_spread < 0.10

    fit = a13["fit_seconds"].to_numpy(dtype=float)
    caption = (
        f"TreeSHAP cost rises ~{fold:.1f}x across the clean iterations "
        f"({s.min():.0f}s to {s.max():.0f}s) with local variation - it falls back on "
        f"{n_down} of {len(diffs)} steps rather than increasing monotonically - "
        f"before flattening" if plateau else
        f"TreeSHAP cost rises ~{fold:.1f}x across the clean iterations "
        f"({s.min():.0f}s to {s.max():.0f}s) with local variation, falling back on "
        f"{n_down} of {len(diffs)} steps")
    caption += (f". Ensemble fit time stays flat throughout "
                f"({fit.min():.0f}-{fit.max():.0f}s), so the growth is specific to "
                f"explanation, not to training.")

    # The full-range fold-increase spans iterations the contention audit excluded,
    # so it cannot be quoted as a clean measurement. It is still reportable as a
    # BOUND, because contention can only inflate a wall-clock reading: an early
    # contended timing is an upper bound on its true clean value, so the true
    # full-range increase is at least as large as the contended data suggests.
    everything = prof[(prof["source_log"] != "screening")
                      & (~prof["cache_replay"].fillna(False))].sort_values("iteration")
    full = everything["shap_seconds"].to_numpy(dtype=float)
    full_fold = float(full.max() / full.min()) if len(full) and full.min() > 0 else float("nan")
    n_excluded = int(len(everything) - len(a13))

    out = {
        "n_eligible": int(len(a13)),
        "n_excluded_contended": n_excluded,
        "eligible_iterations": [int(i) for i in a13["iteration"]],
        "shap_seconds_min": float(s.min()), "shap_seconds_max": float(s.max()),
        "fold_increase_clean_only": fold,
        "fold_increase_full_range_LOWER_BOUND": full_fold,
        "full_range_note": (
            "The full-range figure spans contended iterations and is therefore a LOWER "
            "BOUND, not a measurement: contention inflates wall-clock, so the excluded "
            "early timings are upper bounds on their clean values and the true increase "
            "is at least this large. Quote it as '>=Nx' or re-measure those iterations "
            "without competing load."),
        "monotonic": monotonic,
        "n_decreasing_steps": n_down,
        "n_steps": int(len(diffs)),
        "plateau_detected": plateau,
        "plateau_tail_relative_spread": round(tail_spread, 4),
        "fit_seconds_min": float(fit.min()), "fit_seconds_max": float(fit.max()),
        "caption": caption,
        "forbidden_descriptions": ["steady growth", "monotonic increase",
                                   "steadily increasing", "consistent growth"],
    }
    if not monotonic:
        log.warning("COST CURVE IS NOT MONOTONIC: it decreases on %d of %d steps. Do NOT "
                    "describe it as steady or monotonic growth in any caption or summary; "
                    "use the generated caption, which states the overall fold-increase, "
                    "the local variation, and the plateau separately.", n_down, len(diffs))
    return out


def _report(prof: pd.DataFrame) -> None:
    a13 = prof[prof["source_log"] != "screening"]
    n_cont = int(a13["contended"].fillna(False).sum())
    n_cache = int(a13["cache_replay"].fillna(False).sum())
    n_ok = int(a13["cost_analysis_eligible"].fillna(False).sum())
    log.info("A1.3 iterations recovered      : %d", len(a13))
    log.info("  cache replays (not timings)  : %d", n_cache)
    log.info("  contended (other jobs active): %d", n_cont)
    log.info("  ELIGIBLE for the cost curve  : %d", n_ok)
    if n_cont:
        log.warning("Contended iterations must NOT be presented as clean measurements of "
                    "explanation cost. They are annotated with the competing jobs and "
                    "excluded via cost_analysis_eligible=False.")
    cols = ["iteration", "n_features", "accuracy", "fit_seconds", "shap_seconds",
            "cache_replay", "contended", "competing_jobs", "contention_evidence",
            "cost_analysis_eligible"]
    for line in a13[cols].to_string(index=False).splitlines():
        log.info(line)
    log.info("-> %s", LEAK_DIR / "explanation_cost_profile.csv")


def reproducibility_notes() -> pd.DataFrame:
    """Runtime observations for the reproducibility section, each with its regime.

    Wall-clock is a reported quantity in this paper (B5b), so every runtime figure
    carries whether it was measured under competing load. A contended timing is
    still usable as evidence of a MECHANISM when the structural cause is exact -
    but it must not be quoted as a precise measurement.
    """
    notes = [{
        "observation": "uci_full E2 explanation cost rose ~3.5x against the same data "
                       "under the previous hyperparameters",
        "measured_before_seconds": 447.0,
        "measured_after_seconds": 1581.3,
        "ratio": round(1581.3 / 447.0, 2),
        "driver": "RandomForest n_estimators 100 -> 500",
        "driver_exact": True,
        "driver_detail": ("uci_full received its own hyperparameter search in A4 "
                          "(justified: it carries 11,055 rows against uci_dedup's "
                          "5,849). That search selected 500 trees where the earlier "
                          "uci parameters used 100 - a 5x increase in the forest the "
                          "explainer must walk."),
        "mechanism": ("Interventional TreeSHAP cost scales with the total node count "
                      "of the model being explained, so explanation cost tracks model "
                      "complexity regardless of WHY the complexity changed."),
        "significance": ("An INDEPENDENT confirmation of the B5b mechanism through a "
                         "different driver. In A1.3 node count rose because features "
                         "were removed and the boundary hardened; here it rose because "
                         "a hyperparameter search chose a larger forest. Same "
                         "relationship, unrelated cause - which is what makes "
                         "explanation cost a general proxy for model complexity rather "
                         "than an artifact of the removal procedure."),
        "before_timing_clean": False,
        "after_timing_clean": False,
        "contention_note": ("NEITHER timing is uncontended: the 447s baseline ran "
                            "alongside A1.3, and the 1581s figure ran alongside "
                            "table-building commands (load sampler shows job tags "
                            "A4A5+B2 across 18:38-19:05). The 5x tree count is exact "
                            "and structural; the ~3.5x ratio is approximate and should "
                            "be quoted as 'roughly 3.5x' or re-measured on an idle "
                            "machine before being stated precisely."),
        "n_explained": 5527,
        "source": "A5 uci_full seed 42/43, half A",
    }]
    df = pd.DataFrame(notes)
    write_csv_atomic(df, RESULTS_DIR / "reproducibility_notes.csv")
    banner(log, "REPRODUCIBILITY NOTES", "-")
    for n in notes:
        log.info("%s", n["observation"])
        log.info("    %.0fs -> %.0fs (%.2fx), driver: %s", n["measured_before_seconds"],
                 n["measured_after_seconds"], n["ratio"], n["driver"])
        if not (n["before_timing_clean"] and n["after_timing_clean"]):
            log.warning("    TIMINGS CONTENDED: %s", n["contention_note"])
    log.info("-> %s", RESULTS_DIR / "reproducibility_notes.csv")
    return df


def remeasure_iterations(iterations: list[int] | None = None, seed: int = 42,
                         dataset: str = "phiusiil") -> pd.DataFrame:
    """Re-run selected A1.3 iterations with NO competing load, recording structure.

    Serves two purposes in one pass, which is why they are not separate functions:

      1. Clean timing. Contention inflates wall-clock, so contended iterations
         cannot enter the published cost curve. Re-running them uncontended
         restores the anchor points - in particular iteration 0, the "flat and
         fast while leaking" end of the curve.
      2. Tree structure. `mean_tree_depth` and `total_nodes` were never logged by
         A1.3 and cannot be recovered from the log, but they are the MECHANISM
         behind the cost curve, so they must be measured rather than left NA.

    SHAP is recomputed with use_cache=False: a cache hit would return in ~0s and
    silently produce a fake timing. The refit is required anyway to inspect tree
    structure, so nothing is wasted.

    MUST be run with no other experiment active, or it reproduces the problem it
    exists to fix.
    """
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    from src.benchmark_screen import forest_complexity
    from src.leak_analysis import GREEDY_SHAP_SAMPLE
    from src.models import ENSEMBLE_NAME, build_models
    from src.preprocessing import get_splits, get_xy
    from src.shap_utils import (compute_shap_values, global_importance,
                                make_background, select_shap_sample)
    from src.tuning import get_params_for_models

    banner(log, "B5b CLEAN RE-MEASUREMENT + TREE-STRUCTURE BACKFILL")
    curve_path = LEAK_DIR / "progressive_removal.csv"
    if not curve_path.exists():
        raise FileNotFoundError("progressive_removal.csv not found - run A1.3 first")
    curve = pd.read_csv(curve_path)

    X, _ = get_xy(dataset)
    all_features = list(X.columns)
    params = get_params_for_models(dataset)

    # Every iteration is REFIT, because tree structure is missing for all of them
    # and a refit is cheap (~25s). But SHAP - the expensive part, minutes to tens
    # of minutes - is recomputed only where the original timing was contended.
    # Re-running SHAP on an already-clean iteration would burn hours to reproduce
    # a number we already trust.
    profile_path = LEAK_DIR / "explanation_cost_profile.csv"
    contended: set[int] = set()
    if profile_path.exists():
        prof = pd.read_csv(profile_path)
        prof = prof[prof["source_log"] != "screening"]
        contended = set(prof.loc[prof["contended"].fillna(False).astype(bool),
                                 "iteration"].astype(int))
    wanted = iterations if iterations is not None else list(curve["iteration"])
    shap_for = set(wanted) if iterations is not None else contended
    log.info("refitting %d iteration(s) for tree structure; recomputing SHAP for %d "
             "contended iteration(s): %s", len(wanted), len(shap_for), sorted(shap_for))
    rows = []
    for it in wanted:
        sel = curve[curve["iteration"] == it]
        if sel.empty:
            log.warning("iteration %s not present in the A1.3 curve - skipped", it)
            continue
        # Reconstruct this iteration's feature set from the removals that preceded it.
        removed = [str(f) for f in curve[curve["iteration"] < it]["removed_feature"]
                   if isinstance(f, str) and f]
        keep = [f for f in all_features if f not in removed]
        n_expected = int(sel["n_features_remaining"].iloc[0])
        if len(keep) != n_expected:
            log.error("iteration %d: reconstructed %d features but the curve records %d - "
                      "skipping rather than measuring the wrong feature set",
                      it, len(keep), n_expected)
            continue

        splits = get_splits(dataset, seed, features=keep)
        ens = build_models(seed, params)[ENSEMBLE_NAME]
        import time as _time
        t0 = _time.perf_counter()
        ens.fit(splits.X_train, splits.y_train)
        fit_s = _time.perf_counter() - t0

        pred = ens.predict(splits.X_test)
        struct = forest_complexity(ens)
        if it in shap_for:
            bg = make_background(splits.X_train, splits.y_train, seed)
            X_exp, _ = select_shap_sample(splits.X_test, splits.y_test, seed=seed,
                                          n=GREEDY_SHAP_SAMPLE)
            t0 = _time.perf_counter()
            vals = compute_shap_values(ens, X_exp, seed, background=bg,
                                       cache_key=None, use_cache=False)
            shap_s = _time.perf_counter() - t0
            imp = global_importance(vals, keep).sort_values(ascending=False)
            top_feature, shap_clean = str(imp.index[0]), True
        else:
            # Already an uncontended timing; keep the original and refit only.
            shap_s, top_feature, shap_clean = float("nan"), "", False

        rows.append({
            "dataset": dataset, "iteration": int(it), "n_features": len(keep),
            "accuracy": float(accuracy_score(splits.y_test, pred)),
            "f1": float(f1_score(splits.y_test, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(splits.y_test, pred)),
            "fit_seconds": round(fit_s, 2), "shap_seconds": round(shap_s, 2),
            "mean_tree_depth": struct["mean_tree_depth"],
            "total_nodes": struct["total_nodes"], "n_trees": struct["n_trees"],
            "top_feature": top_feature,
            "shap_remeasured": shap_clean,
            "remeasured_clean": True,
        })
        log.info("[remeasure iter %2d] n=%2d acc=%.6f fit %5.1fs shap %s "
                 "depth %.1f nodes %s", it, len(keep), rows[-1]["accuracy"], fit_s,
                 f"{shap_s:7.1f}s" if shap_clean else "  (kept)",
                 struct["mean_tree_depth"], struct["total_nodes"])

    out = pd.DataFrame(rows)
    write_csv_atomic(out, LEAK_DIR / "explanation_cost_remeasured.csv")
    log.info("-> %s", LEAK_DIR / "explanation_cost_remeasured.csv")
    return out


if __name__ == "__main__":
    import argparse

    scratch = Path("C:/Users/PROFCS~1/AppData/Local/Temp/claude/"
                   "C--Users-Prof-CSCyber-Downloads--74----SCI-Springer-Journal-Code/"
                   "154fcbfb-0e83-474f-ba71-7992eb72f61f/scratchpad")
    ap = argparse.ArgumentParser(description="Task B5b: explanation cost profile")
    ap.add_argument("--logs", nargs="*",
                    default=[str(scratch / "a13.log"), str(scratch / "a13_resume.log")])
    ap.add_argument("--remeasure", nargs="*", type=int, default=None,
                    help="re-run these A1.3 iterations uncontended, recording tree "
                         "structure and clean timings (omit values for all)")
    args = ap.parse_args()
    if args.remeasure is not None:
        remeasure_iterations(args.remeasure or None)
    else:
        build_cost_profile([Path(p) for p in args.logs])
