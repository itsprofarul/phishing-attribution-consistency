"""Background CPU-load sampler, for annotating the Task B5b explanation-cost curve.

B5b turns TreeSHAP wall-clock into a leakage diagnostic, which makes those
timings a published measurement rather than incidental logging. A timing taken
while other jobs compete for the same cores is not a clean measurement of
explanation cost, so contention has to be recorded alongside it or the affected
iterations have to be excluded.

This samples system CPU and the set of running Python processes at a fixed
interval and appends to a CSV. It is deliberately cheap: one psutil call per
tick, a single append, no imports from the rest of the project (so it cannot
perturb what it is measuring).

Run detached alongside a long experiment:
    python -m src.load_sampler --interval 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

SELF_PID = os.getpid()


def sample(own_pid: int = SELF_PID) -> dict:
    """One observation of system load and concurrent Python work."""
    procs = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent"]):
        try:
            if p.info["name"] and "python" in p.info["name"].lower():
                if p.info["pid"] == own_pid:
                    continue
                cmd = " ".join(p.info["cmdline"] or [])
                procs.append((p.info["pid"], cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Identify which experiment each Python process belongs to, so an iteration
    # can be attributed to the specific jobs that were competing with it.
    def tag(cmd: str) -> str:
        for key, name in (("leak_analysis", "A1.3"), ("benchmark_screen", "B2"),
                          ("transfer", "B1"), ("run_addendum", "A4A5"),
                          ("faithfulness", "B4"), ("cross_dataset", "B5"),
                          ("explanation_cost", "B5b"), ("paper_export", "B6"),
                          ("experiment_e2", "E2"), ("evaluate", "E1"),
                          ("tuning", "TUNE"), ("external_datasets", "B2acq"),
                          ("joblib", "worker"), ("loky", "worker")):
            if key in cmd:
                return name
        return "other"

    tags = sorted({tag(c) for _, c in procs})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cpu_percent": psutil.cpu_percent(interval=1.0),
        "n_python_procs": len(procs),
        "job_tags": ";".join(tags),
        "load_avg_1m": (psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else ""),
    }


def run(path: Path, interval: float) -> None:
    new = not path.exists()
    while True:
        row = sample()
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        time.sleep(max(0.0, interval - 1.0))   # the cpu_percent call already slept 1s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Background CPU-load sampler")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "results" / "leak_analysis" / "load_samples.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"sampling every {args.interval}s -> {out}", flush=True)
    try:
        run(out, args.interval)
    except KeyboardInterrupt:
        sys.exit(0)
