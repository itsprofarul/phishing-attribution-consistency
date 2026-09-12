"""Task 1 - dataset acquisition and diagnostics.

Fetches PhiUSIIL (UCI id 967) and Phishing Websites (UCI id 327) from the UCI ML
Repository and caches them to data/raw/*.parquet so re-runs never re-download.

The raw label column is preserved untouched under its original name; label
harmonisation happens in preprocessing.py so that the raw convention stays
auditable on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    EXPECTED_SHAPE,
    RAW_DIR,
    UCI_IDS,
    banner,
    get_logger,
)

log = get_logger("data_loader")

# The raw target column name shipped by each dataset.
RAW_TARGET_COL = {"phiusiil": "label", "uci": "Result"}


def _parquet_path(name: str) -> Path:
    return RAW_DIR / f"{name}.parquet"


def _meta_path(name: str) -> Path:
    return RAW_DIR / f"{name}_meta.json"


def download_dataset(name: str, force: bool = False) -> pd.DataFrame:
    """Fetch one dataset from UCI (or load the parquet cache) as a single frame."""
    if name not in UCI_IDS:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {list(UCI_IDS)}")

    path = _parquet_path(name)
    if path.exists() and not force:
        df = pd.read_parquet(path)
        log.info("[%s] loaded from cache %s -> %s", name, path.name, df.shape)
        return df

    from ucimlrepo import fetch_ucirepo

    uid = UCI_IDS[name]
    log.info("[%s] fetching UCI id=%d (no cache at %s)", name, uid, path)
    repo = fetch_ucirepo(id=uid)

    features = repo.data.features
    targets = repo.data.targets
    if isinstance(targets, pd.DataFrame):
        if targets.shape[1] != 1:
            raise ValueError(f"[{name}] expected a single target column, got {list(targets.columns)}")
        target_name = targets.columns[0]
        target = targets.iloc[:, 0]
    else:
        target_name = RAW_TARGET_COL[name]
        target = targets

    df = features.copy()
    df[target_name] = target.to_numpy()

    df.to_parquet(path, index=False)
    meta = {
        "uci_id": uid,
        "name": repo.metadata.get("name") if isinstance(repo.metadata, dict) else str(repo.metadata),
        "raw_target_column": target_name,
        "shape": list(df.shape),
        "columns": list(df.columns),
    }
    _meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("[%s] downloaded and cached -> %s (target column %r)", name, df.shape, target_name)
    return df


def raw_target_column(name: str, df: pd.DataFrame) -> str:
    """Resolve the raw label column, preferring the on-disk metadata record."""
    meta_file = _meta_path(name)
    if meta_file.exists():
        recorded = json.loads(meta_file.read_text(encoding="utf-8")).get("raw_target_column")
        if recorded in df.columns:
            return recorded
    fallback = RAW_TARGET_COL[name]
    if fallback in df.columns:
        return fallback
    raise KeyError(f"[{name}] cannot locate the raw target column among {list(df.columns)[-5:]}")


def diagnostic_report(name: str, df: pd.DataFrame) -> dict:
    """Print shape, dtypes, missing counts, class distribution and head(5)."""
    banner(log, f"DATASET DIAGNOSTICS: {name}", "-")
    target = raw_target_column(name, df)

    log.info("[%s] shape: %s (features + label)", name, df.shape)
    expected = EXPECTED_SHAPE.get(name)
    if expected is not None:
        got = (df.shape[0], df.shape[1] - 1)  # published shapes exclude the label
        if got != tuple(expected):
            # Not fatal: UCI occasionally reorders or renames, but it must be visible.
            log.warning("[%s] (rows, n_features) = %s differs from the published %s",
                        name, got, expected)
        else:
            log.info("[%s] (rows, n_features) = %s matches the published shape", name, got)

    log.info("[%s] dtypes:\n%s", name, df.dtypes.to_string())

    missing = df.isna().sum()
    total_missing = int(missing.sum())
    log.info("[%s] total missing values: %d", name, total_missing)
    if total_missing:
        log.info("[%s] per-column missing (non-zero only):\n%s",
                 name, missing[missing > 0].to_string())
    else:
        log.info("[%s] per-column missing: all zero", name)

    counts = df[target].value_counts(dropna=False).sort_index()
    pct = (counts / len(df) * 100).round(2)
    log.info("[%s] RAW class distribution in %r (convention NOT yet harmonised):",
             name, target)
    for value, count in counts.items():
        log.info("    %-6s -> %8d  (%5.2f%%)", value, count, pct[value])

    with pd.option_context("display.max_columns", 60, "display.width", 250):
        log.info("[%s] head(5):\n%s", name, df.head(5).to_string())

    return {
        "dataset": name,
        "shape": list(df.shape),
        "raw_target_column": target,
        "total_missing": total_missing,
        "raw_class_counts": {str(k): int(v) for k, v in counts.items()},
    }


def load_all(force: bool = False, datasets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load (and diagnose) every requested dataset."""
    names = datasets or list(UCI_IDS)
    frames: dict[str, pd.DataFrame] = {}
    reports = []
    for name in names:
        df = download_dataset(name, force=force)
        reports.append(diagnostic_report(name, df))
        frames[name] = df
    (RAW_DIR / "load_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return frames


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task 1: fetch and diagnose the datasets")
    ap.add_argument("--force", action="store_true", help="re-download, ignoring the parquet cache")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    args = ap.parse_args()
    sel = None if args.dataset == "both" else [args.dataset]
    load_all(force=args.force, datasets=sel)
