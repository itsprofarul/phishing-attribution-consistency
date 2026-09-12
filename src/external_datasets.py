"""Acquisition of additional public phishing benchmarks for the Task B2 screen.

Every dataset records source, DOI/URL, size, feature count, class balance and the
date it was accessed, because the screening table is a headline figure of the
paper and a reviewer must be able to reproduce the acquisition exactly.

Scope rule from the brief, applied strictly: a dataset that ships RAW URLS rather
than extracted features is recorded and SKIPPED, not fed through a feature
extractor we write ourselves. Building an extractor would inject our own
engineering choices into what is supposed to be an audit of other people's
benchmarks, and any leakage we then found could be ours rather than theirs.
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR, banner, get_logger  # noqa: E402

log = get_logger("external_datasets")

EXTERNAL_DIR = RAW_DIR / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (academic dataset acquisition)"}


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
# `label_col` / `positive` are filled in after inspection; `positive` names the
# raw value meaning PHISHING, which is harmonised to 1 downstream exactly as in
# Phase A.
CATALOGUE = {
    "uci379_website_phishing": {
        "kind": "ucimlrepo",
        "uci_id": 379,
        "name": "Website Phishing (Abdelhamid)",
        "source": "UCI ML Repository",
        "url": "https://archive.ics.uci.edu/dataset/379/website+phishing",
        "doi": "10.24432/C5B301",
        "notes": "Ternary-coded features, same lineage as UCI 327 but a distinct, "
                 "smaller collection.",
    },
    "mendeley_hannousse": {
        "kind": "mendeley",
        "mendeley_id": "c2gw7fy2j4",
        "version": 1,
        "filename": "dataset_B_05_2020.csv",
        "name": "Web page phishing detection (Hannousse & Yahiouche)",
        "source": "Mendeley Data",
        "url": "https://data.mendeley.com/datasets/c2gw7fy2j4",
        "doi": "10.17632/c2gw7fy2j4.3",
        "notes": "87 extracted features over 11,430 URLs; balanced by construction.",
    },
    "mendeley_tan": {
        "kind": "mendeley",
        "mendeley_id": "h3cgnj8hft",
        "version": 1,
        "filename": None,          # resolved from the file listing
        "name": "Phishing Dataset for Machine Learning (Tan)",
        "source": "Mendeley Data",
        "url": "https://data.mendeley.com/datasets/h3cgnj8hft",
        "doi": "10.17632/h3cgnj8hft.1",
        "notes": "48 extracted features over 10,000 web pages.",
    },
}


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout,
                                context=ssl.create_default_context()) as r:
        return r.read()


def mendeley_files(dataset_id: str, version: int = 1) -> list[dict]:
    """File listing for a public Mendeley dataset (the version param is required)."""
    url = (f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
           f"/files?folder_id=root&version={version}")
    js = json.loads(_http_get(url, timeout=60))
    return js if isinstance(js, list) else js.get("results", [])


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read a CSV or ARFF into a frame.

    ARFF is not a blocker - scipy reads it directly. Its string fields arrive as
    bytes, so they are decoded here rather than silently becoming object columns
    that would later look like unusable features.
    """
    if path.suffix.lower() == ".arff":
        from scipy.io import arff

        data, _ = arff.loadarff(str(path))
        df = pd.DataFrame(data)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
        return df
    return pd.read_csv(path)


def acquire(key: str, force: bool = False) -> tuple[pd.DataFrame | None, dict]:
    """Fetch one catalogued dataset, caching the raw file under data/raw/external."""
    spec = CATALOGUE[key]
    cache = EXTERNAL_DIR / f"{key}.parquet"
    meta_path = EXTERNAL_DIR / f"{key}_meta.json"

    if cache.exists() and meta_path.exists() and not force:
        df = pd.read_parquet(cache)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info("[%s] loaded from cache -> %s", key, df.shape)
        return df, meta

    meta = {k: v for k, v in spec.items() if k != "kind"}
    meta.update({"key": key, "accessed": date.today().isoformat()})

    try:
        if spec["kind"] == "ucimlrepo":
            from ucimlrepo import fetch_ucirepo

            repo = fetch_ucirepo(id=spec["uci_id"])
            df = repo.data.features.copy()
            tgt = repo.data.targets
            tcol = tgt.columns[0] if isinstance(tgt, pd.DataFrame) else "target"
            df[tcol] = (tgt.iloc[:, 0] if isinstance(tgt, pd.DataFrame) else tgt).to_numpy()
            meta["label_column"] = tcol
        elif spec["kind"] == "mendeley":
            files = mendeley_files(spec["mendeley_id"], spec.get("version", 1))
            wanted = spec.get("filename")
            chosen = None
            for f in files:
                fn = f.get("filename", "").lower()
                if wanted and f.get("filename") == wanted:
                    chosen = f
                    break
                if not wanted and fn.endswith((".csv", ".arff")):
                    chosen = f
                    break
            if chosen is None:
                raise FileNotFoundError(
                    f"no tabular file (.csv/.arff) in the Mendeley listing: "
                    f"{[f.get('filename') for f in files]}")
            dl = (chosen.get("content_details") or {}).get("download_url")
            raw = _http_get(dl, timeout=300)
            local = EXTERNAL_DIR / chosen["filename"]
            local.write_bytes(raw)
            df = _read_tabular(local)
            meta["filename"] = chosen["filename"]
            meta["bytes"] = len(raw)
        else:
            raise ValueError(f"unknown acquisition kind {spec['kind']!r}")
    except Exception as exc:
        meta.update({"acquired": False, "block_reason": f"{type(exc).__name__}: {exc}"})
        log.error("[%s] ACQUISITION FAILED: %s", key, meta["block_reason"])
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return None, meta

    meta.update({"acquired": True, "shape": list(df.shape),
                 "n_columns": int(df.shape[1]), "n_rows": int(df.shape[0])})
    df.to_parquet(cache, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("[%s] acquired %s -> %s", key, df.shape, cache.name)
    return df, meta


def acquire_all(force: bool = False) -> dict[str, dict]:
    banner(log, "B2.2 EXTERNAL DATASET ACQUISITION")
    out = {}
    for key in CATALOGUE:
        df, meta = acquire(key, force=force)
        out[key] = meta
        if df is not None:
            log.info("    %-28s %-10s cols=%s", key, str(df.shape),
                     list(df.columns[:6]) + (["..."] if df.shape[1] > 6 else []))
    (EXTERNAL_DIR / "acquisition_report.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="B2.2: acquire external phishing benchmarks")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rep = acquire_all(force=args.force)
    banner(log, "ACQUISITION SUMMARY", "-")
    for k, m in rep.items():
        status = "OK" if m.get("acquired") else f"BLOCKED ({m.get('block_reason','')[:70]})"
        log.info("%-28s %-8s %s", k, str(m.get("shape", "")), status)
