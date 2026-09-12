"""Task A6 - feature inventory reconciliation.

E2 reported n_features = 50 for PhiUSIIL. The UCI page advertises 54 features and
the original brief specified dropping 5 identifiers, which should leave 49. This
module reconciles that discrepancy exactly, because the resulting table becomes
supplementary material in the paper.

The reconciliation is done against UCI's own variable-role metadata rather than
against the downloaded frame alone, since the cause turns out to live in that
metadata: a column the brief expected to drop is never delivered in the first
place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DEDUPLICATE,
    PHIUSIIL_DROP_COLS,
    RAW_DIR,
    RESULTS_DIR,
    UCI_IDS,
    banner,
    get_logger,
)
from src.data_loader import download_dataset, raw_target_column  # noqa: E402
from src.preprocessing import LABEL, prepare_dataset  # noqa: E402

log = get_logger("feature_inventory")

DROP_REASONS = {
    "URL": "raw URL string; all engineered URL features are already present",
    "Domain": "raw domain string; identifier, not a feature",
    "TLD": "high-cardinality categorical string; TLDLegitimateProb encodes it numerically",
    "Title": "raw page title string; DomainTitleMatchScore/URLTitleMatchScore encode it",
    "FILENAME": "per-record file identifier, carries no generalisable signal",
}


def _variables_path(name: str) -> Path:
    return RAW_DIR / f"{name}_variables.parquet"


def uci_variables(name: str) -> pd.DataFrame | None:
    """UCI's variables table (name/role/type), cached so re-runs stay offline."""
    path = _variables_path(name)
    if path.exists():
        return pd.read_parquet(path)
    try:
        from ucimlrepo import fetch_ucirepo

        repo = fetch_ucirepo(id=UCI_IDS[name])
        var = repo.variables[["name", "role", "type"]].copy()
        var.to_parquet(path, index=False)
        log.info("[%s] variables metadata cached -> %s (%d rows)", name, path.name, len(var))
        return var
    except Exception as exc:  # network unavailable: degrade, do not fail
        log.warning("[%s] could not fetch UCI variable metadata (%s). The inventory will "
                    "be built from the downloaded frame alone.", name, exc)
        return None


def build_inventory(name: str) -> tuple[pd.DataFrame, dict]:
    """One row per column in the raw download (plus any metadata-only column)."""
    banner(log, f"A6 FEATURE INVENTORY: {name}")

    raw = download_dataset(name)
    target_col = raw_target_column(name, raw)
    prepared = prepare_dataset(name)          # uses the configured dedup policy
    retained = [c for c in prepared.columns if c != LABEL]

    var = uci_variables(name)
    roles = dict(zip(var["name"], var["role"])) if var is not None else {}
    types = dict(zip(var["name"], var["type"])) if var is not None else {}

    rows = []
    for col in raw.columns:
        if col == target_col:
            status, reason = "label", "target variable (harmonised to 1=phishing, 0=legitimate)"
        elif col in retained:
            status, reason = "retained", ""
        else:
            status = "dropped"
            reason = DROP_REASONS.get(col, "dropped during preprocessing")
        rows.append({
            "dataset": name, "column": col,
            "in_raw_download": True,
            "uci_role": roles.get(col, "<unknown>"),
            "uci_type": types.get(col, "<unknown>"),
            "status": status, "reason": reason,
        })

    # Columns that exist in UCI's metadata but never reach the downloaded frame.
    # This is where the PhiUSIIL discrepancy lives, so it must be visible in the
    # table rather than inferred from a missing row.
    delivered = set(raw.columns)
    for _, r in (var if var is not None else pd.DataFrame(columns=["name", "role", "type"])).iterrows():
        if r["name"] in delivered:
            continue
        rows.append({
            "dataset": name, "column": r["name"],
            "in_raw_download": False,
            "uci_role": r["role"], "uci_type": r["type"],
            "status": "not_delivered",
            "reason": (f"present in UCI's variables metadata with role={r['role']!r}, but "
                       f"ucimlrepo returns only role='Feature' columns in data.features "
                       f"(and data.ids is None), so it never reaches the frame"),
        })

    inv = pd.DataFrame(rows)

    n_raw_cols = len(raw.columns)
    n_features_delivered = n_raw_cols - 1
    n_dropped = int((inv["status"] == "dropped").sum())
    n_retained = len(retained)
    n_not_delivered = int((inv["status"] == "not_delivered").sum())
    expected_drops_present = [c for c in PHIUSIIL_DROP_COLS if c in delivered] if name == "phiusiil" else []
    expected_drops_absent = [c for c in PHIUSIIL_DROP_COLS if c not in delivered] if name == "phiusiil" else []

    recon = {
        "dataset": name,
        "uci_variables_rows": int(len(var)) if var is not None else None,
        "columns_in_download": n_raw_cols,
        "features_delivered": n_features_delivered,
        "columns_not_delivered": n_not_delivered,
        "identifier_columns_specified": len(PHIUSIIL_DROP_COLS) if name == "phiusiil" else 0,
        "identifier_columns_actually_present": len(expected_drops_present),
        "identifier_columns_absent": expected_drops_absent,
        "columns_dropped": n_dropped,
        "features_retained": n_retained,
        "deduplicated": bool(DEDUPLICATE.get(name, True)),
        "rows_retained": int(len(prepared)),
    }

    log.info("[%s] UCI variables metadata rows : %s", name, recon["uci_variables_rows"])
    log.info("[%s] columns in the download     : %d (%d features + 1 label)",
             name, n_raw_cols, n_features_delivered)
    if n_not_delivered:
        missing = inv.loc[inv["status"] == "not_delivered", ["column", "uci_role"]]
        for _, m in missing.iterrows():
            log.warning("[%s] NOT DELIVERED: %r has role=%r, so ucimlrepo excludes it from "
                        "data.features - it is not one of the %d features and can never be "
                        "dropped by us.", name, m["column"], m["uci_role"], n_features_delivered)
    log.info("[%s] identifier columns dropped  : %d %s",
             name, n_dropped, sorted(inv.loc[inv["status"] == "dropped", "column"]))
    log.info("[%s] FEATURES RETAINED           : %d", name, n_retained)

    if name == "phiusiil":
        log.info("")
        log.info("[%s] RECONCILIATION of the 50-vs-49 discrepancy:", name)
        log.info("    the brief expected 54 features - 5 identifiers = 49")
        log.info("    actual: %d features delivered - %d identifiers actually present = %d",
                 n_features_delivered, len(expected_drops_present), n_retained)
        if expected_drops_absent:
            log.info("    cause : %s was specified for removal but is not among the %d "
                     "features; UCI gives it role=%r and ucimlrepo returns only "
                     "role='Feature' columns, so only %d of the %d specified identifiers "
                     "exist to be dropped.",
                     expected_drops_absent, n_features_delivered,
                     roles.get(expected_drops_absent[0], "<unknown>"),
                     len(expected_drops_present), len(PHIUSIIL_DROP_COLS))
        log.info("    result: 54 - 4 = %d retained features. The expected 49 assumed a "
                 "column that the download never contains.", n_retained)

    return inv, recon


def run_inventory(datasets: list[str] | None = None) -> pd.DataFrame:
    names = datasets or list(UCI_IDS)
    frames, recons = [], []
    for n in names:
        inv, recon = build_inventory(n)
        frames.append(inv)
        recons.append(recon)

    out = pd.concat(frames, ignore_index=True)
    path = RESULTS_DIR / "feature_inventory.csv"
    out.to_csv(path, index=False)
    (RESULTS_DIR / "feature_inventory_reconciliation.json").write_text(
        json.dumps(recons, indent=2), encoding="utf-8")
    log.info("")
    log.info("-> %s (%d rows)", path, len(out))
    log.info("-> %s", RESULTS_DIR / "feature_inventory_reconciliation.json")

    banner(log, "A6 SUMMARY", "-")
    for r in recons:
        log.info("%-10s delivered=%-3d dropped=%-2d retained=%-3d not_delivered=%d",
                 r["dataset"], r["features_delivered"], r["columns_dropped"],
                 r["features_retained"], r["columns_not_delivered"])
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task A6: feature inventory reconciliation")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    args = ap.parse_args()
    sel = None if args.dataset == "both" else [args.dataset]
    run_inventory(sel)
