"""Task 2 - label harmonisation, column removal, de-duplication, splits, scaling.

The single most important piece of code in the project lives here: the assertion
in `harmonise_labels`. The two datasets ship with OPPOSITE label conventions, and
a silent inversion would invert every SHAP sign in the paper while leaving the
accuracy figures untouched (and therefore undetectable downstream). The published
class balances are used as an external ground truth and a mismatch is fatal.

Conventions after this module:  1 = phishing, 0 = legitimate  (BOTH datasets).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CACHE_DIR,
    DEDUPLICATE,
    EXPECTED_PHISHING_RATE,
    EXPECTED_RATE_TOLERANCE,
    PHIUSIIL_DROP_COLS,
    RESULTS_DIR,
    TEST_SIZE,
    VAL_SIZE,
    ExperimentConfig,
    banner,
    get_logger,
    resolve_config,
    set_global_seed,
)
from src.data_loader import download_dataset, raw_target_column  # noqa: E402

log = get_logger("preprocessing")

LABEL = "label"  # harmonised target name used everywhere downstream


class LabelHarmonisationError(AssertionError):
    """Raised when the harmonised class balance contradicts published figures."""


@dataclass
class Splits:
    """One seeded 70/15/15 split, scaled with a train-only StandardScaler."""

    dataset: str
    seed: int
    feature_names: list[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler

    def summary(self) -> str:
        return (
            f"[{self.dataset} seed={self.seed}] "
            f"train={self.X_train.shape} val={self.X_val.shape} test={self.X_test.shape} "
            f"phishing rate train/val/test = "
            f"{self.y_train.mean():.4f}/{self.y_val.mean():.4f}/{self.y_test.mean():.4f}"
        )


# --------------------------------------------------------------------------- #
# 2.1 Label harmonisation
# --------------------------------------------------------------------------- #
def harmonise_labels(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Map both datasets onto 1 = phishing, 0 = legitimate, then verify.

    PhiUSIIL ships 1 = legitimate, 0 = phishing  -> invert.
    UCI      ships 1 = legitimate, -1 = phishing -> map -1 to 1 and 1 to 0.

    The mapping statements above are NOT trusted. After mapping, the resulting
    phishing fraction is compared against the published figure and any deviation
    beyond EXPECTED_RATE_TOLERANCE aborts the run.
    """
    banner(log, f"LABEL HARMONISATION: {name}", "-")
    df = df.copy()
    raw_col = raw_target_column(name, df)
    raw_values = sorted(pd.unique(df[raw_col]).tolist())
    log.info("[%s] raw label column %r with values %s", name, raw_col, raw_values)

    if name == "phiusiil":
        expected_raw = {0, 1}
        if set(raw_values) != expected_raw:
            raise LabelHarmonisationError(
                f"[{name}] expected raw labels {expected_raw}, found {set(raw_values)}"
            )
        mapping = {1: 0, 0: 1}  # 1 legitimate -> 0 ; 0 phishing -> 1
        log.info("[%s] applying inversion: raw 0 (phishing) -> 1, raw 1 (legitimate) -> 0", name)
    elif name == "uci":
        expected_raw = {-1, 1}
        if set(raw_values) != expected_raw:
            raise LabelHarmonisationError(
                f"[{name}] expected raw labels {expected_raw}, found {set(raw_values)}"
            )
        mapping = {1: 0, -1: 1}  # 1 legitimate -> 0 ; -1 phishing -> 1
        log.info("[%s] applying mapping: raw -1 (phishing) -> 1, raw 1 (legitimate) -> 0", name)
    else:
        raise KeyError(f"No harmonisation rule defined for dataset {name!r}")

    harmonised = df[raw_col].map(mapping).astype("int8")
    if harmonised.isna().any():
        raise LabelHarmonisationError(f"[{name}] label mapping produced NaNs")

    df = df.drop(columns=[raw_col])
    df[LABEL] = harmonised.to_numpy()

    # ---- the assertion the whole study rests on ---------------------------- #
    n_phish = int((df[LABEL] == 1).sum())
    n_total = len(df)
    rate = n_phish / n_total
    expected = EXPECTED_PHISHING_RATE[name]
    log.info("[%s] harmonised: phishing(1)=%d legitimate(0)=%d  ->  phishing rate %.4f "
             "(published %.4f)", name, n_phish, n_total - n_phish, rate, expected)

    if abs(rate - expected) > EXPECTED_RATE_TOLERANCE:
        raise LabelHarmonisationError(
            f"[{name}] LABEL HARMONISATION FAILED: phishing rate after mapping is "
            f"{rate:.4f} but published work reports {expected:.4f} "
            f"(tolerance {EXPECTED_RATE_TOLERANCE}). The label convention has changed "
            f"or the mapping is inverted. Refusing to continue - every SHAP sign in "
            f"the study would be wrong."
        )
    log.info("[%s] LABEL ASSERTION PASSED (|%.4f - %.4f| = %.4f <= %.4f)",
             name, rate, expected, abs(rate - expected), EXPECTED_RATE_TOLERANCE)
    return df


# --------------------------------------------------------------------------- #
# 2.2 Column removal
# --------------------------------------------------------------------------- #
def drop_identifier_columns(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop free-text identifier columns (PhiUSIIL only)."""
    if name != "phiusiil":
        log.info("[%s] no identifier columns to drop", name)
        return df, []

    reasons = {
        "FILENAME": "per-record file identifier, carries no generalisable signal",
        "URL": "raw URL string; all engineered URL features are already present",
        "Domain": "raw domain string; identifier, not a feature",
        "TLD": "high-cardinality categorical string; TLDLegitimateProb encodes it numerically",
        "Title": "raw page title string; DomainTitleMatchScore/URLTitleMatchScore encode it",
    }
    present = [c for c in PHIUSIIL_DROP_COLS if c in df.columns]
    missing = [c for c in PHIUSIIL_DROP_COLS if c not in df.columns]
    if missing:
        log.warning("[%s] expected-to-drop columns absent from the frame: %s", name, missing)

    log.info("[%s] dropping %d identifier columns:", name, len(present))
    for c in present:
        log.info("    - %-10s : %s", c, reasons[c])
    df = df.drop(columns=present)

    leftover = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
    if leftover:
        raise ValueError(f"[{name}] non-numeric columns remain after removal: {leftover}")
    return df, present


# --------------------------------------------------------------------------- #
# 2.3 Duplicates
# --------------------------------------------------------------------------- #
def remove_duplicates(name: str, df: pd.DataFrame,
                      apply_removal: bool | None = None) -> tuple[pd.DataFrame, dict]:
    """Measure exact duplicate rows, and remove them only when policy says so.

    The duplicate RATE is a property of the dataset that the paper reports either
    way, so it is always computed and always logged. Whether the rows are actually
    dropped is governed by config.DEDUPLICATE - see the rationale there: UCI's
    ternary coding makes identical feature vectors expected collisions between
    distinct websites rather than duplicated records, so removing them would
    discard real data and break comparability with the published class balance.
    """
    if apply_removal is None:
        apply_removal = DEDUPLICATE.get(name, True)

    before = len(df)
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())

    # Feature-only duplicates that disagree on the label are contradictory rows:
    # worth reporting because they cap achievable accuracy.
    feat_cols = [c for c in df.columns if c != LABEL]
    n_feat_dup = int(df.duplicated(subset=feat_cols, keep="first").sum())
    n_conflicting = n_feat_dup - n_dup

    log.info("[%s] duplicate audit: rows=%d exact_duplicate_rows=%d (%.3f%%) "
             "feature-vector_duplicates=%d conflicting=%d",
             name, before, n_dup, 100 * n_dup / max(before, 1), n_feat_dup, n_conflicting)

    if apply_removal:
        df = df.loc[~dup_mask].reset_index(drop=True)
        log.info("[%s] de-duplication APPLIED: %d -> %d rows", name, before, len(df))
    else:
        log.info("[%s] de-duplication NOT applied (config.DEDUPLICATE[%r]=False): all %d "
                 "rows retained; the %d exact duplicates are reported but kept. See the "
                 "rationale recorded alongside config.DEDUPLICATE.", name, name, before, n_dup)

    if n_conflicting > 0:
        log.warning("[%s] %d rows share an identical feature vector but disagree on the "
                    "label - these are irreducible errors and cap achievable accuracy",
                    name, n_conflicting)
    return df, {
        "deduplication_applied": bool(apply_removal),
        "rows_before": before,
        "exact_duplicates_detected": n_dup,
        "exact_duplicates_removed": n_dup if apply_removal else 0,
        "feature_vector_duplicates": n_feat_dup,
        "duplicate_rate": round(n_dup / max(before, 1), 6),
        "rows_after": len(df),
        "conflicting_feature_duplicates": n_conflicting,
    }


# --------------------------------------------------------------------------- #
# Prepared-dataset pipeline (2.1 -> 2.3), cached
# --------------------------------------------------------------------------- #
def _variant(dedup: bool) -> str:
    return "dedup" if dedup else "full"


def _prepared_path(name: str, dedup: bool) -> Path:
    return CACHE_DIR / f"{name}_prepared_{_variant(dedup)}.parquet"


def _report_path(name: str, dedup: bool) -> Path:
    return RESULTS_DIR / f"preprocessing_{name}_{_variant(dedup)}.json"


def prepare_dataset(name: str, dedup: bool | None = None, force: bool = False) -> pd.DataFrame:
    """Harmonise -> drop identifiers -> audit (and optionally remove) duplicates.

    Cached to data/cache, keyed by the de-duplication variant so the primary and
    sensitivity analyses can coexist without invalidating each other's cache.

    Note on which rate is checked where: the published class balance describes the
    RAW dataset, so it is asserted inside harmonise_labels (pre-de-duplication).
    De-duplication legitimately shifts the balance - dramatically so for the UCI
    dataset - therefore the cache guard compares against the post-de-duplication
    rate recorded when the cache was written, not against the published figure.
    """
    if dedup is None:
        dedup = DEDUPLICATE.get(name, True)
    path = _prepared_path(name, dedup)
    report_file = _report_path(name, dedup)
    if path.exists() and report_file.exists() and not force:
        df = pd.read_parquet(path)
        report = json.loads(report_file.read_text(encoding="utf-8"))
        rate = float(df[LABEL].mean())
        recorded = report.get("phishing_rate")
        if recorded is None or abs(rate - recorded) > 1e-6:
            raise LabelHarmonisationError(
                f"[{name}] cached prepared frame has phishing rate {rate:.6f} but the "
                f"recorded value is {recorded}. The cache is stale - delete {path} "
                f"and re-run."
            )
        # The published-balance assertion, re-verified from the recorded pre-dedup rate.
        pre = report.get("phishing_rate_before_dedup")
        if pre is None or abs(pre - EXPECTED_PHISHING_RATE[name]) > EXPECTED_RATE_TOLERANCE:
            raise LabelHarmonisationError(
                f"[{name}] cached run recorded a pre-de-duplication phishing rate of "
                f"{pre}, which contradicts the published {EXPECTED_PHISHING_RATE[name]}. "
                f"Delete {path} and re-run."
            )
        log.info("[%s] prepared frame loaded from cache -> %s (phishing rate %.4f, "
                 "pre-dedup %.4f matches published)", name, df.shape, rate, pre)
        return df

    raw = download_dataset(name)
    df = harmonise_labels(name, raw)
    rate_before_dedup = float(df[LABEL].mean())
    df, dropped = drop_identifier_columns(name, df)
    df, dup_stats = remove_duplicates(name, df, apply_removal=dedup)

    rate_after = float(df[LABEL].mean())
    if abs(rate_after - rate_before_dedup) > 0.02:
        log.warning("[%s] de-duplication shifted the class balance materially: "
                    "%.4f -> %.4f phishing. This is a reportable dataset property, "
                    "not a bug.", name, rate_before_dedup, rate_after)

    # With removal switched off the prepared frame still describes the published
    # dataset, so the published-balance assertion must hold end-to-end and not
    # merely pre-de-duplication. Checking it here catches any later change to the
    # column-dropping or harmonisation path that silently alters the balance.
    if not dedup and abs(rate_after - EXPECTED_PHISHING_RATE[name]) > EXPECTED_RATE_TOLERANCE:
        raise LabelHarmonisationError(
            f"[{name}] de-duplication is disabled, so the prepared frame should still "
            f"carry the published phishing rate {EXPECTED_PHISHING_RATE[name]:.4f}, but "
            f"it is {rate_after:.4f}."
        )

    df.to_parquet(path, index=False)
    report = {
        "dataset": name,
        "deduplicated": bool(dedup),
        "dropped_columns": dropped,
        "duplicates": dup_stats,
        "final_shape": list(df.shape),
        "n_features": df.shape[1] - 1,
        "phishing_rate": rate_after,
        "phishing_rate_before_dedup": rate_before_dedup,
        "published_phishing_rate": EXPECTED_PHISHING_RATE[name],
        "phishing_count": int(df[LABEL].sum()),
        "legitimate_count": int((df[LABEL] == 0).sum()),
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("[%s/%s] prepared frame: %s, %d features, phishing rate %.4f (published %.4f)",
             name, _variant(dedup), df.shape, df.shape[1] - 1, rate_after,
             EXPECTED_PHISHING_RATE[name])
    return df


def get_xy(name: str, dedup: bool | None = None, features: Sequence[str] | None = None,
           force: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and harmonised label, optionally restricted to `features`."""
    df = prepare_dataset(name, dedup=dedup, force=force)
    X, y = df.drop(columns=[LABEL]), df[LABEL].astype(int)
    if features is not None:
        missing = [f for f in features if f not in X.columns]
        if missing:
            raise KeyError(f"[{name}] requested features absent from the frame: {missing}")
        X = X.loc[:, list(features)]
    return X, y


# --------------------------------------------------------------------------- #
# 2.4 / 2.5 Splits and scaling
# --------------------------------------------------------------------------- #
_SPLIT_CACHE: dict[tuple, Splits] = {}


def get_splits(dataset: str, seed: int, dedup: bool | None = None,
               features: Sequence[str] | None = None, force: bool = False) -> Splits:
    """Stratified 70/15/15 train/val/test split for one seed, scaled train-only.

    Two-stage split: first hold out the test fraction, then carve the validation
    fraction out of the remainder so the final proportions are exactly 70/15/15.
    """
    if dedup is None:
        dedup = DEDUPLICATE.get(dataset, True)
    feat_key = None if features is None else tuple(features)
    key = (dataset, seed, dedup, feat_key)
    if key in _SPLIT_CACHE and not force:
        return _SPLIT_CACHE[key]

    set_global_seed(seed)
    X, y = get_xy(dataset, dedup=dedup, features=features)
    feature_names = list(X.columns)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed, shuffle=True)
    val_fraction_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, stratify=y_temp,
        random_state=seed, shuffle=True)

    # 2.5 Scaling: fit on TRAIN ONLY, transform val/test.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # The scaler must have seen exactly the training rows and nothing else.
    # n_samples_seen_ is incremented by every partial/full fit, so this catches
    # both an accidental second fit and a fit on the wrong frame.
    n_seen = scaler.n_samples_seen_
    n_seen = int(np.max(n_seen)) if np.ndim(n_seen) else int(n_seen)
    assert n_seen == len(X_train), (
        f"[{dataset} seed={seed}] scaler saw {n_seen} samples but the training split has "
        f"{len(X_train)} - the scaler was fitted on val/test data (LEAKAGE)."
    )
    assert len(set(X_train.index) & set(X_val.index)) == 0, "train/val overlap"
    assert len(set(X_train.index) & set(X_test.index)) == 0, "train/test overlap"
    assert len(set(X_val.index) & set(X_test.index)) == 0, "val/test overlap"

    splits = Splits(
        dataset=dataset, seed=seed, feature_names=feature_names,
        X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
        y_train=y_train.to_numpy(), y_val=y_val.to_numpy(), y_test=y_test.to_numpy(),
        scaler=scaler,
    )
    import joblib
    tag = f"{dataset}_{_variant(dedup)}"
    if feat_key is not None:
        tag += f"_n{len(feat_key)}"
    joblib.dump(scaler, CACHE_DIR / f"scaler_{tag}_seed{seed}.joblib")

    _SPLIT_CACHE[key] = splits
    log.info(splits.summary())
    return splits


def get_splits_for_config(config: ExperimentConfig | str, seed: int,
                          force: bool = False) -> Splits:
    """`get_splits` addressed by experiment-config name (Tasks A4/A5)."""
    cfg = resolve_config(config) if isinstance(config, str) else config
    return get_splits(cfg.dataset, seed, dedup=cfg.dedup, features=cfg.features, force=force)


def stratified_halves(dataset: str, seed: int, dedup: bool | None = None,
                      features: Sequence[str] | None = None,
                      ) -> tuple[tuple[np.ndarray, np.ndarray],
                                 tuple[np.ndarray, np.ndarray],
                                 list[str]]:
    """Task 9 helper: two disjoint, class-stratified halves of the full dataset.

    Each half is scaled by its OWN train-only scaler (see run_e2), so this returns
    unscaled arrays plus the feature names.
    """
    set_global_seed(seed)
    X, y = get_xy(dataset, dedup=dedup, features=features)
    idx_a, idx_b = train_test_split(
        np.arange(len(X)), test_size=0.5, stratify=y, random_state=seed, shuffle=True)
    assert len(set(idx_a) & set(idx_b)) == 0, "halves are not disjoint"
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy()
    return (Xv[idx_a], yv[idx_a]), (Xv[idx_b], yv[idx_b]), list(X.columns)


def stratified_halves_for_config(config: ExperimentConfig | str, seed: int):
    cfg = resolve_config(config) if isinstance(config, str) else config
    return stratified_halves(cfg.dataset, seed, dedup=cfg.dedup, features=cfg.features)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task 2: preprocessing")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--force", action="store_true", help="ignore the prepared-frame cache")
    ap.add_argument("--both-variants", action="store_true",
                    help="prepare the de-duplicated AND full variant of each dataset")
    args = ap.parse_args()

    names = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
    for ds in names:
        banner(log, f"PREPROCESSING: {ds}")
        variants = [True, False] if args.both_variants else [DEDUPLICATE.get(ds, True)]
        for dd in variants:
            prepare_dataset(ds, dedup=dd, force=args.force)
        get_splits(ds, seed=42, force=True)
