"""Task 8 - consistency metrics between two feature-importance profiles.

These five functions produce the paper's headline numbers, so they are written to
be obvious rather than clever, and they are covered by tests/test_consistency.py.

Vocabulary used throughout:
  * "importance"  - a vector of per-feature magnitudes (mean |SHAP|), aligned to a
                    shared feature ordering. Higher = more important.
  * "ranking"     - any vector that induces an ordering; Kendall's tau is computed
                    on the values directly, since tau is invariant to monotone
                    transformations of the inputs.
Both inputs to every function MUST be aligned to the same feature order. Callers
that hold dicts should use `align(a, b)` first.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import kendalltau

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import JACCARD_KS, N_BOOTSTRAP, N_PERMUTATIONS, get_logger  # noqa: E402

log = get_logger("consistency")


def align(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two {feature: value} dicts onto their shared feature ordering."""
    shared = [k for k in a if k in b]
    if not shared:
        raise ValueError("the two importance dicts share no features")
    if len(shared) != len(a) or len(shared) != len(b):
        log.warning("aligning on %d shared features (a has %d, b has %d)",
                    len(shared), len(a), len(b))
    return (np.array([a[k] for k in shared], dtype=float),
            np.array([b[k] for k in shared], dtype=float),
            shared)


def _as_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty input")
    return arr


# --------------------------------------------------------------------------- #
# 1. Kendall's tau-b
# --------------------------------------------------------------------------- #
def kendall_tau(ranking_a, ranking_b) -> tuple[float, float]:
    """Kendall's tau-b between two importance/ranking vectors.

    tau-b (scipy's default) is used rather than tau-a because SHAP importance
    vectors routinely contain ties at exactly zero for features the model never
    split on; tau-b corrects for those ties.

    Returns (statistic, p_value).
    """
    a, b = _as_array(ranking_a), _as_array(ranking_b)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    res = kendalltau(a, b, variant="b")
    return float(res.statistic), float(res.pvalue)


# --------------------------------------------------------------------------- #
# 2. Permutation null for tau
# --------------------------------------------------------------------------- #
def permutation_null_tau(a, b, n_perm: int = N_PERMUTATIONS, seed: int = 42) -> dict:
    """Empirical p-value for tau against a null of unrelated rankings.

    The analytic p-value from kendalltau assumes no ties and an asymptotic normal
    approximation, both shaky for short, tie-heavy SHAP vectors (30-50 features).
    This shuffles one vector to rebuild the null distribution directly.

    The p-value is two-sided: it counts null taus at least as extreme in absolute
    value as the observed one. The (n_extreme + 1) / (n_perm + 1) form keeps the
    estimate from ever being exactly zero, which is the standard correction for a
    finite number of permutations.
    """
    a, b = _as_array(a), _as_array(b)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")

    observed, _ = kendall_tau(a, b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    b_perm = b.copy()
    for i in range(n_perm):
        rng.shuffle(b_perm)
        null[i] = kendalltau(a, b_perm, variant="b").statistic
    null = np.nan_to_num(null, nan=0.0)

    n_extreme = int(np.sum(np.abs(null) >= abs(observed)))
    p_emp = (n_extreme + 1) / (n_perm + 1)
    return {
        "tau": observed,
        "p_empirical": float(p_emp),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_perm": int(n_perm),
        "n_extreme": n_extreme,
    }


# --------------------------------------------------------------------------- #
# 3. Jaccard overlap of the top-k sets
# --------------------------------------------------------------------------- #
def jaccard_at_k(importance_a, importance_b, k: int) -> float:
    """Jaccard similarity of the top-k features under each importance vector.

    Ties at the k-th position are broken by index order (numpy argsort is stable),
    which is deterministic given a fixed feature ordering. With |A| = |B| = k the
    Jaccard index reduces to overlap / (2k - overlap).
    """
    a, b = _as_array(importance_a), _as_array(importance_b)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, a.size)

    top_a = set(np.argsort(-a, kind="stable")[:k].tolist())
    top_b = set(np.argsort(-b, kind="stable")[:k].tolist())
    union = top_a | top_b
    return float(len(top_a & top_b) / len(union))


def jaccard_profile(importance_a, importance_b, ks: Sequence[int] = tuple(JACCARD_KS)) -> dict:
    return {f"jaccard_{k}": jaccard_at_k(importance_a, importance_b, k) for k in ks}


# --------------------------------------------------------------------------- #
# 4. Sign agreement  (Task A2)
# --------------------------------------------------------------------------- #
# Phase A computed sign agreement over ALL features and got ~50% on both datasets
# (PhiUSIIL 43.6%, UCI 50.7%). That is not a finding, it is an artefact: most
# features have a mean signed SHAP of approximately zero, their sign is numerical
# noise, and noise agrees with noise half the time. Averaging over the long tail
# measures the tail, not the signal.
#
# The corrected metric restricts the comparison to features that BOTH attribution
# sets consider important. A threshold-free importance-weighted variant is also
# provided as a robustness check, and the original all-features version is kept as
# `sign_agreement_unfiltered` - the gap between the two is itself reportable.

SIGN_MIN_FEATURES = 3   # below this the percentage is too coarse to mean anything


class SignAgreementResult(NamedTuple):
    """Percentage plus the n it was computed over, so `n` can be reported with it.

    Ordered so that element 0 is the percentage, which lets the result be passed
    straight into `bootstrap_ci` alongside the scalar-returning metrics.
    """

    percent: float
    n_features: int
    features: tuple[int, ...]


def sign_agreement(signed_a, signed_b, abs_a, abs_b, top_k: int = 10) -> SignAgreementResult:
    """Sign agreement over features important in BOTH attribution sets.

    Selection: take each set's top_k features by mean |SHAP| and keep those
    present in both. (The brief phrases this as "union ... then keep only those
    appearing in both sets' top_k", which is the intersection - implemented as
    such.) Fewer than SIGN_MIN_FEATURES qualifying features returns NaN with a
    warning rather than a misleadingly precise percentage.

    Operates on signed importances: it answers "does this feature push predictions
    towards phishing in both models?", which is the direction question a reader
    actually cares about.
    """
    a, b = _as_array(signed_a), _as_array(signed_b)
    aa, ab = _as_array(abs_a), _as_array(abs_b)
    if not (a.size == b.size == aa.size == ab.size):
        raise ValueError(f"length mismatch: signed {a.size}/{b.size}, abs {aa.size}/{ab.size}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    k = min(top_k, a.size)
    top_a = set(np.argsort(-aa, kind="stable")[:k].tolist())
    top_b = set(np.argsort(-ab, kind="stable")[:k].tolist())
    keep = sorted(top_a & top_b)

    if len(keep) < SIGN_MIN_FEATURES:
        log.warning("sign_agreement(top_k=%d): only %d feature(s) are in both top-k sets "
                    "(minimum %d) - returning NaN rather than a percentage over too few "
                    "features.", top_k, len(keep), SIGN_MIN_FEATURES)
        return SignAgreementResult(float("nan"), len(keep), tuple(keep))

    idx = np.array(keep, dtype=int)
    pct = float(100.0 * np.mean(np.sign(a[idx]) == np.sign(b[idx])))
    return SignAgreementResult(pct, len(keep), tuple(keep))


def sign_agreement_weighted(signed_a, signed_b, abs_a, abs_b) -> SignAgreementResult:
    """Importance-weighted sign agreement across ALL features - threshold-free.

    Each feature's sign match is weighted by its importance, normalised to sum to
    one, so the noisy near-zero tail contributes almost nothing without any
    cut-off having to be chosen.

    The weight is the mean of the two sets' |SHAP| rather than either one alone:
    a consistency metric must not depend on the order of its arguments, and
    weighting by only one side would make it do exactly that.
    """
    a, b = _as_array(signed_a), _as_array(signed_b)
    aa, ab = _as_array(abs_a), _as_array(abs_b)
    if not (a.size == b.size == aa.size == ab.size):
        raise ValueError(f"length mismatch: signed {a.size}/{b.size}, abs {aa.size}/{ab.size}")

    w = 0.5 * (aa + ab)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        log.warning("sign_agreement_weighted: all importances are zero - returning NaN.")
        return SignAgreementResult(float("nan"), 0, ())
    w = w / total
    pct = float(100.0 * np.sum(w * (np.sign(a) == np.sign(b))))
    return SignAgreementResult(pct, int(a.size), tuple(range(a.size)))


def sign_agreement_unfiltered(signed_a, signed_b) -> float:
    """The original all-features metric, retained for comparison.

    Kept because the GAP between this and the thresholded version is itself
    informative: a large gap is direct evidence that the unfiltered number was
    dominated by the unimportant tail, which is the justification for thresholding
    that the paper's methodology section will need.

    Exact zeros are treated as agreeing only with other exact zeros - a feature the
    model never used has no direction, and calling that a match with a positive
    contribution would inflate the metric.
    """
    a, b = _as_array(signed_a), _as_array(signed_b)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    return float(100.0 * np.mean(np.sign(a) == np.sign(b)))


def sign_agreement_profile(signed_a, signed_b, abs_a, abs_b,
                           ks: Sequence[int] = tuple(JACCARD_KS)) -> dict:
    """Every sign-agreement variant at once, mirroring `jaccard_profile`."""
    out: dict[str, float] = {}
    for k in ks:
        res = sign_agreement(signed_a, signed_b, abs_a, abs_b, top_k=k)
        out[f"sign_agreement_top{k}"] = res.percent
        out[f"n_features_sign_top{k}"] = res.n_features
    weighted = sign_agreement_weighted(signed_a, signed_b, abs_a, abs_b)
    out["sign_agreement_weighted"] = weighted.percent
    out["sign_agreement_unfiltered"] = sign_agreement_unfiltered(signed_a, signed_b)
    return out


# --------------------------------------------------------------------------- #
# 4b. Directional consistency  (Task B3)
# --------------------------------------------------------------------------- #
# The addendum established that mean signed SHAP is the wrong direction statistic:
# it cancels for any feature that pushes both ways at different values, so its
# sign is noise even for the most important features (url_of_anchor: mean |SHAP|
# 0.1504, mean signed SHAP +0.0033). Spearman rho between a feature's VALUES and
# its SHAP VALUES does not cancel, because it correlates the contribution with the
# value rather than averaging the contribution away.
#
# This promotes that from a supplementary column to a defined metric. The
# threshold rho_min exists because a near-zero rho has no reliable sign either -
# agreement on a direction that neither model really expresses is not agreement.

DEFAULT_RHO_MIN = 0.3
DIRECTIONAL_RHO_MINS = (0.3, 0.5)


def feature_directions(X, shap_values, feature_names: Sequence[str] | None = None) -> np.ndarray:
    """Per-feature Spearman rho between feature value and SHAP value.

    Rank correlation rather than Pearson: the effect need only be monotone, not
    linear. Rank correlation is also invariant to any monotone rescaling, so a
    StandardScaler applied upstream cannot change the answer.

    A feature that is constant across the sample, or that the model never used,
    has no direction and returns 0.0 - which fails any positive rho_min and is
    therefore counted as "no reliable direction" rather than silently dropped.
    """
    from scipy.stats import spearmanr

    Xv = np.asarray(X, dtype=float)
    S = np.asarray(shap_values, dtype=float)
    if Xv.shape != S.shape:
        raise ValueError(f"feature matrix {Xv.shape} does not match SHAP array {S.shape}")
    if feature_names is not None and len(feature_names) != Xv.shape[1]:
        raise ValueError(f"{Xv.shape[1]} columns vs {len(feature_names)} feature names")

    out = np.zeros(Xv.shape[1], dtype=float)
    for j in range(Xv.shape[1]):
        xj, sj = Xv[:, j], S[:, j]
        if np.ptp(xj) == 0 or np.ptp(sj) == 0:
            continue
        rho = spearmanr(xj, sj).statistic
        out[j] = 0.0 if not np.isfinite(rho) else float(rho)
    return out


def directional_consistency(X_a, shap_a, X_b, shap_b, feature_names,
                            top_k: int = 10, rho_min: float = DEFAULT_RHO_MIN) -> dict:
    """Do two attribution sets agree on the DIRECTION of their important features?

    For each feature, Spearman rho between its values and its SHAP values is
    computed within each set. Directional agreement holds when the two rhos share
    a sign AND both satisfy |rho| >= rho_min.

    The percentage is taken over the top_k features by mean |SHAP| (averaged
    across the two sets, so the selection cannot depend on argument order). The
    count of those features that actually meet rho_min is reported alongside it,
    because a high percentage over few qualifying features means something very
    different from the same percentage over all of them.
    """
    Sa, Sb = np.asarray(shap_a, dtype=float), np.asarray(shap_b, dtype=float)
    names = list(feature_names)
    if Sa.shape[1] != len(names) or Sb.shape[1] != len(names):
        raise ValueError(f"SHAP arrays have {Sa.shape[1]}/{Sb.shape[1]} columns vs "
                         f"{len(names)} feature names")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    rho_a = feature_directions(X_a, Sa, names)
    rho_b = feature_directions(X_b, Sb, names)

    imp_a, imp_b = np.abs(Sa).mean(axis=0), np.abs(Sb).mean(axis=0)
    combined = 0.5 * (imp_a + imp_b)
    k = min(top_k, len(names))
    idx = np.argsort(-combined, kind="stable")[:k]

    strong = (np.abs(rho_a[idx]) >= rho_min) & (np.abs(rho_b[idx]) >= rho_min)
    same_sign = np.sign(rho_a[idx]) == np.sign(rho_b[idx])
    agree = strong & same_sign

    pairs = [{"feature": names[j], "rho_a": float(rho_a[j]), "rho_b": float(rho_b[j]),
              "mean_abs_shap": float(combined[j]),
              "meets_rho_min": bool((abs(rho_a[j]) >= rho_min) and (abs(rho_b[j]) >= rho_min)),
              "agrees": bool((abs(rho_a[j]) >= rho_min) and (abs(rho_b[j]) >= rho_min)
                             and np.sign(rho_a[j]) == np.sign(rho_b[j]))}
             for j in idx]

    return {
        "agreement_percent": float(100.0 * agree.sum() / k),
        "n_considered": int(k),
        "n_meeting_rho_min": int(strong.sum()),
        "n_agreeing": int(agree.sum()),
        "top_k": int(top_k), "rho_min": float(rho_min),
        "rho_pairs": pairs,
        "min_abs_rho_considered": float(np.minimum(np.abs(rho_a[idx]),
                                                   np.abs(rho_b[idx])).min()) if k else float("nan"),
    }


def directional_consistency_profile(X_a, shap_a, X_b, shap_b, feature_names,
                                    ks: Sequence[int] = tuple(JACCARD_KS),
                                    rho_mins: Sequence[float] = DIRECTIONAL_RHO_MINS) -> dict:
    """Every (top_k, rho_min) combination, so threshold sensitivity is visible."""
    out: dict[str, float] = {}
    for k in ks:
        for rm in rho_mins:
            res = directional_consistency(X_a, shap_a, X_b, shap_b, feature_names,
                                          top_k=k, rho_min=rm)
            tag = f"top{k}_rho{str(rm).replace('.', '')}"
            out[f"directional_{tag}"] = res["agreement_percent"]
            out[f"n_meeting_rho_min_{tag}"] = res["n_meeting_rho_min"]
    return out


# --------------------------------------------------------------------------- #
# 5. Bootstrap confidence interval
# --------------------------------------------------------------------------- #
def bootstrap_ci(metric_fn: Callable, data, n: int = N_BOOTSTRAP, seed: int = 42,
                 ci: float = 95.0) -> dict:
    """Percentile bootstrap CI for any metric above.

    `data` is either
      * a 1-D sample (e.g. per-seed tau values) -> metric_fn(resample), or
      * a tuple/list of two paired vectors      -> metric_fn(a[idx], b[idx]),
        resampled with a SHARED index so the pairing between the two importance
        vectors is preserved (resampling them independently would destroy exactly
        the correspondence the metric measures).
    """
    rng = np.random.default_rng(seed)

    if isinstance(data, (tuple, list)) and len(data) == 2 and np.ndim(data[0]) >= 1 \
            and np.size(data[0]) == np.size(data[1]) and np.size(data[0]) > 1:
        a, b = _as_array(data[0]), _as_array(data[1])
        n_items = a.size
        stats = []
        for _ in range(n):
            idx = rng.integers(0, n_items, n_items)
            val = metric_fn(a[idx], b[idx])
            stats.append(val[0] if isinstance(val, tuple) else val)
        point = metric_fn(a, b)
        point = point[0] if isinstance(point, tuple) else point
    else:
        sample = _as_array(data)
        n_items = sample.size
        stats = []
        for _ in range(n):
            idx = rng.integers(0, n_items, n_items)
            val = metric_fn(sample[idx])
            stats.append(val[0] if isinstance(val, tuple) else val)
        point = metric_fn(sample)
        point = point[0] if isinstance(point, tuple) else point

    stats = np.asarray(stats, dtype=float)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        raise ValueError("all bootstrap replicates were non-finite")
    lo = float(np.percentile(stats, (100 - ci) / 2))
    hi = float(np.percentile(stats, 100 - (100 - ci) / 2))
    return {
        "point": float(point),
        "ci_lower": lo,
        "ci_upper": hi,
        "bootstrap_mean": float(stats.mean()),
        "bootstrap_std": float(stats.std(ddof=1)) if stats.size > 1 else 0.0,
        "n_bootstrap": int(stats.size),
        "ci_level": ci,
    }


# --------------------------------------------------------------------------- #
# Convenience: every metric between two importance profiles
# --------------------------------------------------------------------------- #
def all_metrics(importance_a, importance_b, signed_a=None, signed_b=None,
                seed: int = 42, n_perm: int = N_PERMUTATIONS) -> dict:
    """Compute tau (+ p-values), Jaccard@{5,10,15} and every sign-agreement variant.

    `importance_a`/`importance_b` are the mean |SHAP| profiles and double as the
    importance weights for the sign-agreement selection.
    """
    tau, p_analytic = kendall_tau(importance_a, importance_b)
    out = {"kendall_tau": tau, "tau_pvalue": p_analytic}
    perm = permutation_null_tau(importance_a, importance_b, n_perm=n_perm, seed=seed)
    out["tau_pvalue_permutation"] = perm["p_empirical"]
    out["tau_null_mean"] = perm["null_mean"]
    out["tau_null_std"] = perm["null_std"]
    out.update(jaccard_profile(importance_a, importance_b))
    if signed_a is not None and signed_b is not None:
        out.update(sign_agreement_profile(signed_a, signed_b, importance_a, importance_b))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.random(30)
    imp = np.abs(a)
    log.info("identical  : tau=%.3f jac10=%.3f sign_top10=%.1f%% sign_unfiltered=%.1f%%",
             kendall_tau(a, a)[0], jaccard_at_k(a, a, 10),
             sign_agreement(a, a, imp, imp).percent, sign_agreement_unfiltered(a, a))
    log.info("reversed   : tau=%.3f", kendall_tau(a, -a)[0])
    log.info("sign-flip  : sign_top10=%.1f%%",
             sign_agreement(a, -a, imp, imp).percent)
    log.info("random     : tau=%.3f", kendall_tau(a, rng.random(30))[0])
