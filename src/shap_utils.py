"""Task 7 - TreeSHAP utilities.

Two methodological points that matter for the paper:

1. `model_output='probability'` forces `feature_perturbation='interventional'`,
   which in turn requires a background dataset. Attributions are therefore in
   probability space (units: change in P(phishing)), which is what makes the
   signed values directly interpretable. The cost is that runtime scales with the
   background size, so the background is a small stratified sample of the
   TRAINING split - never of the data being explained.

2. The soft-voting ensemble's SHAP values are the equal-weight mean of the three
   base learners' SHAP values. This is exact rather than an approximation:
   soft voting averages the base probabilities, Shapley values are linear in the
   model output, and every base learner is explained in the same probability
   space against the same background. Averaging is implemented explicitly here
   because no library shortcut is trusted to preserve that equivalence.

check_additivity is always True. If it ever trips, the fix is to investigate the
model or the background, never to silence the check.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shap  # noqa: E402

from config import (  # noqa: E402
    RESULTS_DIR,
    SHAP_CACHE_DIR,
    SHAP_CONVERGENCE_SIZES,
    SHAP_CONVERGENCE_TAU_THRESHOLD,
    SHAP_SAMPLE_SIZE,
    banner,
    get_logger,
)
from src.consistency import kendall_tau  # noqa: E402
from src.models import ENSEMBLE_NAME, base_estimators_of  # noqa: E402

log = get_logger("shap_utils")

# Background size for interventional TreeSHAP. Runtime is roughly linear in this,
# and SHAP's own guidance is that 100 well-chosen rows are plenty; larger
# backgrounds change attributions only marginally while multiplying cost.
BACKGROUND_SIZE = 100


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def stratified_sample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of a class-stratified sample of size <= n, deterministic in `seed`.

    Class proportions are preserved. When n exceeds the available rows the whole
    set is returned (shuffled), which is the normal case for the small dataset.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if n >= len(y):
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return idx

    picked = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        cls_idx = np.flatnonzero(y == cls)
        take = int(round(n * cnt / len(y)))
        take = min(max(take, 1), len(cls_idx))
        picked.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(picked)
    rng.shuffle(idx)
    return idx[:n]


def select_shap_sample(X: np.ndarray, y: np.ndarray, seed: int,
                       n: int = SHAP_SAMPLE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    idx = stratified_sample_indices(y, n, seed)
    return X[idx], idx


def make_background(X_train: np.ndarray, y_train: np.ndarray, seed: int,
                    size: int = BACKGROUND_SIZE) -> np.ndarray:
    """Stratified background sample drawn from the TRAINING split only."""
    idx = stratified_sample_indices(y_train, size, seed)
    return np.ascontiguousarray(X_train[idx], dtype=float)


# --------------------------------------------------------------------------- #
# Core SHAP computation
# --------------------------------------------------------------------------- #
def _normalise_shap_output(values, n_samples: int, n_features: int) -> np.ndarray:
    """Reduce SHAP's various shapes to (n_samples, n_features) for the POSITIVE class.

    Across model types and SHAP versions the output is either
      * a list of per-class arrays,
      * (n, f, n_classes), or
      * (n, f) already (binary models that expose a single output).
    """
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]          # positive class = phishing
        elif arr.shape[0] == 2:
            arr = arr[1]
        else:
            raise ValueError(f"unexpected 3-D SHAP shape {arr.shape}")
    if arr.shape != (n_samples, n_features):
        raise ValueError(f"SHAP array shape {arr.shape} != expected "
                         f"{(n_samples, n_features)}")
    return arr


def _cache_path(cache_key: str, X: np.ndarray) -> Path:
    """Cache file keyed by caller-supplied tag plus a hash of the explained data."""
    h = hashlib.sha1(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()[:12]
    return SHAP_CACHE_DIR / f"{cache_key}__n{X.shape[0]}_{h}.npy"


def compute_shap_values(model, X: np.ndarray, seed: int, background: np.ndarray | None = None,
                        cache_key: str | None = None, check_additivity: bool = True,
                        use_cache: bool = True) -> np.ndarray:
    """Raw SHAP array (n_samples, n_features) in probability space.

    Dispatches automatically: a fitted VotingClassifier is explained per base
    learner and averaged; anything else goes straight to TreeExplainer.
    """
    X = np.ascontiguousarray(X, dtype=float)

    if cache_key and use_cache:
        path = _cache_path(cache_key, X)
        if path.exists():
            arr = np.load(path)
            log.info("SHAP cache hit %s -> %s", path.name, arr.shape)
            return arr

    if hasattr(model, "estimators_") and hasattr(model, "voting"):
        values = _ensemble_shap_values(model, X, seed, background, check_additivity)
    else:
        values = _single_model_shap_values(model, X, seed, background, check_additivity)

    if cache_key and use_cache:
        _save_atomic(_cache_path(cache_key, X), values)
    return values


def _save_atomic(path: Path, values: np.ndarray) -> None:
    """Write the cache entry via a temp file and an atomic rename.

    np.save is not atomic, so a power cut or a hard kill part-way through would
    leave a truncated .npy on disk. The cache-hit path does not validate what it
    loads, so that truncated file would resurface as a confusing failure on the
    next run rather than as a clean recompute. Renaming into place means the
    cache only ever contains complete arrays: an interrupted run loses the
    in-flight computation and nothing else.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        # Write to an open handle, NOT a path: given a path that does not end in
        # .npy, np.save silently appends the extension, so passing `tmp` directly
        # would create "<name>.npy.tmp.npy" and leave the rename with nothing to
        # move. Handing it a file object bypasses that entirely.
        with open(tmp, "wb") as fh:
            np.save(fh, values, allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _single_model_shap_values(model, X: np.ndarray, seed: int,
                              background: np.ndarray | None,
                              check_additivity: bool) -> np.ndarray:
    if background is None:
        raise ValueError(
            "model_output='probability' requires interventional feature perturbation, "
            "which requires a background dataset. Pass background=make_background(...)."
        )
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(
        model,
        data=background,
        model_output="probability",
        feature_perturbation="interventional",
    )
    values = explainer.shap_values(X, check_additivity=check_additivity)
    arr = _normalise_shap_output(values, X.shape[0], X.shape[1])
    log.info("    TreeSHAP %-22s n=%d bg=%d -> %.1fs",
             model.__class__.__name__, X.shape[0], len(background), time.perf_counter() - t0)
    return arr


def _ensemble_shap_values(ensemble, X: np.ndarray, seed: int,
                          background: np.ndarray | None,
                          check_additivity: bool) -> np.ndarray:
    """Equal-weight mean of the base learners' probability-space SHAP values."""
    members = base_estimators_of(ensemble)
    log.info("    ensemble SHAP: averaging %d base learners %s",
             len(members), list(members))
    per_model = []
    for name, est in members.items():
        per_model.append(_single_model_shap_values(est, X, seed, background, check_additivity))
    stacked = np.stack(per_model, axis=0)
    weights = np.full(len(per_model), 1.0 / len(per_model))  # soft voting = equal weights
    return np.tensordot(weights, stacked, axes=(0, 0))


# --------------------------------------------------------------------------- #
# Importance profiles
# --------------------------------------------------------------------------- #
def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean |SHAP| per feature - the magnitude profile used for tau and Jaccard."""
    arr = np.asarray(shap_values)
    if arr.shape[1] != len(feature_names):
        raise ValueError(f"{arr.shape[1]} SHAP columns vs {len(feature_names)} feature names")
    return pd.Series(np.abs(arr).mean(axis=0), index=feature_names, name="mean_abs_shap")


def signed_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean signed SHAP per feature - the direction profile used for sign agreement.

    WARNING - read `directional_importance` before using this as a direction
    statistic. For a feature that pushes towards phishing at some values and away
    at others, the positive and negative contributions cancel and the mean lands
    near zero however important the feature is. Measured on UCI, the top-10
    features have |mean signed| / mean |SHAP| of only 0.02-0.08, so their SIGN is
    numerical noise even though their MAGNITUDE is large. See
    results/sign_agreement_diagnostic_v2.json.
    """
    arr = np.asarray(shap_values)
    if arr.shape[1] != len(feature_names):
        raise ValueError(f"{arr.shape[1]} SHAP columns vs {len(feature_names)} feature names")
    return pd.Series(arr.mean(axis=0), index=feature_names, name="mean_signed_shap")


def directional_importance(shap_values: np.ndarray, X: np.ndarray,
                           feature_names: list[str]) -> pd.Series:
    """Direction of each feature's effect, as Spearman corr(feature value, SHAP value).

    This is the direction statistic that mean signed SHAP fails to be. It answers
    "when this feature's value goes up, does it push the prediction towards
    phishing?" - a question whose answer survives a feature contributing in both
    directions at different values, because it correlates the contribution with
    the value rather than averaging the contribution away.

    Spearman rather than Pearson: the effect need only be monotone, not linear,
    and the UCI features are ternary-coded. Rank correlation is invariant under
    the StandardScaler applied upstream, so scaled and unscaled inputs agree.

    A feature that is constant across the explained sample, or that the model
    never used, has no direction; it returns 0.0 rather than NaN so that it is
    counted as a non-match rather than silently dropped.
    """
    from scipy.stats import spearmanr

    arr = np.asarray(shap_values)
    Xv = np.asarray(X, dtype=float)
    if arr.shape != Xv.shape:
        raise ValueError(f"SHAP array {arr.shape} does not match feature matrix {Xv.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(f"{arr.shape[1]} SHAP columns vs {len(feature_names)} feature names")

    out = np.zeros(arr.shape[1], dtype=float)
    for j in range(arr.shape[1]):
        xj, sj = Xv[:, j], arr[:, j]
        if np.ptp(xj) == 0 or np.ptp(sj) == 0:
            continue
        rho = spearmanr(xj, sj).statistic
        out[j] = 0.0 if not np.isfinite(rho) else float(rho)
    return pd.Series(out, index=feature_names, name="direction_spearman")


# --------------------------------------------------------------------------- #
# Convergence check
# --------------------------------------------------------------------------- #
def convergence_check(model, X_pool: np.ndarray, y_pool: np.ndarray, feature_names: list[str],
                      seed: int, background: np.ndarray, dataset: str, model_name: str,
                      sizes: list[int] | None = None) -> pd.DataFrame:
    """Is SHAP_SAMPLE_SIZE large enough for the importance RANKING to be stable?

    Recomputes mean |SHAP| at each sample size and reports Kendall's tau between
    consecutive rankings. The decisive comparison is 10k vs 20k: if that tau is
    below the threshold, the 10k sample used everywhere else is too small and the
    consistency numbers would be partly sampling noise.
    """
    sizes = list(sizes or SHAP_CONVERGENCE_SIZES)
    banner(log, f"SHAP CONVERGENCE CHECK: {dataset}/{model_name}", "-")

    pool_n = len(y_pool)
    # When the pool cannot reach the requested sizes, two nominal sizes collapse
    # onto the same sample and their tau is 1.0 by construction - which is not
    # evidence of convergence. Those rows are flagged, and an adaptive ladder
    # over the available pool is added so a real statement is still possible.
    if pool_n < max(sizes):
        ladder = sorted({max(int(pool_n * f), 50) for f in (0.25, 0.5, 1.0)})
        log.warning("[%s] explanation pool holds only %d instances, so the requested "
                    "ladder %s cannot be run in full. Adding an adaptive ladder %s.",
                    dataset, pool_n, sizes, ladder)
        sizes = sorted(set(sizes) | set(ladder))

    rankings: dict[int, pd.Series] = {}
    effective_of: dict[int, int] = {}
    computed: dict[int, pd.Series] = {}   # keyed by EFFECTIVE size, to avoid recompute
    for n in sizes:
        effective = min(n, pool_n)
        effective_of[n] = effective
        if effective in computed:
            log.info("[%s] nominal n=%d clamps to %d, already computed - reused",
                     dataset, n, effective)
            rankings[n] = computed[effective]
            continue
        Xs, _ = select_shap_sample(X_pool, y_pool, seed=seed, n=effective)
        values = compute_shap_values(
            model, Xs, seed, background=background,
            cache_key=f"conv_{dataset}_{model_name}_seed{seed}")
        imp = global_importance(values, feature_names)
        computed[effective] = imp
        rankings[n] = imp
        log.info("[%s] n=%-6d (effective %d) top-5: %s", dataset, n, effective,
                 ", ".join(imp.nlargest(5).index))

    rows = []
    ordered = sorted(rankings)
    for lo, hi in zip(ordered, ordered[1:]):
        a = rankings[lo]
        b = rankings[hi].reindex(a.index)
        tau, p = kendall_tau(a.to_numpy(), b.to_numpy())
        degenerate = effective_of[lo] == effective_of[hi]
        rows.append({
            "dataset": dataset, "model": model_name, "seed": seed,
            "size_a": lo, "size_b": hi,
            "effective_a": effective_of[lo], "effective_b": effective_of[hi],
            "kendall_tau": tau, "p_value": p,
            "n_available": int(pool_n),
            "degenerate": bool(degenerate),
        })
        log.info("[%s] tau(%d vs %d) = %.4f (p=%.3g)%s", dataset, lo, hi, tau, p,
                 "  [DEGENERATE: both clamp to the same sample]" if degenerate else "")

    df = pd.DataFrame(rows)
    decisive = df[(df["size_a"] == 10000) & (df["size_b"] == 20000)]
    if len(decisive):
        tau = float(decisive["kendall_tau"].iloc[0])
        if bool(decisive["degenerate"].iloc[0]):
            log.warning("NOT EVALUABLE for %s/%s: the 10k and 20k points both clamp to "
                        "the %d available instances, so tau = %.4f is 1.0 by construction "
                        "and says nothing about convergence. Use the adaptive ladder rows "
                        "instead.", dataset, model_name, pool_n, tau)
        elif tau < SHAP_CONVERGENCE_TAU_THRESHOLD:
            log.warning("*" * 78)
            log.warning("SHAP SAMPLE TOO SMALL: tau(10k vs 20k) = %.4f < %.2f for %s/%s.",
                        tau, SHAP_CONVERGENCE_TAU_THRESHOLD, dataset, model_name)
            log.warning("The %d-instance sample used elsewhere has not converged; part of "
                        "any measured inconsistency is sampling noise.",
                        SHAP_SAMPLE_SIZE)
            log.warning("*" * 78)
        else:
            log.info("CONVERGED: tau(10k vs 20k) = %.4f >= %.2f - the %d-instance sample "
                     "is adequate for %s/%s.", tau, SHAP_CONVERGENCE_TAU_THRESHOLD,
                     SHAP_SAMPLE_SIZE, dataset, model_name)
    return df


def run_convergence_checks(models_by_dataset: dict, sizes: list[int] | None = None) -> pd.DataFrame:
    """Append convergence results for every supplied (dataset -> fitted model) pair."""
    frames = []
    for dataset, payload in models_by_dataset.items():
        frames.append(convergence_check(
            model=payload["model"], X_pool=payload["X_pool"], y_pool=payload["y_pool"],
            feature_names=payload["feature_names"], seed=payload["seed"],
            background=payload["background"], dataset=dataset,
            model_name=payload.get("model_name", ENSEMBLE_NAME), sizes=sizes))
    df = pd.concat(frames, ignore_index=True)
    out = RESULTS_DIR / "shap_convergence.csv"
    df.to_csv(out, index=False)
    log.info("convergence results -> %s", out)
    return df


if __name__ == "__main__":
    import argparse

    from src.evaluate import train_and_evaluate
    from src.preprocessing import get_splits
    from src.tuning import get_params_for_models

    ap = argparse.ArgumentParser(description="Task 7: SHAP convergence check")
    ap.add_argument("--dataset", default="uci", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
    payloads = {}
    for ds in datasets:
        splits = get_splits(ds, args.seed)
        _, _, fitted = train_and_evaluate(ds, args.seed, get_params_for_models(ds),
                                          keep_models=True)
        payloads[ds] = {
            "model": fitted[ENSEMBLE_NAME],
            "X_pool": splits.X_test, "y_pool": splits.y_test,
            "feature_names": splits.feature_names, "seed": args.seed,
            "background": make_background(splits.X_train, splits.y_train, args.seed),
            "model_name": ENSEMBLE_NAME,
        }
    run_convergence_checks(payloads)
