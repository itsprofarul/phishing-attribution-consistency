"""Task B4 - faithfulness and robustness diagnostics.

B4.1  Sufficiency  - retrain on the top-k SHAP features only; how much performance
                     is RETAINED? On a leaking dataset this is near-total at k=3,
                     which is the leak restated in faithfulness terms.
      Comprehensiveness - retrain with the top-k REMOVED; how much is LOST?

B4.2  Spurious-feature injection - add one controlled artificial feature and see
      where SHAP ranks it. The weak-label-proxy variant at increasing strength is
      the important one: it builds a controlled analogue of what PhiUSIIL contains
      naturally, so injected shortcuts can be related to the naturally occurring
      kind.

B4.3  SHAP vs permutation importance - Spearman rho between the two rankings,
      computed over ALL features rather than a top-k subset (see the note in
      `shap_permutation_agreement` for why the broader scope matters).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import spearmanr  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score  # noqa: E402

from config import (  # noqa: E402
    RESULTS_DIR,
    SEEDS,
    banner,
    get_logger,
    set_global_seed,
    write_csv_atomic,
)
from src.models import ENSEMBLE_NAME, build_models  # noqa: E402
from src.preprocessing import get_splits  # noqa: E402
from src.shap_utils import (  # noqa: E402
    compute_shap_values,
    global_importance,
    make_background,
    select_shap_sample,
)
from src.tuning import get_params_for_models  # noqa: E402

log = get_logger("faithfulness")

FAITH_KS = [3, 5, 10]
INJECTION_STRATEGIES = ["permuted_copy", "pure_noise", "weak_proxy"]
PROXY_STRENGTHS = [0.05, 0.25, 0.50]
# Smaller than SHAP_SAMPLE_SIZE: B4 retrains the ensemble many times (3 k-values x
# 2 directions x 5 seeds x N datasets), and the ranking it needs is the arg-sort of
# mean |SHAP|, which is stable well below the full sample. Recorded as a judgement
# call rather than left implicit.
FAITH_SHAP_SAMPLE = 2000

# B4.2 seed policy - DELIBERATELY ASYMMETRIC, and recorded in the output so the
# asymmetry is documented rather than looking arbitrary.
#
# PhiUSIIL runs 3 seeds, UCI 5. The reason is what B4.2 delivers: a THRESHOLD
# ("at what proxy strength does the injected feature reach rank 1?"), not a
# variance estimate. Seeds 42 and 43 already agreed closely on PhiUSIIL
# (injected rank 41/40/23 at strengths 0.05/0.25/0.50), so the threshold is
# settled. Each additional PhiUSIIL seed costs ~2h because B4.2's SHAP cannot be
# cached - every injection is a unique feature set - and the host service has
# twice killed multi-hour runs mid-flight.
#
# This does NOT apply to E2, where across-seed variance IS the deliverable and
# the 5-seed protocol is a stated methodological advantage over the 3 retrain
# runs in Maseno et al. E2 ran all 5 seeds on every config.
B42_SEEDS = {"phiusiil": 3, "uci": 5}
B42_SEED_POLICY_REASON = (
    "B4.2 reports a threshold, not a variance estimate; PhiUSIIL seeds 42-43 "
    "already agreed on the injected-feature rank, and each further PhiUSIIL seed "
    "costs ~2h of uncacheable SHAP. E2, where across-seed variance is the "
    "deliverable, ran all 5 seeds on every config."
)



def _completed(path: Path) -> set:
    """(dataset, seed) pairs already present in an output file.

    B4 has been killed mid-run more than once by host teardown, and B4.2's SHAP is
    deliberately uncached - each injection is a unique feature set, so there is
    nothing to cache - which makes a restart from zero expensive. Resuming from
    what is already on disk makes an interruption cost at most one seed.
    """
    if not path.exists():
        return set()
    try:
        d = pd.read_csv(path)
        return {(str(r["dataset"]), int(r["seed"])) for _, r in d.iterrows()}
    except Exception:
        return set()


def _existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def _score(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "roc_auc": float(roc_auc_score(y, proba))}


def _fit_ensemble(X_tr, y_tr, seed, params):
    ens = build_models(seed, params)[ENSEMBLE_NAME]
    ens.fit(X_tr, y_tr)
    return ens


def _shap_ranking(model, X_tr, y_tr, X_pool, y_pool, names, seed, cache_key=None):
    bg = make_background(X_tr, y_tr, seed)
    X_exp, _ = select_shap_sample(X_pool, y_pool, seed=seed, n=FAITH_SHAP_SAMPLE)
    vals = compute_shap_values(model, X_exp, seed, background=bg, cache_key=cache_key)
    return global_importance(vals, names).sort_values(ascending=False), vals, X_exp


# --------------------------------------------------------------------------- #
# B4.1 Sufficiency and comprehensiveness
# --------------------------------------------------------------------------- #
def faithfulness_for_seed(dataset: str, seed: int, params: dict) -> list[dict]:
    set_global_seed(seed)
    splits = get_splits(dataset, seed)
    names = splits.feature_names

    full = _fit_ensemble(splits.X_train, splits.y_train, seed, params)
    base = _score(full, splits.X_test, splits.y_test)
    ranking, _, _ = _shap_ranking(full, splits.X_train, splits.y_train,
                                  splits.X_test, splits.y_test, names, seed,
                                  cache_key=f"faith_{dataset}_full_seed{seed}")

    rows = []
    for k in FAITH_KS:
        top = list(ranking.index[:k])
        idx_top = [names.index(f) for f in top]
        idx_rest = [i for i in range(len(names)) if i not in idx_top]

        suff_model = _fit_ensemble(splits.X_train[:, idx_top], splits.y_train, seed, params)
        suff = _score(suff_model, splits.X_test[:, idx_top], splits.y_test)

        comp_model = _fit_ensemble(splits.X_train[:, idx_rest], splits.y_train, seed, params)
        comp = _score(comp_model, splits.X_test[:, idx_rest], splits.y_test)

        rows.append({
            "dataset": dataset, "model": ENSEMBLE_NAME, "seed": seed, "k": k,
            "top_k_features": "; ".join(top),
            "full_accuracy": base["accuracy"], "full_f1": base["f1"],
            "full_mcc": base["mcc"], "full_roc_auc": base["roc_auc"],
            "sufficiency_accuracy": suff["accuracy"], "sufficiency_f1": suff["f1"],
            "sufficiency_mcc": suff["mcc"], "sufficiency_roc_auc": suff["roc_auc"],
            "comprehensiveness_accuracy": comp["accuracy"], "comprehensiveness_f1": comp["f1"],
            "comprehensiveness_mcc": comp["mcc"],
            # Retained: what fraction of full performance the top-k alone preserves.
            "performance_retained": suff["accuracy"] / base["accuracy"],
            # Lost: how far performance falls when the top-k are taken away.
            "performance_lost": base["accuracy"] - comp["accuracy"],
        })
        log.info("[%s seed=%d k=%2d] sufficiency acc=%.4f (%.1f%% of full %.4f) | "
                 "comprehensiveness acc=%.4f (lost %.4f) | top-%d: %s",
                 dataset, seed, k, suff["accuracy"],
                 100 * rows[-1]["performance_retained"], base["accuracy"],
                 comp["accuracy"], rows[-1]["performance_lost"], k, ", ".join(top[:3]))
    return rows


# --------------------------------------------------------------------------- #
# B4.2 Spurious-feature injection
# --------------------------------------------------------------------------- #
def _make_spurious(strategy: str, X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                   strength: float = 0.0, donor: int = 0) -> np.ndarray:
    """One artificial column, built by the named strategy."""
    n = len(y)
    if strategy == "permuted_copy":
        # Same marginal distribution as a real feature, but its link to the label
        # is destroyed - so any importance it attracts is pure artefact.
        return rng.permutation(X[:, donor].copy())
    if strategy == "pure_noise":
        return rng.normal(size=n)
    if strategy == "weak_proxy":
        # With probability `strength` the feature reveals the label; otherwise it
        # is a coin flip. Expected correlation with the label scales with strength,
        # giving a dial from "harmless" to "obvious shortcut".
        reveal = rng.random(n) < strength
        noise = rng.integers(0, 2, size=n)
        return np.where(reveal, y, noise).astype(float)
    raise ValueError(f"unknown strategy {strategy!r}")


def injection_for_seed(dataset: str, seed: int, params: dict) -> list[dict]:
    set_global_seed(seed)
    splits = get_splits(dataset, seed)
    names = splits.feature_names
    rng = np.random.default_rng(seed)

    full = _fit_ensemble(splits.X_train, splits.y_train, seed, params)
    base_rank, _, _ = _shap_ranking(full, splits.X_train, splits.y_train,
                                    splits.X_test, splits.y_test, names, seed,
                                    cache_key=f"faith_{dataset}_full_seed{seed}")

    rows = []
    combos = [(s, None) for s in ("permuted_copy", "pure_noise")] + \
             [("weak_proxy", st) for st in PROXY_STRENGTHS]
    for strategy, strength in combos:
        col_tr = _make_spurious(strategy, splits.X_train, splits.y_train, rng,
                                strength or 0.0)
        col_te = _make_spurious(strategy, splits.X_test, splits.y_test, rng,
                                strength or 0.0)
        Xtr = np.column_stack([splits.X_train, col_tr])
        Xte = np.column_stack([splits.X_test, col_te])
        inj_names = names + ["__injected__"]

        model = _fit_ensemble(Xtr, splits.y_train, seed, params)
        rank, _, _ = _shap_ranking(model, Xtr, splits.y_train, Xte, splits.y_test,
                                   inj_names, seed, cache_key=None)
        order = list(rank.index)
        inj_rank = order.index("__injected__") + 1

        # Did adding the artificial feature reorder the GENUINE ones? Compared on
        # the shared feature set only, so the injected column cannot itself move
        # the correlation.
        shared = [f for f in base_rank.index if f in rank.index]
        rho = float(spearmanr(base_rank[shared].to_numpy(),
                              rank[shared].to_numpy()).statistic)

        rows.append({
            "dataset": dataset, "seed": seed, "strategy": strategy,
            "strength": strength if strength is not None else "",
            "injected_shap_rank": int(inj_rank),
            "injected_reached_rank_1": bool(inj_rank == 1),
            "n_features_after_injection": len(inj_names),
            "genuine_feature_rank_correlation": rho,
            "injected_mean_abs_shap": float(rank["__injected__"]),
            "top_feature_after_injection": order[0],
        })
        log.info("[%s seed=%d] %-13s strength=%-5s -> injected SHAP rank %3d/%d "
                 "(genuine-feature rank rho=%.4f)", dataset, seed, strategy,
                 strength if strength is not None else "-", inj_rank, len(inj_names), rho)
    return rows


# --------------------------------------------------------------------------- #
# B4.3 SHAP vs permutation importance
# --------------------------------------------------------------------------- #
PERM_SCORING = "neg_log_loss"


def shap_permutation_agreement(dataset: str, seed: int, params: dict,
                               n_repeats: int = 5,
                               scoring: str = PERM_SCORING) -> dict:
    """Spearman rho between the mean |SHAP| ranking and permutation importance.

    Computed over ALL features, not the top-10 SHAP-ranked subset used in the
    cited prior work. Restricting permutation importance to features SHAP already
    ranked highly couples the two measures: the comparison is then conditioned on
    one of them, and a high correlation is partly built in. Scoring every feature
    keeps the two rankings independent, at the cost of more compute.

    Scored by LOG-LOSS, not F1 or ROC-AUC. On a benchmark whose model sits at
    ceiling accuracy with redundant features, a threshold- or rank-based score
    cannot register the removal of any single feature - the others compensate
    exactly - so the importance vector collapses to ties at zero and the rank
    correlation becomes meaningless. A proper scoring rule still responds to the
    change in predicted probability. `n_zero_importance` records how far from that
    failure mode each run actually was.
    """
    set_global_seed(seed)
    splits = get_splits(dataset, seed)
    names = splits.feature_names

    model = _fit_ensemble(splits.X_train, splits.y_train, seed, params)
    shap_rank, _, _ = _shap_ranking(model, splits.X_train, splits.y_train,
                                    splits.X_test, splits.y_test, names, seed,
                                    cache_key=f"faith_{dataset}_full_seed{seed}")

    t0 = time.perf_counter()
    perm = permutation_importance(model, splits.X_test, splits.y_test,
                                  n_repeats=n_repeats, random_state=seed,
                                  scoring=scoring, n_jobs=1)
    wall = time.perf_counter() - t0

    perm_s = pd.Series(perm.importances_mean, index=names)
    shared = list(shap_rank.index)
    rho = float(spearmanr(shap_rank[shared].to_numpy(), perm_s[shared].to_numpy()).statistic)

    top_shap = list(shap_rank.index[:10])
    top_perm = list(perm_s.sort_values(ascending=False).index[:10])
    overlap = len(set(top_shap) & set(top_perm))

    # Saturation check: a vector of ties cannot support a rank correlation.
    vals = perm.importances_mean
    n_zero = int(np.sum(np.abs(vals) < 1e-12))
    n_distinct = int(len(np.unique(np.round(vals, 12))))
    degenerate = n_zero > len(vals) // 2

    log.info("[%s seed=%d] SHAP vs permutation (%s): rho=%.4f over ALL %d features "
             "(top-10 overlap %d/10, %d zero / %d distinct, %.1fs)",
             dataset, seed, scoring, rho, len(shared), overlap, n_zero, n_distinct, wall)
    if degenerate:
        log.warning("[%s seed=%d] DEGENERATE: %d of %d permutation importances are "
                    "exactly zero, so rho is determined by tie structure rather than by "
                    "the data and must not be reported as a measurement.",
                    dataset, seed, n_zero, len(vals))
    return {"dataset": dataset, "seed": seed, "spearman_rho": rho,
            "n_features_scored": len(shared), "top10_overlap": overlap,
            "permutation_seconds": round(wall, 2), "n_repeats": n_repeats,
            "scoring": scoring,
            "n_zero_importance": n_zero, "n_distinct_importance": n_distinct,
            "degenerate": degenerate,
            "scope": "all_features",
            "scope_note": "prior work reports this over the top-10 SHAP subset; scoring "
                          "all features avoids conditioning one ranking on the other",
            "top10_shap": "; ".join(top_shap), "top10_permutation": "; ".join(top_perm)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_faithfulness(datasets: list[str], seeds: list[int] | None = None,
                     stages: tuple[str, ...] = ("b41", "b42", "b43")) -> dict:
    seeds = seeds or SEEDS
    out: dict[str, pd.DataFrame] = {}

    if "b41" in stages:
        banner(log, "B4.1 SUFFICIENCY AND COMPREHENSIVENESS")
        out_path = RESULTS_DIR / "faithfulness.csv"
        rows = _existing_rows(out_path)
        done = _completed(out_path)
        for ds in datasets:
            params = get_params_for_models(ds)
            for seed in seeds:
                if (ds, seed) in done:
                    log.info("[B4.1] %s seed=%d already on disk - skipped", ds, seed)
                    continue
                rows.extend(faithfulness_for_seed(ds, seed, params))
                write_csv_atomic(pd.DataFrame(rows), RESULTS_DIR / "faithfulness.csv")
        out["faithfulness"] = pd.DataFrame(rows)
        _summarise_faithfulness(out["faithfulness"])

    if "b42" in stages:
        banner(log, "B4.2 SPURIOUS-FEATURE INJECTION")
        out_path = RESULTS_DIR / "spurious_injection.csv"
        rows = _existing_rows(out_path)
        done = _completed(out_path)
        for ds in datasets:
            params = get_params_for_models(ds)
            ds_seeds = seeds[:B42_SEEDS.get(ds, len(seeds))]
            if len(ds_seeds) < len(seeds):
                log.warning("[B4.2] %s runs %d of %d seeds by policy: %s", ds,
                            len(ds_seeds), len(seeds), B42_SEED_POLICY_REASON)
            for seed in ds_seeds:
                if (ds, seed) in done:
                    log.info("[B4.2] %s seed=%d already on disk - skipped", ds, seed)
                    continue
                new_rows = injection_for_seed(ds, seed, params)
                for r in new_rows:
                    r["n_seeds_for_dataset"] = len(ds_seeds)
                    r["seed_policy_reason"] = B42_SEED_POLICY_REASON
                rows.extend(new_rows)
                write_csv_atomic(pd.DataFrame(rows),
                                 RESULTS_DIR / "spurious_injection.csv")
        out["injection"] = pd.DataFrame(rows)
        _summarise_injection(out["injection"])

    if "b43" in stages:
        banner(log, "B4.3 SHAP vs PERMUTATION IMPORTANCE")
        out_path = RESULTS_DIR / "shap_permutation_agreement.csv"
        rows = _existing_rows(out_path)
        done = _completed(out_path)
        for ds in datasets:
            params = get_params_for_models(ds)
            for seed in seeds:
                if (ds, seed) in done:
                    log.info("[B4.3] %s seed=%d already on disk - skipped", ds, seed)
                    continue
                rows.append(shap_permutation_agreement(ds, seed, params))
                write_csv_atomic(pd.DataFrame(rows),
                                 RESULTS_DIR / "shap_permutation_agreement.csv")
        out["permutation"] = pd.DataFrame(rows)
    return out


def _summarise_faithfulness(df: pd.DataFrame) -> None:
    banner(log, "B4.1 SUMMARY (mean across seeds)", "-")
    g = df.groupby(["dataset", "k"])[["performance_retained", "performance_lost",
                                      "sufficiency_accuracy", "comprehensiveness_accuracy",
                                      "full_accuracy"]].mean().round(4)
    for line in g.to_string().splitlines():
        log.info(line)
    for ds, sub in df.groupby("dataset"):
        k3 = sub[sub["k"] == 3]
        if k3.empty:
            continue
        ret = float(k3["performance_retained"].mean())
        if ret > 0.99:
            log.warning("[%s] sufficiency at k=3 retains %.2f%% of full accuracy - three "
                        "features reproduce the whole model. That is the leak stated in "
                        "faithfulness terms.", ds, 100 * ret)


def _summarise_injection(df: pd.DataFrame) -> None:
    banner(log, "B4.2 SUMMARY", "-")
    g = (df.groupby(["dataset", "strategy", "strength"])
           [["injected_shap_rank", "genuine_feature_rank_correlation"]]
           .mean().round(4))
    for line in g.to_string().splitlines():
        log.info(line)
    for ds, sub in df.groupby("dataset"):
        proxy = sub[sub["strategy"] == "weak_proxy"]
        if proxy.empty:
            continue
        reached = proxy[proxy["injected_reached_rank_1"]]
        if len(reached):
            s = sorted({float(x) for x in reached["strength"]})[0]
            log.warning("[%s] the injected label proxy reaches SHAP rank 1 at strength "
                        "%.2f - a shortcut of that strength is enough to dominate the "
                        "explanation.", ds, s)
        else:
            log.info("[%s] the injected proxy never reached rank 1 at strengths %s.",
                     ds, sorted({float(x) for x in proxy['strength']}))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task B4: faithfulness diagnostics")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--stages", nargs="*", default=["b41", "b42", "b43"])
    args = ap.parse_args()
    ds = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
    res = run_faithfulness(ds, seeds=args.seeds, stages=tuple(args.stages))

    # MERGE into the existing summary. A stage-scoped run (e.g. --stages b43)
    # must not delete the records written by stages it did not run.
    summary_path = RESULTS_DIR / "faithfulness_summary.json"
    merged = {}
    if summary_path.exists():
        try:
            merged = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update({k: v.to_dict("records") for k, v in res.items()})
    summary_path.write_text(json.dumps(merged, indent=2, default=str),
                            encoding="utf-8")
    log.info("faithfulness_summary.json now holds: %s",
             {k: len(v) for k, v in merged.items()})
