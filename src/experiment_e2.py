"""Task 9 (E2) - within-dataset SHAP consistency baseline.

This is the ceiling experiment. Two ensembles are trained on two disjoint,
class-stratified halves of the SAME dataset, so they see the same distribution,
the same features and the same tuned hyperparameters. Whatever disagreement
appears between their SHAP rankings is therefore attributable to sampling and
model randomness alone - not to any dataset difference.

That number is what a later cross-dataset tau has to be judged against: a
cross-dataset drop only means something if the within-dataset baseline is high.

Methodological note on what is explained: each ensemble is explained on a
stratified sample drawn from its OWN half, with a background also drawn from that
half. The two halves are disjoint, so the two SHAP profiles are computed on
genuinely independent data. Explanation is in-sample with respect to each
ensemble's training data; both sides are treated identically, so the comparison
between them stays symmetric.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DEDUPLICATE,
    FIGURES_DIR,
    JACCARD_KS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SEEDS,
    SHAP_SAMPLE_SIZE,
    TAU_STABLE_THRESHOLD,
    TAU_UNSTABLE_THRESHOLD,
    available_configs,
    banner,
    get_logger,
    resolve_config,
    set_global_seed,
    write_csv_atomic,
)
from src.consistency import (  # noqa: E402
    bootstrap_ci,
    directional_consistency_profile,
    jaccard_at_k,
    kendall_tau,
    permutation_null_tau,
    sign_agreement_profile,
)
from src.models import ENSEMBLE_NAME, build_models  # noqa: E402
from src.preprocessing import stratified_halves, stratified_halves_for_config  # noqa: E402
from src.shap_utils import (  # noqa: E402
    compute_shap_values,
    directional_importance,
    global_importance,
    make_background,
    select_shap_sample,
    signed_importance,
)
from src.tuning import get_params_for_models  # noqa: E402

log = get_logger("experiment_e2")

E2_COLUMNS = ["dataset", "seed", "kendall_tau", "tau_pvalue",
              "jaccard_5", "jaccard_10", "jaccard_15", "sign_agreement"]


def _fit_half_and_explain(dataset: str, seed: int, half_id: str, X: np.ndarray,
                          y: np.ndarray, feature_names: list[str],
                          params: dict, cache_key: str | None = None,
                          ) -> tuple[pd.Series, pd.Series, dict]:
    """Train one ensemble on one half and return its (|SHAP|, signed SHAP) profiles."""
    # Each half is an independent pipeline: its own scaler, fitted on its own data.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ensemble = build_models(seed, params)[ENSEMBLE_NAME]
    t0 = time.perf_counter()
    ensemble.fit(Xs, y)
    fit_s = time.perf_counter() - t0

    train_acc = float(ensemble.score(Xs, y))
    background = make_background(Xs, y, seed)
    X_explain, _ = select_shap_sample(Xs, y, seed=seed, n=SHAP_SAMPLE_SIZE)

    t0 = time.perf_counter()
    values = compute_shap_values(
        ensemble, X_explain, seed, background=background,
        cache_key=cache_key or f"e2_{dataset}_half{half_id}_seed{seed}")
    shap_s = time.perf_counter() - t0

    imp = global_importance(values, feature_names)
    signed = signed_importance(values, feature_names)
    direction = directional_importance(values, X_explain, feature_names)
    log.info("[%s seed=%d half=%s] n_train=%d fit=%.1fs train_acc=%.4f "
             "shap(n=%d)=%.1fs top-5: %s", dataset, seed, half_id, len(y), fit_s,
             train_acc, len(X_explain), shap_s, ", ".join(imp.nlargest(5).index))
    return imp, signed, {"fit_seconds": round(fit_s, 2), "shap_seconds": round(shap_s, 2),
                         "n_train": int(len(y)), "n_explained": int(len(X_explain)),
                         "train_accuracy": train_acc, "_direction": direction,
                         # Retained for Task B3's directional_consistency, which needs
                         # the raw values rather than a per-feature summary.
                         "_shap_values": values, "_X_explain": X_explain}


def run_seed(dataset: str, seed: int, params: dict) -> dict:
    """One E2 replicate: two halves, two ensembles, all consistency metrics."""
    set_global_seed(seed)
    (Xa, ya), (Xb, yb), feature_names = stratified_halves(dataset, seed)
    log.info("[%s seed=%d] halves: A=%s (phish %.4f)  B=%s (phish %.4f)",
             dataset, seed, Xa.shape, ya.mean(), Xb.shape, yb.mean())

    imp_a, signed_a, meta_a = _fit_half_and_explain(dataset, seed, "A", Xa, ya,
                                                    feature_names, params)
    imp_b, signed_b, meta_b = _fit_half_and_explain(dataset, seed, "B", Xb, yb,
                                                    feature_names, params)

    # Both profiles are indexed by the same feature list in the same order.
    b_aligned = imp_b.reindex(imp_a.index)
    signed_b_aligned = signed_b.reindex(signed_a.index)

    tau, p_analytic = kendall_tau(imp_a.to_numpy(), b_aligned.to_numpy())
    perm = permutation_null_tau(imp_a.to_numpy(), b_aligned.to_numpy(), seed=seed)
    row = {
        "dataset": dataset, "seed": seed,
        "kendall_tau": tau,
        "tau_pvalue": p_analytic,
        "tau_pvalue_permutation": perm["p_empirical"],
        "n_features": int(len(imp_a)),
        "top1_half_a": imp_a.idxmax(),
        "top1_half_b": imp_b.idxmax(),
    }
    for k in JACCARD_KS:
        row[f"jaccard_{k}"] = jaccard_at_k(imp_a.to_numpy(), b_aligned.to_numpy(), k)
    # The v1 `sign_agreement` column is the UNFILTERED metric, which is what this
    # path always reported. The corrected variants are emitted alongside it so the
    # v1 output stays comparable with the A5 table.
    profile = sign_agreement_profile(signed_a.to_numpy(), signed_b_aligned.to_numpy(),
                                     imp_a.to_numpy(), b_aligned.to_numpy())
    row["sign_agreement"] = profile["sign_agreement_unfiltered"]
    row.update(profile)
    for _m in (meta_a, meta_b):       # v1 schema carries no direction columns
        for _k in ("_direction", "_shap_values", "_X_explain"):
            _m.pop(_k, None)

    # Feature-level bootstrap CI for this replicate's tau.
    ci = bootstrap_ci(kendall_tau, (imp_a.to_numpy(), b_aligned.to_numpy()),
                      n=N_BOOTSTRAP, seed=seed)
    row["tau_ci_lower"] = ci["ci_lower"]
    row["tau_ci_upper"] = ci["ci_upper"]
    row.update({f"half_a_{k}": v for k, v in meta_a.items()})
    row.update({f"half_b_{k}": v for k, v in meta_b.items()})

    log.info("[%s seed=%d] tau=%.4f (p=%.3g, perm p=%.4f, 95%% CI [%.3f, %.3f]) "
             "J@5=%.3f J@10=%.3f J@15=%.3f sign=%.1f%%", dataset, seed, tau, p_analytic,
             perm["p_empirical"], ci["ci_lower"], ci["ci_upper"], row["jaccard_5"],
             row["jaccard_10"], row["jaccard_15"], row["sign_agreement"])
    return row


# --------------------------------------------------------------------------- #
# Task A5 - E2 re-run across configs, with the corrected sign-agreement metric
# --------------------------------------------------------------------------- #
E2_V2_COLUMNS = [
    "config", "dataset", "seed", "kendall_tau", "tau_pvalue",
    "jaccard_5", "jaccard_10", "jaccard_15",
    "sign_agreement_top5", "sign_agreement_top10", "sign_agreement_top15",
    "sign_agreement_weighted", "sign_agreement_unfiltered", "n_features_sign_top10",
    "top1_half_a", "top1_half_b",
]


def _e2_cache_key(cfg, half_id: str, seed: int) -> str:
    """SHAP cache key for one half of one config.

    phiusiil_full trains on byte-identical data to the Phase A E2 run, so it
    deliberately reuses that run's key rather than recomputing several hours of
    TreeSHAP. This is safe regardless: `compute_shap_values` appends a hash of the
    explained matrix to the filename, so a key collision on different data cannot
    produce a false cache hit.
    """
    if cfg.name in _LEGACY_KEY_CONFIGS:
        return f"e2_{cfg.dataset}_half{half_id}_seed{seed}"
    return f"e2_{cfg.name}_half{half_id}_seed{seed}"


# Configs whose halves are byte-identical to the Phase A E2 run (same de-duplication
# policy, same features, same seed, same split), so they reuse that run's cached
# arrays instead of recomputing hours of TreeSHAP.
_LEGACY_KEY_CONFIGS = {"phiusiil_full", "uci_dedup"}


def run_seed_config(config, seed: int, params: dict) -> dict:
    """One E2 replicate for one experiment config."""
    cfg = resolve_config(config) if isinstance(config, str) else config
    set_global_seed(seed)
    (Xa, ya), (Xb, yb), feature_names = stratified_halves_for_config(cfg, seed)
    log.info("[%s seed=%d] halves: A=%s (phish %.4f)  B=%s (phish %.4f)",
             cfg.name, seed, Xa.shape, ya.mean(), Xb.shape, yb.mean())

    imp_a, signed_a, meta_a = _fit_half_and_explain(
        cfg.dataset, seed, "A", Xa, ya, feature_names, params,
        cache_key=_e2_cache_key(cfg, "A", seed))
    imp_b, signed_b, meta_b = _fit_half_and_explain(
        cfg.dataset, seed, "B", Xb, yb, feature_names, params,
        cache_key=_e2_cache_key(cfg, "B", seed))

    b_aligned = imp_b.reindex(imp_a.index)
    signed_b_aligned = signed_b.reindex(signed_a.index)
    ia, ib = imp_a.to_numpy(), b_aligned.to_numpy()
    sa, sb = signed_a.to_numpy(), signed_b_aligned.to_numpy()

    tau, p_analytic = kendall_tau(ia, ib)
    perm = permutation_null_tau(ia, ib, seed=seed)
    row = {
        "config": cfg.name, "dataset": cfg.dataset, "seed": seed,
        "dedup": cfg.dedup,
        "kendall_tau": tau,
        "tau_pvalue": p_analytic,
        "tau_pvalue_permutation": perm["p_empirical"],
        "n_features": int(len(imp_a)),
        "top1_half_a": imp_a.idxmax(),
        "top1_half_b": imp_b.idxmax(),
        "top1_agrees": bool(imp_a.idxmax() == imp_b.idxmax()),
    }
    for k in JACCARD_KS:
        row[f"jaccard_{k}"] = jaccard_at_k(ia, ib, k)
    row.update(sign_agreement_profile(sa, sb, ia, ib))

    # Supplementary, NOT a replacement for the A2-specified metric above. Mean
    # signed SHAP cancels to near zero for features that contribute in both
    # directions, so its sign is noise even for important features - see
    # `directional_importance` and results/sign_agreement_diagnostic_v2.json.
    # These columns repeat the same agreement measurement using a direction
    # statistic that does not cancel, so the two can be compared in the paper.
    da = meta_a.pop("_direction").to_numpy()
    db = meta_b.pop("_direction").reindex(signed_a.index).to_numpy()
    direction_profile = sign_agreement_profile(da, db, ia, ib)
    row.update({f"direction_{k}": v for k, v in direction_profile.items()})

    # Task B3: the formalised directional-consistency metric, across the full
    # (top_k, rho_min) grid so threshold sensitivity is visible rather than
    # assumed. Reported ALONGSIDE sign_agreement_unfiltered - the gap between a
    # naive mean-signed-SHAP agreement and this is itself the finding.
    sa_vals, xa_exp = meta_a.pop("_shap_values"), meta_a.pop("_X_explain")
    sb_vals, xb_exp = meta_b.pop("_shap_values"), meta_b.pop("_X_explain")
    row.update(directional_consistency_profile(xa_exp, sa_vals, xb_exp, sb_vals,
                                               list(imp_a.index)))
    row["direction_cancellation_ratio_a"] = float(
        np.median(np.abs(sa) / np.maximum(ia, 1e-12)))
    row["direction_cancellation_ratio_b"] = float(
        np.median(np.abs(sb) / np.maximum(ib, 1e-12)))

    ci = bootstrap_ci(kendall_tau, (ia, ib), n=N_BOOTSTRAP, seed=seed)
    row["tau_ci_lower"] = ci["ci_lower"]
    row["tau_ci_upper"] = ci["ci_upper"]
    row.update({f"half_a_{k}": v for k, v in meta_a.items()})
    row.update({f"half_b_{k}": v for k, v in meta_b.items()})

    log.info("[%s seed=%d] tau=%.4f J@10=%.3f | sign top5=%.1f top10=%.1f top15=%.1f "
             "weighted=%.1f unfiltered=%.1f (n@10=%d) | top1 %s/%s",
             cfg.name, seed, tau, row["jaccard_10"], row["sign_agreement_top5"],
             row["sign_agreement_top10"], row["sign_agreement_top15"],
             row["sign_agreement_weighted"], row["sign_agreement_unfiltered"],
             row["n_features_sign_top10"], row["top1_half_a"], row["top1_half_b"])
    log.info("[%s seed=%d] direction-based (supplementary): top5=%.1f top10=%.1f "
             "top15=%.1f weighted=%.1f | mean-signed cancellation ratio %.3f/%.3f",
             cfg.name, seed, row["direction_sign_agreement_top5"],
             row["direction_sign_agreement_top10"], row["direction_sign_agreement_top15"],
             row["direction_sign_agreement_weighted"],
             row["direction_cancellation_ratio_a"], row["direction_cancellation_ratio_b"])
    return row


def aggregate_v2(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std across seeds per config, plus bootstrap CIs."""
    metric_cols = ["kendall_tau", "jaccard_5", "jaccard_10", "jaccard_15",
                   "sign_agreement_top5", "sign_agreement_top10", "sign_agreement_top15",
                   "sign_agreement_weighted", "sign_agreement_unfiltered"]
    rows = []
    for cfg_name, g in df.groupby("config"):
        rec = {"config": cfg_name, "dataset": g["dataset"].iloc[0],
               "n_seeds": int(len(g)), "n_features": int(g["n_features"].iloc[0]),
               "top1_agreement_rate": float(g["top1_agrees"].mean()),
               "n_features_sign_top10_mean": float(g["n_features_sign_top10"].mean())}
        for m in metric_cols:
            vals = g[m].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                rec[f"{m}_mean"] = rec[f"{m}_std"] = float("nan")
                rec[f"{m}_ci_lower"] = rec[f"{m}_ci_upper"] = float("nan")
                continue
            rec[f"{m}_mean"] = float(vals.mean())
            rec[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci = bootstrap_ci(np.mean, vals, n=N_BOOTSTRAP, seed=42)
            rec[f"{m}_ci_lower"] = ci["ci_lower"]
            rec[f"{m}_ci_upper"] = ci["ci_upper"]
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_baseline_v2(summary: pd.DataFrame, per_seed: pd.DataFrame) -> Path:
    """Kendall's tau by CONFIG with error bars across seeds."""
    summary = summary.sort_values("config").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(summary))
    colours = ["#4C72B0" if c.startswith("uci") else "#C44E52" for c in summary["config"]]
    ax.bar(x, summary["kendall_tau_mean"], yerr=summary["kendall_tau_std"],
           capsize=8, color=colours, alpha=0.85, width=0.55,
           error_kw={"elinewidth": 1.6, "ecolor": "#22303f"})

    for i, cfg_name in enumerate(summary["config"]):
        pts = per_seed.loc[per_seed["config"] == cfg_name, "kendall_tau"].to_numpy()
        jitter = np.linspace(-0.13, 0.13, len(pts))
        ax.scatter(np.full(len(pts), i) + jitter, pts, color="#22303f", zorder=3,
                   s=26, label="per-seed" if i == 0 else None)

    ax.axhline(TAU_STABLE_THRESHOLD, ls="--", color="green", lw=1.3,
               label=f"stable (tau > {TAU_STABLE_THRESHOLD})")
    ax.axhline(TAU_UNSTABLE_THRESHOLD, ls="--", color="red", lw=1.3,
               label=f"unstable (tau < {TAU_UNSTABLE_THRESHOLD})")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["config"], rotation=12, ha="right")
    ax.set_ylabel("Kendall's tau between disjoint halves")
    ax.set_ylim(0, 1.05)
    ax.set_title("E2 v2: within-dataset SHAP consistency baseline by config\n"
                 f"(mean +/- std over {int(summary['n_seeds'].max())} seeds)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIGURES_DIR / "e2_within_baseline_v2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def interpret_v2(summary: pd.DataFrame) -> list[str]:
    lines = []
    for _, r in summary.iterrows():
        cfg_name, tau = r["config"], r["kendall_tau_mean"]
        lines.append(f"[{cfg_name}] within-dataset tau = {tau:.4f} +/- {r['kendall_tau_std']:.4f} "
                     f"(95% CI [{r['kendall_tau_ci_lower']:.4f}, {r['kendall_tau_ci_upper']:.4f}])"
                     f", top-1 agreement {100 * r['top1_agreement_rate']:.0f}% of seeds")
        if tau > TAU_STABLE_THRESHOLD:
            lines.append(f"    STABLE: tau > {TAU_STABLE_THRESHOLD}.")
        elif tau < TAU_UNSTABLE_THRESHOLD:
            lines.append(f"    UNSTABLE: tau < {TAU_UNSTABLE_THRESHOLD}. SHAP is not stable "
                         f"even within this dataset; this must be reported prominently.")
        else:
            lines.append(f"    INTERMEDIATE: {TAU_UNSTABLE_THRESHOLD} <= tau <= "
                         f"{TAU_STABLE_THRESHOLD}.")
        gap = r["sign_agreement_top10_mean"] - r["sign_agreement_unfiltered_mean"]
        lines.append(f"    sign agreement: top10 {r['sign_agreement_top10_mean']:.1f}% vs "
                     f"unfiltered {r['sign_agreement_unfiltered_mean']:.1f}% "
                     f"(gap {gap:+.1f} points, n@10 = {r['n_features_sign_top10_mean']:.1f})")
    return lines


def _write_e2_v2(rows: list[dict]) -> pd.DataFrame:
    """Persist the A5 table built so far, after every seed."""
    per_seed = pd.DataFrame(rows)
    ordered = E2_V2_COLUMNS + [c for c in per_seed.columns if c not in E2_V2_COLUMNS]
    per_seed = per_seed[ordered]
    write_csv_atomic(per_seed, RESULTS_DIR / "e2_within_dataset_baseline_v2.csv")
    return per_seed


def sign_cancellation_diagnostic(per_seed: pd.DataFrame) -> dict:
    """Evidence for why the A2 metric behaves as it does, written to results/.

    The A2 brief attributes the ~50% Phase A sign agreement to the unimportant
    tail. That diagnosis predicts thresholding will RAISE the number. Where the
    thresholded value instead falls BELOW the unfiltered one, the cause must be
    different, and the cancellation ratio identifies it: mean signed SHAP divided
    by mean |SHAP|, which is near zero exactly when a feature contributes in both
    directions and its mean therefore carries no direction information.
    """
    out = {}
    for cfg_name, g in per_seed.groupby("config"):
        top10 = float(g["sign_agreement_top10"].mean())
        unfiltered = float(g["sign_agreement_unfiltered"].mean())
        direction10 = float(g["direction_sign_agreement_top10"].mean())
        ratio = float(g[["direction_cancellation_ratio_a",
                         "direction_cancellation_ratio_b"]].to_numpy().mean())
        thresholding_helped = top10 > unfiltered
        out[cfg_name] = {
            "sign_agreement_top10": top10,
            "sign_agreement_unfiltered": unfiltered,
            "thresholding_raised_the_metric": bool(thresholding_helped),
            "direction_based_top10": direction10,
            "median_cancellation_ratio": ratio,
            "verdict": (
                "A2's premise holds: thresholding raises the metric."
                if thresholding_helped else
                "A2's premise does NOT hold here: thresholding LOWERS the metric. Mean "
                "signed SHAP cancels for the important features (median |mean signed| / "
                f"mean |SHAP| = {ratio:.3f}), so their sign is noise regardless of "
                "importance. The direction-based supplement is the trustworthy figure."),
        }
        log.info("[%s] sign top10=%.1f%% vs unfiltered=%.1f%% -> %s", cfg_name, top10,
                 unfiltered, "thresholding helped" if thresholding_helped
                 else "THRESHOLDING DID NOT HELP")
        if not thresholding_helped:
            log.warning("[%s] median cancellation ratio %.3f: mean signed SHAP is ~0 "
                        "relative to mean |SHAP| for these features, so sign agreement "
                        "computed on it measures noise. Direction-based agreement "
                        "(Spearman corr of feature value with SHAP value) is %.1f%%.",
                        cfg_name, ratio, direction10)
    (RESULTS_DIR / "sign_agreement_diagnostic_v2.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info("-> %s", RESULTS_DIR / "sign_agreement_diagnostic_v2.json")
    return out


def run_e2_v2(configs: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    """Task A5: E2 across every experiment config, with corrected sign agreement."""
    from src.tuning import get_params_for_config

    seeds = seeds or SEEDS
    configs = configs or available_configs()
    banner(log, "TASK A5 (E2 v2): WITHIN-DATASET CONSISTENCY ACROSS CONFIGS")

    rows = []
    for name in configs:
        cfg = resolve_config(name)
        log.info("")
        log.info("=== config %s ===", cfg.describe())
        params = get_params_for_config(cfg)
        if not params:
            log.warning("[%s] no tuned hyperparameters - using library defaults.", cfg.name)
        for seed in seeds:
            rows.append(run_seed_config(cfg, seed, params))
            # Each replicate costs two ensemble fits plus two TreeSHAP passes -
            # up to ~15 minutes. Written after every seed so an interrupted sweep
            # keeps the replicates it already paid for.
            _write_e2_v2(rows)
            log.info("[%s seed=%d] e2_within_dataset_baseline_v2.csv updated (%d rows)",
                     cfg.name, seed, len(rows))

    per_seed = _write_e2_v2(rows)
    out = RESULTS_DIR / "e2_within_dataset_baseline_v2.csv"
    log.info("per-seed E2 v2 results -> %s (%d rows)", out, len(per_seed))

    summary = aggregate_v2(per_seed)
    summary.to_csv(RESULTS_DIR / "e2_within_dataset_summary_v2.csv", index=False)
    fig = plot_baseline_v2(summary, per_seed)
    log.info("figure -> %s", fig)

    banner(log, "A2 SIGN-AGREEMENT DIAGNOSTIC", "-")
    sign_cancellation_diagnostic(per_seed)

    banner(log, "E2 v2 INTERPRETATION")
    for line in interpret_v2(summary):
        log.info(line)

    (RESULTS_DIR / "e2_summary_v2.json").write_text(
        json.dumps({"summary": summary.to_dict("records"),
                    "interpretation": interpret_v2(summary)}, indent=2, default=float),
        encoding="utf-8")
    return per_seed


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std across seeds, plus a percentile bootstrap CI over the seed values."""
    metric_cols = ["kendall_tau", "jaccard_5", "jaccard_10", "jaccard_15", "sign_agreement"]
    rows = []
    for ds, g in df.groupby("dataset"):
        rec = {"dataset": ds, "n_seeds": int(len(g))}
        for m in metric_cols:
            vals = g[m].to_numpy(dtype=float)
            rec[f"{m}_mean"] = float(vals.mean())
            rec[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            # Bootstrapping 5 seed-level values is a coarse interval; it is reported
            # because the spec asks for it, with the small n stated alongside.
            ci = bootstrap_ci(np.mean, vals, n=N_BOOTSTRAP, seed=42)
            rec[f"{m}_ci_lower"] = ci["ci_lower"]
            rec[f"{m}_ci_upper"] = ci["ci_upper"]
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_baseline(summary: pd.DataFrame, per_seed: pd.DataFrame) -> Path:
    """Within-dataset tau per dataset, with std error bars and the seed points."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(summary))
    ax.bar(x, summary["kendall_tau_mean"], yerr=summary["kendall_tau_std"],
           capsize=8, color="#4C72B0", alpha=0.85, width=0.55,
           error_kw={"elinewidth": 1.6, "ecolor": "#22303f"})

    for i, ds in enumerate(summary["dataset"]):
        pts = per_seed.loc[per_seed["dataset"] == ds, "kendall_tau"].to_numpy()
        jitter = np.linspace(-0.13, 0.13, len(pts))
        ax.scatter(np.full(len(pts), i) + jitter, pts, color="#22303f", zorder=3,
                   s=26, label="per-seed" if i == 0 else None)

    ax.axhline(TAU_STABLE_THRESHOLD, ls="--", color="green", lw=1.3,
               label=f"stable (tau > {TAU_STABLE_THRESHOLD})")
    ax.axhline(TAU_UNSTABLE_THRESHOLD, ls="--", color="red", lw=1.3,
               label=f"unstable (tau < {TAU_UNSTABLE_THRESHOLD})")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"])
    ax.set_ylabel("Kendall's tau between disjoint halves")
    ax.set_ylim(0, 1.05)
    ax.set_title("E2: within-dataset SHAP consistency baseline\n"
                 f"(mean +/- std over {int(summary['n_seeds'].max())} seeds)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIGURES_DIR / "e2_within_baseline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def interpret(summary: pd.DataFrame) -> list[str]:
    """The interpretation block the spec requires at the end of the run."""
    lines = []
    for _, r in summary.iterrows():
        ds, tau = r["dataset"], r["kendall_tau_mean"]
        lines.append(f"[{ds}] within-dataset tau = {tau:.4f} +/- {r['kendall_tau_std']:.4f} "
                     f"(95% CI [{r['kendall_tau_ci_lower']:.4f}, "
                     f"{r['kendall_tau_ci_upper']:.4f}])")
        if tau > TAU_STABLE_THRESHOLD:
            lines.append(f"    STABLE: tau > {TAU_STABLE_THRESHOLD}. SHAP attributions are "
                         f"reproducible on identical distributions, so any later "
                         f"cross-dataset drop is a genuine dataset effect.")
        elif tau < TAU_UNSTABLE_THRESHOLD:
            lines.append(f"    UNSTABLE: tau < {TAU_UNSTABLE_THRESHOLD}. SHAP is not stable "
                         f"even within this dataset. This materially changes the paper's "
                         f"framing and MUST be reported prominently: a cross-dataset "
                         f"difference cannot be attributed to the datasets when the "
                         f"within-dataset baseline is itself this noisy.")
        else:
            lines.append(f"    INTERMEDIATE: {TAU_UNSTABLE_THRESHOLD} <= tau <= "
                         f"{TAU_STABLE_THRESHOLD}. Neither clearly stable nor clearly "
                         f"unstable; cross-dataset results must be read against this "
                         f"partial baseline rather than against an assumed ceiling.")
    return lines


def run_e2(datasets: list[str], seeds: list[int] | None = None) -> pd.DataFrame:
    seeds = seeds or SEEDS
    banner(log, "TASK 9 (E2): WITHIN-DATASET CONSISTENCY BASELINE")

    rows = []
    for ds in datasets:
        params = get_params_for_models(ds)
        if not params:
            log.warning("[%s] no tuned hyperparameters - E2 will use library defaults, "
                        "which is NOT the configuration reported in the paper.", ds)
        for seed in seeds:
            rows.append(run_seed(ds, seed, params))

    per_seed = pd.DataFrame(rows)
    out = RESULTS_DIR / "e2_within_dataset_baseline.csv"
    # Spec-mandated column order first, then the extra diagnostics.
    ordered = E2_COLUMNS + [c for c in per_seed.columns if c not in E2_COLUMNS]
    per_seed[ordered].to_csv(out, index=False)
    log.info("per-seed E2 results -> %s", out)

    summary = aggregate(per_seed)
    summary.to_csv(RESULTS_DIR / "e2_within_dataset_summary.csv", index=False)
    fig = plot_baseline(summary, per_seed)
    log.info("figure -> %s", fig)

    banner(log, "E2 INTERPRETATION")
    for line in interpret(summary):
        log.info(line)

    (RESULTS_DIR / "e2_summary.json").write_text(
        json.dumps({"summary": summary.to_dict("records"),
                    "interpretation": interpret(summary)}, indent=2, default=float),
        encoding="utf-8")
    return per_seed


if __name__ == "__main__":
    import argparse

    from config import CONFIG_NAMES

    ap = argparse.ArgumentParser(description="Task 9 / A5: E2 within-dataset baseline")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--configs", nargs="*", default=None, choices=CONFIG_NAMES,
                    help="run the Task A5 config sweep instead of the Phase A datasets")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.configs is not None:
        run_e2_v2(args.configs or None, seeds=args.seeds)
    else:
        ds = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
        run_e2(ds, seeds=args.seeds)
