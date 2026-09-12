"""Task B5 - cross-dataset attribution consistency, final form.

GATED ON B2: runs only if the screen found at least one additional dataset with
verdict `clean` or `repairable`.

On PhiUSIIL: the brief instructed that it never be substituted to fill the gap,
written on the premise that A1.3 had returned `unrepairable`. It did not - A1.3
de-leaked it after 14 removals (1.000000 -> 0.989191), so a leak-CONTROLLED
variant exists. That variant is therefore run as an ADDITIONAL pairing alongside
the clean partners, never in place of them: the instruction's purpose was to stop
a known-artefactual dataset standing in for a clean one, and running both
satisfies that while adding a comparison the brief could not have anticipated.

Partner selection: the cleanest available benchmark, preferring one whose feature
schema genuinely aligns with UCI's rather than one that merely scores well. UCI-379
is the natural partner - its nine features carry the SAME NAMES and the SAME
ternary encoding as their UCI-327 counterparts, so the shared space needs no
value-level harmonisation and none of the PhiUSIIL mapping caveats apply.

Everything is interpreted against the within-dataset ceiling from A5: a
cross-dataset tau only means something relative to how well a dataset agrees with
itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import (  # noqa: E402
    JACCARD_KS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SEEDS,
    banner,
    get_logger,
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
from src.shap_utils import (  # noqa: E402
    compute_shap_values,
    directional_importance,
    global_importance,
    make_background,
    select_shap_sample,
    signed_importance,
)

log = get_logger("cross_dataset")

CROSS_SHAP_SAMPLE = 2000
USABLE_VERDICTS = {"clean", "repairable"}


def gate_open() -> tuple[bool, list[str], str]:
    """Did B2 find a usable partner? Returns (open, candidates, reason)."""
    path = RESULTS_DIR / "screening" / "benchmark_screening_summary.csv"
    if not path.exists():
        return False, [], "B2 screening has not produced a summary - gate cannot be evaluated"
    scr = pd.read_csv(path)
    usable = scr[scr["verdict"].isin(USABLE_VERDICTS)]
    # A sensitivity variant is the same dataset under a different label handling,
    # not an independent benchmark, so it cannot open the gate on its own.
    usable = usable[~usable["dataset"].str.contains("__", na=False)]
    if usable.empty:
        return False, [], ("no screened benchmark returned clean or repairable; "
                           "cross-dataset consistency cannot be evaluated on leak-free data")
    return True, list(usable["dataset"]), ""


def shared_schema(partner_key: str) -> tuple[pd.DataFrame, pd.Series,
                                             pd.DataFrame, pd.Series, list[str]]:
    """UCI-327 and the partner restricted to their shared, identically encoded features."""
    from src.benchmark_screen import load_external
    from src.preprocessing import get_xy

    Xu, yu = get_xy("uci", dedup=False)
    Xp, yp, _ = load_external(partner_key)

    # Match on case-insensitive name. For UCI-379 this is exact and lossless: the
    # columns are the same features under the same ternary convention.
    lut = {c.lower(): c for c in Xp.columns}
    shared_u = [c for c in Xu.columns if c.lower() in lut]
    shared_p = [lut[c.lower()] for c in shared_u]
    if len(shared_u) < 3:
        raise ValueError(f"only {len(shared_u)} shared features with {partner_key!r} - "
                         f"too few for a consistency analysis")
    log.info("shared schema with %s: %d features %s", partner_key, len(shared_u), shared_u)
    return Xu[shared_u], yu, Xp[shared_p], yp, shared_u


def shared_schema_leakfree() -> tuple[pd.DataFrame, pd.Series,
                                      pd.DataFrame, pd.Series, list[str], dict]:
    """UCI-327 against the LEAK-CONTROLLED PhiUSIIL variant, on shared concepts.

    Uses the B1.1 conceptual mapping rather than name matching, since the two
    schemas share no column names. Only concepts whose PhiUSIIL source features
    SURVIVE de-leaking can be used - and that exclusion is itself a result worth
    recording, because de-leaking removes IsHTTPS, NoOfExternalRef and NoOfSelfRef,
    which are precisely the sources of the two strongest shared concepts
    (ssl_state, |rho| 0.61-0.74; anchor_external_ratio, |rho| 0.70 on UCI).

    So the leak-controlled comparison is not merely a cleaner version of the full
    one: it necessarily runs on a weaker shared space, and any drop in agreement
    must be read with that in mind rather than attributed to de-leaking alone.
    """
    import config as _config
    from src.preprocessing import get_xy
    from src.transfer import FEATURE_MAP, to_shared_schema

    leakfree = _config.PHIUSIIL_LEAKFREE_FEATURES
    if not leakfree:
        raise RuntimeError("PHIUSIIL_LEAKFREE_FEATURES is not defined")
    survivors = set(leakfree)

    usable, lost = [], {}
    for m in FEATURE_MAP:
        missing = [s for s in m.phiusiil_source if s not in survivors]
        if missing:
            lost[m.concept] = f"source feature(s) removed by de-leaking: {missing}"
        else:
            usable.append(m.concept)

    if len(usable) < 3:
        raise ValueError(f"only {len(usable)} concepts survive de-leaking - too few")

    Xp_all, yp = to_shared_schema("phiusiil")
    Xu_all, yu = to_shared_schema("uci", dedup=False)
    notes = {"concepts_used": usable, "concepts_lost_to_deleaking": lost,
             "n_used": len(usable), "n_lost": len(lost)}

    log.info("leak-free shared schema: %d of %d concepts usable", len(usable),
             len(FEATURE_MAP))
    for c, why in lost.items():
        log.warning("  concept %r DROPPED - %s", c, why)
    return Xu_all[usable], yu, Xp_all[usable], yp, usable, notes


def _explain(X: pd.DataFrame, y: pd.Series, names: list[str], seed: int, tag: str):
    """Train an ensemble on one dataset and return its attribution profiles."""
    from src.tuning import get_params_for_models

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.to_numpy(dtype=float))
    yv = y.to_numpy()
    params = get_params_for_models("uci")      # same family/encoding for both sides
    ens = build_models(seed, params)[ENSEMBLE_NAME]
    ens.fit(Xs, yv)

    bg = make_background(Xs, yv, seed)
    X_exp, _ = select_shap_sample(Xs, yv, seed=seed, n=CROSS_SHAP_SAMPLE)
    vals = compute_shap_values(ens, X_exp, seed, background=bg,
                               cache_key=f"cross_{tag}_seed{seed}")
    return (global_importance(vals, names), signed_importance(vals, names),
            directional_importance(vals, X_exp, names), vals, X_exp)


def run_cross_dataset(partner_key: str, seeds: list[int] | None = None,
                      out_name: str = "cross_dataset_consistency.csv") -> pd.DataFrame:
    seeds = seeds or SEEDS
    banner(log, f"B5 CROSS-DATASET CONSISTENCY: uci vs {partner_key}")
    schema_notes = {}
    if partner_key == "phiusiil_leakfree":
        Xu, yu, Xp, yp, names, schema_notes = shared_schema_leakfree()
    else:
        Xu, yu, Xp, yp, names = shared_schema(partner_key)

    rows = []
    for seed in seeds:
        set_global_seed(seed)
        iu, su, du, vu, xu = _explain(Xu, yu, names, seed, "uci")
        ip, sp, dp, vp, xp = _explain(Xp, yp, names, seed, partner_key)

        a, b = iu.to_numpy(), ip.reindex(iu.index).to_numpy()
        tau, p = kendall_tau(a, b)
        perm = permutation_null_tau(a, b, seed=seed)
        ci = bootstrap_ci(kendall_tau, (a, b), n=N_BOOTSTRAP, seed=seed)

        row = {"dataset_a": "uci", "dataset_b": partner_key, "seed": seed,
               "n_features": len(names), "kendall_tau": tau, "tau_pvalue": p,
               "tau_pvalue_permutation": perm["p_empirical"],
               "tau_ci_lower": ci["ci_lower"], "tau_ci_upper": ci["ci_upper"],
               "top1_a": iu.idxmax(), "top1_b": ip.idxmax(),
               "top1_agrees": bool(iu.idxmax() == ip.idxmax())}
        for k in JACCARD_KS:
            row[f"jaccard_{k}"] = jaccard_at_k(a, b, min(k, len(names)))
        row.update(sign_agreement_profile(su.to_numpy(),
                                          sp.reindex(su.index).to_numpy(), a, b))
        row.update(directional_consistency_profile(xu, vu, xp, vp, names,
                                                   ks=tuple(k for k in JACCARD_KS
                                                            if k <= len(names)) or (len(names),)))

        # Group-level SHAP mass: each feature's share of total attribution, so the
        # two datasets can be compared on where explanation weight concentrates
        # rather than only on rank order.
        mass_a, mass_b = a / a.sum(), b / b.sum()
        row["shap_mass_l1_distance"] = float(np.abs(mass_a - mass_b).sum())
        row["shap_mass_top1_a"] = float(mass_a.max())
        row["shap_mass_top1_b"] = float(mass_b.max())
        rows.append(row)
        log.info("[seed %d] tau=%.4f (perm p=%.4f, CI [%.3f, %.3f]) J@5=%.3f "
                 "top1 %s/%s | SHAP-mass L1=%.3f", seed, tau, perm["p_empirical"],
                 ci["ci_lower"], ci["ci_upper"], row["jaccard_5"],
                 row["top1_a"], row["top1_b"], row["shap_mass_l1_distance"])
        write_csv_atomic(pd.DataFrame(rows), RESULTS_DIR / out_name)

    df = pd.DataFrame(rows)
    if schema_notes:
        df["n_concepts_lost_to_deleaking"] = schema_notes["n_lost"]
        df["concepts_lost_to_deleaking"] = "; ".join(schema_notes["concepts_lost_to_deleaking"])
        write_csv_atomic(df, RESULTS_DIR / out_name)
        (RESULTS_DIR / "cross_dataset_leakfree_schema.json").write_text(
            json.dumps(schema_notes, indent=2), encoding="utf-8")
    _interpret(df, partner_key)
    return df


def _interpret(df: pd.DataFrame, partner_key: str) -> None:
    """Judge the cross-dataset tau against the within-dataset ceiling from A5."""
    banner(log, "B5 INTERPRETATION", "-")
    tau_mean = float(df["kendall_tau"].mean())
    tau_std = float(df["kendall_tau"].std(ddof=1)) if len(df) > 1 else 0.0
    log.info("cross-dataset tau (uci vs %s) = %.4f +/- %.4f", partner_key, tau_mean, tau_std)

    ceiling_path = RESULTS_DIR / "e2_within_dataset_summary_v2.csv"
    ceiling = None
    if ceiling_path.exists():
        c = pd.read_csv(ceiling_path)
        row = c[c["config"] == "uci_full"]
        if len(row):
            ceiling = float(row["kendall_tau_mean"].iloc[0])
    if ceiling is None:
        log.warning("within-dataset ceiling unavailable (A5 has not produced "
                    "e2_within_dataset_summary_v2.csv) - the cross-dataset figure cannot "
                    "yet be interpreted against it, and must not be reported alone.")
        return

    log.info("within-dataset ceiling (uci_full) = %.4f", ceiling)
    gap = ceiling - tau_mean
    log.info("gap = %.4f", gap)
    if gap <= 0.05:
        log.info("Cross-dataset agreement is close to the within-dataset ceiling: "
                 "attributions are about as consistent ACROSS these datasets as they are "
                 "WITHIN one, so the dataset boundary costs little.")
    elif tau_mean < 0.3:
        log.warning("Cross-dataset agreement is far below the ceiling (%.4f vs %.4f). "
                    "Attributions largely do not survive the move between collections.",
                    tau_mean, ceiling)
    else:
        log.info("Cross-dataset agreement is materially below the within-dataset ceiling "
                 "(%.4f vs %.4f): a real dataset effect, but not a collapse.",
                 tau_mean, ceiling)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task B5: cross-dataset consistency")
    ap.add_argument("--partner", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    is_open, candidates, reason = gate_open()
    if not is_open:
        banner(log, "B5 SKIPPED - GATE CLOSED")
        log.warning("%s", reason)
        log.warning("Reporting this as a finding about the state of available phishing "
                    "benchmarks. PhiUSIIL is NOT substituted: it is unrepairable, and "
                    "using it would reintroduce the artifact under study.")
        (RESULTS_DIR / "cross_dataset_skipped.json").write_text(
            json.dumps({"ran": False, "reason": reason}, indent=2), encoding="utf-8")
        raise SystemExit(0)

    import config as _config

    partners = [args.partner] if args.partner else list(candidates)
    # The brief said never to substitute PhiUSIIL, written on the assumption it was
    # unusable. A1.3 in fact de-leaked it, so the leak-CONTROLLED variant is a
    # legitimate additional comparison - run ALONGSIDE the clean partners, never
    # instead of them.
    if not args.partner and _config.PHIUSIIL_LEAKFREE_FEATURES:
        partners.append("phiusiil_leakfree")
    log.info("gate OPEN; clean partners: %s", candidates)
    log.info("running pairings: %s", partners)

    frames = []
    for pk in partners:
        try:
            out = f"cross_dataset_{pk}.csv"
            frames.append(run_cross_dataset(pk, seeds=args.seeds, out_name=out))
        except Exception as exc:
            log.error("pairing uci vs %s FAILED: %s: %s", pk, type(exc).__name__, exc)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        write_csv_atomic(combined, RESULTS_DIR / "cross_dataset_consistency.csv")
        banner(log, "B5 ALL PAIRINGS", "-")
        g = combined.groupby("dataset_b")[["kendall_tau", "jaccard_5",
                                           "shap_mass_l1_distance"]].mean().round(4)
        for line in g.to_string().splitlines():
            log.info(line)
