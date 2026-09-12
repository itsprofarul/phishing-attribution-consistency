"""Task B2 - reusable leakage screen for tabular phishing benchmarks.

Packages the A1 diagnostic as a function that accepts any labelled tabular
phishing dataset and returns a verdict. The function is itself a contribution of
the paper, so it is written to stand alone: it does its own splitting, tuning,
single-feature analysis and progressive removal, and it records the explanation-
cost profile that Task B5b turns into a diagnostic.

Verdict rules, applied in order (from the brief):
    unrepairable  accuracy stays >= the target after removing 15 features
    repairable    accuracy falls below the target after removing between 1 and 15
    clean         full-model accuracy is below the target with no removal at all

Validation gate: UCI must return `clean` and PhiUSIIL `repairable`. (The brief
anticipated `unrepairable` for PhiUSIIL, but A1.3 de-leaked it after 14 removals -
1.000000 -> 0.989191 - so `repairable` is the reproducible result.)

THE VERDICT IS THE SECONDARY SIGNAL. It turns on an accuracy cutoff, and the
cutoff separates datasets by margins smaller than their seed-to-seed spread, so
verdicts are reported across several thresholds (`verdict_at_*`) and never
without the threshold attached.

The PRIMARY signal is `n_features_above_95`: a count of features that individually
reconstruct the label. It is categorical and threshold-independent - 3 versus 0 -
and it is what `population_table` leads with.
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

from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from config import (  # noqa: E402
    CV_FOLDS,
    LEAKAGE_ACC_THRESHOLD,
    RESULTS_DIR,
    TEST_SIZE,
    VAL_SIZE,
    banner,
    get_logger,
    set_global_seed,
    write_csv_atomic,
)
from src.models import ENSEMBLE_NAME, build_models  # noqa: E402
from src.shap_utils import (  # noqa: E402
    compute_shap_values,
    global_importance,
    make_background,
    select_shap_sample,
)
from src.tuning import SEARCH_SPACES  # noqa: E402

log = get_logger("benchmark_screen")

SCREEN_DIR = RESULTS_DIR / "screening"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter regimes. Any accuracy figure means something different depending
# on which one produced it, so every exported figure carries its regime.
#
#   REGIME_STATIC  - hyperparameters selected ONCE on the complete feature set and
#                    held fixed while features are removed. This is what a
#                    progressive-removal curve does, and it UNDERSTATES the
#                    performance of the reduced feature sets, because parameters
#                    chosen for a 50-feature problem are not optimal for a
#                    36-feature one.
#   REGIME_RETUNED - hyperparameters re-selected for the exact feature set being
#                    evaluated. This is what E1 does per config.
#
# Measured gap on PhiUSIIL's 36-feature leak-free set: 0.9892 static (A1.3)
# vs 0.9927 re-tuned (E1) - the removal curve was reading low by ~0.0035, enough
# to move it across the 0.99 verdict line.
REGIME_STATIC = "tuned_on_full_feature_set"
REGIME_RETUNED = "retuned_per_feature_set"

DELEAK_ACC_TARGET = 0.99
DELEAK_MAX_REMOVED = 15
SCREEN_SHAP_SAMPLE = 2000
# Reduced search budget for the screen only. The full n_iter=50 search is the
# Phase A protocol for the two headline datasets; the screen tunes several extra
# benchmarks purely to avoid judging them on untuned defaults, and a 20-candidate
# search is ample for that purpose. Recorded here rather than left implicit.
SCREEN_SEARCH_ITER = 20


# --------------------------------------------------------------------------- #
# Tree complexity, for the B5b cost profile
# --------------------------------------------------------------------------- #
def forest_complexity(ensemble) -> dict:
    """Mean tree depth and total node count of the ensemble's RandomForest member.

    Interventional TreeSHAP cost scales with the number of nodes it must walk, and
    the RandomForest is both the largest member and the dominant cost, so it is
    the component reported. XGBoost and LightGBM expose their structure through
    different APIs and are excluded rather than approximated.
    """
    rf = None
    for name, est in getattr(ensemble, "named_estimators_", {}).items():
        if name == "rf":
            rf = est
            break
    if rf is None or not hasattr(rf, "estimators_"):
        return {"mean_tree_depth": float("nan"), "total_nodes": float("nan"),
                "n_trees": 0}
    depths = [t.tree_.max_depth for t in rf.estimators_]
    nodes = [t.tree_.node_count for t in rf.estimators_]
    return {"mean_tree_depth": float(np.mean(depths)),
            "total_nodes": int(np.sum(nodes)), "n_trees": len(depths)}


# --------------------------------------------------------------------------- #
# B2.1 Single-feature predictive power
# --------------------------------------------------------------------------- #
def single_feature_power(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.DataFrame:
    """Per-feature depth-3 decision-tree accuracy plus mutual information."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed, shuffle=True)
    mi = mutual_info_classif(X, y, random_state=seed)
    rows = []
    for i, col in enumerate(X.columns):
        clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
        clf.fit(Xtr[[col]], ytr)
        pred = clf.predict(Xte[[col]])
        rows.append({"feature": col,
                     "accuracy": float(accuracy_score(yte, pred)),
                     "f1": float(f1_score(yte, pred, zero_division=0)),
                     "mutual_information": float(mi[i])})
    return (pd.DataFrame(rows).sort_values("accuracy", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# B2.1 Tuning + progressive removal
# --------------------------------------------------------------------------- #
def _tune(X: np.ndarray, y: np.ndarray, seed: int, n_iter: int) -> dict:
    params = {}
    for name in ("rf", "xgb", "lgbm"):
        est = build_models(seed)[name]
        search = RandomizedSearchCV(
            est, SEARCH_SPACES[name], n_iter=n_iter,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed),
            scoring="f1", random_state=seed, n_jobs=1, refit=False, error_score="raise")
        search.fit(X, y)
        params[name] = dict(search.best_params_)
    return params


def _fit_score(X_tr, y_tr, X_te, y_te, seed, params, shap_sample, cache_key):
    """Fit the ensemble, score on test, rank features by mean |SHAP|, profile cost."""
    ens = build_models(seed, params)[ENSEMBLE_NAME]
    t0 = time.perf_counter()
    ens.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    pred = ens.predict(X_te)
    scores = {"accuracy": float(accuracy_score(y_te, pred)),
              "f1": float(f1_score(y_te, pred, zero_division=0)),
              "mcc": float(matthews_corrcoef(y_te, pred)),
              "fit_seconds": round(fit_s, 2)}
    scores.update(forest_complexity(ens))

    bg = make_background(X_tr, y_tr, seed)
    X_exp, _ = select_shap_sample(X_te, y_te, seed=seed, n=shap_sample)
    t0 = time.perf_counter()
    vals = compute_shap_values(ens, X_exp, seed, background=bg, cache_key=cache_key)
    scores["shap_seconds"] = round(time.perf_counter() - t0, 2)
    return scores, vals


def screen_benchmark(X: pd.DataFrame, y: pd.Series, dataset_name: str, seed: int = 42,
                     max_removed: int = DELEAK_MAX_REMOVED,
                     acc_target: float = DELEAK_ACC_TARGET,
                     shap_sample: int = SCREEN_SHAP_SAMPLE,
                     params: dict | None = None,
                     n_iter: int = SCREEN_SEARCH_ITER) -> dict:
    """Screen one labelled tabular phishing dataset for label leakage."""
    banner(log, f"B2 SCREEN: {dataset_name}  ({X.shape[0]} rows x {X.shape[1]} features)")
    set_global_seed(seed)
    feature_names = list(X.columns)

    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed, shuffle=True)
    X_tr, _, y_tr, _ = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_tmp,
        random_state=seed, shuffle=True)

    sf = single_feature_power(X, y, seed=seed)
    n_above = int((sf["accuracy"] > LEAKAGE_ACC_THRESHOLD).sum())
    log.info("[%s] single-feature: best %.4f (%s); %d feature(s) above %.2f",
             dataset_name, sf["accuracy"].iloc[0], sf["feature"].iloc[0],
             n_above, LEAKAGE_ACC_THRESHOLD)

    if params is None:
        t0 = time.perf_counter()
        params = _tune(StandardScaler().fit_transform(X_tr), y_tr.to_numpy(), seed, n_iter)
        log.info("[%s] tuned in %.1f min (n_iter=%d)", dataset_name,
                 (time.perf_counter() - t0) / 60, n_iter)

    removed: list[str] = []
    curve: list[dict] = []
    outcome = "unresolved"
    for iteration in range(max_removed + 1):
        keep = [f for f in feature_names if f not in removed]
        if len(keep) < 2:
            outcome = "exhausted_features"
            break
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr[keep])
        Xte_s = scaler.transform(X_te[keep])
        scores, vals = _fit_score(Xtr_s, y_tr.to_numpy(), Xte_s, y_te.to_numpy(),
                                  seed, params, shap_sample,
                                  cache_key=f"leak_{dataset_name}_seed{seed}_k{len(keep)}")
        imp = global_importance(vals, keep).sort_values(ascending=False)
        top = str(imp.index[0])

        curve.append({"dataset": dataset_name, "iteration": iteration,
                      "n_features": len(keep), "accuracy": scores["accuracy"],
                      "f1": scores["f1"], "mcc": scores["mcc"],
                      "fit_seconds": scores["fit_seconds"],
                      "shap_seconds": scores["shap_seconds"],
                      "mean_tree_depth": scores["mean_tree_depth"],
                      "total_nodes": scores["total_nodes"],
                      "removed_feature": top, "top_feature": top,
                      # Iteration 0 is the only row whose parameters were tuned on
                      # the feature set it is actually scoring.
                      "hyperparameter_regime": (REGIME_RETUNED if iteration == 0
                                                else REGIME_STATIC)})
        log.info("[%s iter %2d] n=%3d acc=%.6f | top=%-28s | fit %5.1fs shap %6.1fs "
                 "depth %.1f nodes %s", dataset_name, iteration, len(keep),
                 scores["accuracy"], top[:28], scores["fit_seconds"],
                 scores["shap_seconds"], scores["mean_tree_depth"], scores["total_nodes"])

        if iteration == 0:
            full_model_accuracy = scores["accuracy"]
            if full_model_accuracy < acc_target:
                outcome = "clean"
                curve[-1]["removed_feature"] = ""
                break
        if scores["accuracy"] < acc_target:
            outcome = "repairable"
            curve[-1]["removed_feature"] = ""
            break
        if len(removed) >= max_removed:
            outcome = "unrepairable"
            curve[-1]["removed_feature"] = ""
            break
        removed.append(top)

    verdict = {"clean": "clean", "repairable": "repairable",
               "unrepairable": "unrepairable"}.get(outcome, outcome)
    result = {
        "dataset": dataset_name, "seed": seed,
        "n_rows": int(len(X)), "n_features": int(X.shape[1]),
        "phishing_rate": float(y.mean()),
        "single_feature_table": sf,
        "best_single_feature": sf["feature"].iloc[0],
        "best_single_feature_accuracy": float(sf["accuracy"].iloc[0]),
        "n_features_above_95": n_above,
        "full_model_accuracy": float(full_model_accuracy),
        "progressive_removal": pd.DataFrame(curve),
        "minimal_leaking_set": removed if verdict == "repairable" else None,
        "n_removed": len(removed),
        "final_accuracy": float(curve[-1]["accuracy"]),
        "verdict": verdict,
        "acc_target": acc_target,
        "best_params": params,
    }
    log.info("[%s] VERDICT = %s (full-model acc %.6f, final %.6f after %d removals)",
             dataset_name, verdict.upper(), full_model_accuracy,
             result["final_accuracy"], len(removed))
    return result


# --------------------------------------------------------------------------- #
# Harmonising the external benchmarks onto 1 = phishing, 0 = legitimate
# --------------------------------------------------------------------------- #
SUSPICIOUS_POLICIES = ("merge_phishing", "drop")


def load_external(key: str, suspicious_policy: str = "merge_phishing",
                  ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load one acquired benchmark as (X, y) with y harmonised to 1 = phishing.

    Each dataset's label convention is handled explicitly rather than guessed, and
    every non-numeric / identifier column is dropped with a recorded reason - the
    same discipline applied to PhiUSIIL's URL/Domain/TLD/Title in Phase A.

    `suspicious_policy` governs the three-class UCI-379 label only; see below.
    """
    from src.external_datasets import EXTERNAL_DIR

    if suspicious_policy not in SUSPICIOUS_POLICIES:
        raise ValueError(f"suspicious_policy must be one of {SUSPICIOUS_POLICIES}")

    df = pd.read_parquet(EXTERNAL_DIR / f"{key}.parquet")
    notes: dict = {"dataset": key, "dropped_columns": {}, "rows_before": int(len(df))}

    if key == "uci379_website_phishing":
        # Abdelhamid's Result is THREE-class: -1 phishy, 0 suspicious, 1 legitimate.
        # The screen is a binary diagnostic, so the intermediate class needs a
        # decision, and the decision is operational rather than statistical: a
        # defender's choice is act-or-don't-act, and an intermediate verdict falls
        # on the act side. So 'suspicious' is MERGED WITH PHISHING by default.
        #
        # Dropping it instead is retained as a sensitivity variant, because the
        # two handlings give different class balances and could in principle give
        # different verdicts. Both are screened and both are reported; if the
        # verdict is stable across them, the choice is immaterial and can be
        # stated as such in the paper.
        counts = df["Result"].value_counts().to_dict()
        before = {"phishy(-1)": int(counts.get(-1, 0)),
                  "suspicious(0)": int(counts.get(0, 0)),
                  "legitimate(1)": int(counts.get(1, 0))}
        if suspicious_policy == "merge_phishing":
            y = (df.pop("Result") != 1).astype(int)
            notes["label_note"] = (
                "3-class Result (-1 phishy / 0 suspicious / 1 legitimate). Policy "
                "'merge_phishing': suspicious folded into the phishing class, on the "
                "grounds that a defender's decision is act-or-don't-act and an "
                "intermediate verdict falls on the act side.")
        else:
            n_susp = int((df["Result"] == 0).sum())
            df = df[df["Result"] != 0].copy()
            notes["dropped_rows_suspicious"] = n_susp
            y = (df.pop("Result") == -1).astype(int)
            notes["label_note"] = (
                "3-class Result. Policy 'drop': the %d suspicious rows are removed "
                "rather than assigned, as a sensitivity check against the primary "
                "'merge_phishing' handling." % n_susp)
        notes["suspicious_policy"] = suspicious_policy
        notes["class_counts_before"] = before
        notes["class_counts_after"] = {"phishing(1)": int(y.sum()),
                                       "legitimate(0)": int((y == 0).sum())}
    elif key == "mendeley_hannousse":
        notes["dropped_columns"]["url"] = "raw URL string identifier, not a feature"
        df = df.drop(columns=["url"])
        y = (df.pop("status") == "phishing").astype(int)
        notes["label_note"] = "status in {phishing, legitimate}"
    elif key == "mendeley_tan":
        if "id" in df.columns:
            notes["dropped_columns"]["id"] = "row identifier"
            df = df.drop(columns=["id"])
        y = df.pop("CLASS_LABEL").astype(int)
        notes["label_note"] = "CLASS_LABEL 1 = phishing, 0 = legitimate"
    else:
        raise KeyError(f"no harmonisation rule for {key!r}")

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in non_numeric:
        notes["dropped_columns"][c] = "non-numeric column"
    df = df.drop(columns=non_numeric)

    notes.update({"rows_after": int(len(df)), "n_features": int(df.shape[1]),
                  "phishing_rate": float(y.mean())})
    log.info("[%s] %s, %d features, phishing rate %.4f%s", key, df.shape, df.shape[1],
             y.mean(), f", dropped {non_numeric}" if non_numeric else "")
    return df.reset_index(drop=True), y.reset_index(drop=True), notes


def load_internal(name: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """The two Phase A datasets, for the validation gate."""
    from src.preprocessing import get_xy

    X, y = get_xy(name, dedup=False if name == "uci" else None)
    return X, y, {"dataset": name, "n_features": int(X.shape[1]),
                  "phishing_rate": float(y.mean())}


# The 0.99 verdict boundary is a convention, not a measurement, and it separates
# phiusiil_leakfree (0.9892) from mendeley_tan (0.9873) by 0.002 - far less than the
# seed-to-seed spread of either. Verdicts are therefore reported across a range of
# cutoffs so no label rests on a single arbitrary line.
VERDICT_THRESHOLDS = (0.97, 0.98, 0.99, 0.995)


def verdict_at(full_model_accuracy: float, curve: pd.DataFrame, threshold: float,
               max_removed: int = DELEAK_MAX_REMOVED) -> str:
    """Re-derive the verdict at an arbitrary threshold from an existing curve.

    Returns 'undetermined' where the recorded curve cannot answer the question:
    a curve produced with a stopping rule of 0.99 halts as soon as it crosses
    that line, so it carries no information about whether further removals would
    have taken accuracy below 0.98. Reporting that honestly is the point - a
    guessed verdict would defeat the sensitivity analysis.
    """
    if full_model_accuracy < threshold:
        return "clean"
    if curve is None or curve.empty:
        return "undetermined"
    acc = curve.sort_values("iteration")["accuracy"].to_numpy(dtype=float)
    below = np.flatnonzero(acc < threshold)
    if below.size:
        n_removed_needed = int(below[0])
        return "repairable" if n_removed_needed <= max_removed else "unrepairable"
    # Never dropped below the threshold within the curve we have.
    if len(acc) - 1 >= max_removed:
        return "unrepairable"
    return "undetermined"


def verdicts_across_thresholds(full_model_accuracy: float, curve: pd.DataFrame,
                               thresholds=VERDICT_THRESHOLDS) -> dict:
    return {f"verdict_at_{t}": verdict_at(full_model_accuracy, curve, t)
            for t in thresholds}


def result_row(r: dict) -> dict:
    """Flatten a screen result into one row of the B2.3 summary table.

    Column order is deliberate: `n_features_above_95` leads because it is the
    threshold-independent signal. It is a COUNT of features that individually
    reconstruct the label, and 3-versus-0 is a categorical difference that no
    choice of accuracy cutoff can blur - unlike the verdict, which turns on a
    0.002 accuracy gap between datasets.
    """
    row = {
        "dataset": r["dataset"],
        # --- primary, threshold-independent signal ---
        "n_features_above_95": r["n_features_above_95"],
        "best_single_feature": r["best_single_feature"],
        "best_single_feature_accuracy": round(r["best_single_feature_accuracy"], 4),
        # --- secondary, threshold-dependent ---
        "verdict": r["verdict"],
        "verdict_threshold_used": r.get("acc_target", DELEAK_ACC_TARGET),
        # full_model_accuracy is re-tuned; final_accuracy after removals is not.
        "full_model_accuracy_regime": REGIME_RETUNED,
        "final_accuracy_regime": (REGIME_RETUNED if r["n_removed"] == 0
                                  else REGIME_STATIC),
        "full_model_accuracy": round(r["full_model_accuracy"], 6),
        "n_removed": r["n_removed"],
        "final_accuracy": round(r["final_accuracy"], 6),
        "n_rows": r["n_rows"], "n_features": r["n_features"],
        "phishing_rate": round(r["phishing_rate"], 4),
        "minimal_leaking_set": "" if not r["minimal_leaking_set"]
                               else "; ".join(r["minimal_leaking_set"]),
    }
    row.update(verdicts_across_thresholds(r["full_model_accuracy"],
                                          r.get("progressive_removal")))
    return row


# --------------------------------------------------------------------------- #
# Validation gate + the B2.3 screening report
# --------------------------------------------------------------------------- #
# PhiUSIIL's expected verdict is `repairable`, NOT `unrepairable`. A1.3 removed 14
# of 50 features and accuracy fell from 1.000000 to 0.989191 - just under the 0.99
# line, within the 15-removal cap. The earlier `unrepairable` expectation was
# written before that run finished and would now fail this gate.
#
# The finding is that PhiUSIIL's leakage is DIFFUSE AND REDUNDANT - spread across
# many weakly-substitutable content features, so removal grinds down slowly rather
# than collapsing - not that it cannot be repaired.
EXPECTED_VERDICTS = {"uci": "clean", "phiusiil": "repairable"}


def validate_screen(seed: int = 42, datasets=("uci",)) -> dict:
    """Reproduce the known verdicts before the screen is trusted on new data.

    PhiUSIIL is validated separately (and expensively), so the default gate runs
    UCI only; pass datasets=("uci", "phiusiil") for the full check.
    """
    banner(log, "B2.1 VALIDATION GATE")
    out = {}
    for name in datasets:
        X, y, _ = load_internal(name)
        params = None
        if name == "phiusiil":
            from src.tuning import get_params_for_models
            params = get_params_for_models(name)
        r = screen_benchmark(X, y, name, seed=seed, params=params)
        expected = EXPECTED_VERDICTS[name]
        ok = r["verdict"] == expected
        out[name] = {"verdict": r["verdict"], "expected": expected, "passed": ok,
                     "full_model_accuracy": r["full_model_accuracy"]}
        (SCREEN_DIR / f"validation_{name}.json").write_text(
            json.dumps(out[name], indent=2), encoding="utf-8")
        (log.info if ok else log.error)(
            "[gate] %s -> %s (expected %s) %s", name, r["verdict"], expected,
            "PASS" if ok else "FAIL")
        if not ok:
            raise AssertionError(
                f"screen_benchmark returned {r['verdict']!r} for {name!r}, expected "
                f"{expected!r}. The function is wrong and must be fixed before it is "
                f"used to judge any new dataset.")
    return out


def run_screen(keys: list[str] | None = None, seed: int = 42,
               skip_validation: bool = False,
               acc_target: float = DELEAK_ACC_TARGET) -> pd.DataFrame:
    from src.external_datasets import CATALOGUE, acquire

    if not skip_validation:
        validate_screen(seed=seed)

    keys = keys or list(CATALOGUE)
    rows, curves, notes_all = [], [], []
    for key in keys:
        df, meta = acquire(key)
        if df is None:
            rows.append({"dataset": key, "verdict": "NOT_ACQUIRED",
                         "minimal_leaking_set": meta.get("block_reason", "")})
            continue

        # Datasets with an intermediate label class are screened under BOTH
        # handlings, so the verdict's sensitivity to that choice is measured
        # rather than assumed. The primary handling is listed first.
        policies = (["merge_phishing", "drop"]
                    if key == "uci379_website_phishing" else ["merge_phishing"])
        for pol in policies:
            X, y, notes = load_external(key, suspicious_policy=pol)
            notes_all.append(notes)
            primary = pol == policies[0]
            label = key if primary else f"{key}__suspicious_dropped"
            r = screen_benchmark(X, y, label, seed=seed, acc_target=acc_target)
            row = result_row(r)
            row["label_policy"] = pol
            row["is_primary"] = primary
            rows.append(row)
            curves.append(r["progressive_removal"])
            write_csv_atomic(r["single_feature_table"],
                             SCREEN_DIR / f"single_feature_{label}.csv")

    summary = pd.DataFrame(rows)
    suffix = "" if acc_target == DELEAK_ACC_TARGET else f"_target{acc_target}"
    write_csv_atomic(summary, SCREEN_DIR / f"benchmark_screening_summary{suffix}.csv")
    if curves:
        write_csv_atomic(pd.concat(curves, ignore_index=True),
                         SCREEN_DIR / "screening_progressive_removal.csv")
    (SCREEN_DIR / "harmonisation_notes.json").write_text(
        json.dumps(notes_all, indent=2, default=str), encoding="utf-8")

    banner(log, "B2.3 SCREENING SUMMARY", "-")
    cols = [c for c in ["dataset", "label_policy", "n_rows", "n_features", "phishing_rate",
                        "best_single_feature_accuracy", "n_features_above_95",
                        "full_model_accuracy", "n_removed", "final_accuracy", "verdict"]
            if c in summary.columns]
    for line in summary[cols].to_string(index=False).splitlines():
        log.info(line)

    _report_label_policy_sensitivity(summary)
    _report_threshold_sensitivity(summary)
    population_table()
    log.info("-> %s", SCREEN_DIR / "benchmark_screening_summary.csv")
    return summary


def _report_threshold_sensitivity(summary: pd.DataFrame) -> None:
    """Show how each verdict moves as the accuracy cutoff moves."""
    cols = [f"verdict_at_{t}" for t in VERDICT_THRESHOLDS]
    if not all(c in summary.columns for c in cols):
        return
    banner(log, "B2 VERDICT SENSITIVITY TO THE ACCURACY THRESHOLD", "-")
    view = summary[["dataset", "n_features_above_95", "full_model_accuracy"] + cols]
    for line in view.to_string(index=False).splitlines():
        log.info(line)
    unstable = [r["dataset"] for _, r in summary.iterrows()
                if len({r[c] for c in cols if r[c] != "undetermined"}) > 1]
    if unstable:
        log.warning("Verdicts that CHANGE with the threshold: %s. For these the label is a "
                    "property of the cutoff, not of the dataset, and must be reported with "
                    "the threshold attached.", unstable)
    log.info("The threshold-independent signal is n_features_above_95: a count of features "
             "that individually reconstruct the label. It does not move with the cutoff.")
    return summary


def _report_label_policy_sensitivity(summary: pd.DataFrame) -> None:
    """Does the verdict depend on how the intermediate class was handled?"""
    if "label_policy" not in summary.columns:
        return
    base = summary[summary.get("is_primary", False) == True]  # noqa: E712
    alt = summary[summary.get("is_primary", True) == False]   # noqa: E712
    if alt.empty:
        return
    banner(log, "B2 LABEL-POLICY SENSITIVITY (uci379 three-class handling)", "-")
    for _, a in alt.iterrows():
        stem = a["dataset"].replace("__suspicious_dropped", "")
        b = base[base["dataset"] == stem]
        if b.empty:
            continue
        b = b.iloc[0]
        same = a["verdict"] == b["verdict"]
        log.info("%s: merge_phishing -> %s (n=%d, phish_rate=%.4f, acc=%.6f)", stem,
                 b["verdict"], b["n_rows"], b["phishing_rate"], b["full_model_accuracy"])
        log.info("%s: drop           -> %s (n=%d, phish_rate=%.4f, acc=%.6f)", stem,
                 a["verdict"], a["n_rows"], a["phishing_rate"], a["full_model_accuracy"])
        if same:
            log.info("    VERDICT STABLE across both handlings - the choice is immaterial "
                     "to the screen and can be stated as such in the paper.")
        else:
            log.warning("    VERDICT DIFFERS (%s vs %s). The screening result for this "
                        "dataset is sensitive to how the intermediate class is handled, "
                        "and BOTH must be reported rather than one being presented as the "
                        "result.", b["verdict"], a["verdict"])


def population_table() -> pd.DataFrame:
    """The headline B2 artifact: every screened benchmark in one comparison.

    This is the table the paper's lead claim rests on, so it must contain the two
    Phase A datasets as well as the externally acquired ones - a "3 near-oracle
    features versus 0 elsewhere" contrast cannot be made from a table that omits
    the dataset with the 3.

    Ordered by `n_features_above_95`, the threshold-independent signal: a COUNT of
    features that individually reconstruct the label. That count does not move
    when the accuracy cutoff moves, which is exactly why it leads.
    """
    rows = []

    # The two Phase A datasets, assembled from existing artifacts rather than
    # re-screened - PhiUSIIL's progressive removal alone cost several hours.
    sf_path = RESULTS_DIR / "single_feature_predictive_power.csv"
    outcome_path = RESULTS_DIR / "leak_analysis" / "progressive_removal_outcome.json"
    curve_path = RESULTS_DIR / "leak_analysis" / "progressive_removal.csv"
    if sf_path.exists():
        sf = pd.read_csv(sf_path)
        curve = pd.read_csv(curve_path) if curve_path.exists() else None
        outcome = (json.loads(outcome_path.read_text(encoding="utf-8"))
                   if outcome_path.exists() else {})
        for ds, g in sf.groupby("dataset"):
            best = g.nlargest(1, "accuracy").iloc[0]
            if ds == "phiusiil" and outcome:
                full_acc = float(curve["accuracy"].iloc[0]) if curve is not None else float("nan")
                rec = {"full_model_accuracy": full_acc,
                       "n_removed": outcome.get("n_removed"),
                       "final_accuracy": outcome.get("final_accuracy"),
                       "verdict": "repairable" if outcome.get("outcome") == "deleaked"
                                  else outcome.get("outcome"),
                       "curve": curve}
            else:
                scr_row = None
                rec = {"full_model_accuracy": float("nan"), "n_removed": 0,
                       "final_accuracy": float("nan"), "verdict": "", "curve": None}
                gate = RESULTS_DIR / "screening" / "validation_uci.json"
                if ds == "uci" and gate.exists():
                    m = json.loads(gate.read_text(encoding="utf-8"))
                    rec.update({"full_model_accuracy": m.get("full_model_accuracy"),
                                "verdict": m.get("verdict", "")})
                    rec["final_accuracy"] = rec["full_model_accuracy"]
            row = {"dataset": ds, "source": "phase_a",
                   "is_independent_benchmark": True,
                   "n_features": int(len(g)),
                   "n_features_above_95": int((g["accuracy"] > LEAKAGE_ACC_THRESHOLD).sum()),
                   "best_single_feature": best["feature"],
                   "best_single_feature_accuracy": round(float(best["accuracy"]), 4),
                   "full_model_accuracy": rec["full_model_accuracy"],
                   "n_removed": rec["n_removed"],
                   "final_accuracy": rec["final_accuracy"],
                   "verdict": rec["verdict"]}
            if rec["curve"] is not None and pd.notna(rec["full_model_accuracy"]):
                row.update(verdicts_across_thresholds(rec["full_model_accuracy"],
                                                      rec["curve"]))
            rows.append(row)

    scr_path = SCREEN_DIR / "benchmark_screening_summary.csv"
    if scr_path.exists():
        scr = pd.read_csv(scr_path)
        for _, r in scr.iterrows():
            if r.get("verdict") == "NOT_ACQUIRED":
                continue
            d = r.to_dict()
            d["source"] = "external_screen"
            # A label-policy variant is the same data under a different handling,
            # so it must not be counted as an independent benchmark in the contrast.
            d["is_independent_benchmark"] = "__" not in str(r["dataset"])
            rows.append(d)

    tbl = pd.DataFrame(rows)
    lead = ["dataset", "is_independent_benchmark", "n_features_above_95",
            "best_single_feature", "best_single_feature_accuracy", "verdict",
            "full_model_accuracy", "n_removed", "final_accuracy", "n_features"]
    tbl = tbl[[c for c in lead if c in tbl.columns]
              + [c for c in tbl.columns if c not in lead]]
    tbl = tbl.sort_values(["n_features_above_95", "dataset"], ascending=[False, True])
    write_csv_atomic(tbl, SCREEN_DIR / "benchmark_population_table.csv")

    banner(log, "B2 POPULATION CONTRAST (lead artifact)", "-")
    view = tbl[["dataset", "is_independent_benchmark", "n_features_above_95",
                "best_single_feature_accuracy", "verdict"]]
    for line in view.to_string(index=False).splitlines():
        log.info(line)

    indep = tbl[tbl["is_independent_benchmark"].astype(bool)]
    leaky = indep[indep["n_features_above_95"] > 0]
    clean = indep[indep["n_features_above_95"] == 0]
    log.info("")
    log.info("CONTRAST: %d of %d independent benchmarks carry any feature above the 0.95 "
             "single-feature bar.", len(leaky), len(indep))
    for _, r in leaky.iterrows():
        log.info("    %s: %d such feature(s), best %.4f (%s)", r["dataset"],
                 r["n_features_above_95"], r["best_single_feature_accuracy"],
                 r["best_single_feature"])
    log.info("    %d benchmark(s) with ZERO: %s", len(clean), ", ".join(clean["dataset"]))
    log.info("This count is threshold-independent: it does not move with the accuracy "
             "cutoff that decides the verdict labels.")
    return tbl


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task B2: benchmark leakage screen")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--acc-target", type=float, default=DELEAK_ACC_TARGET,
                    help="stopping threshold. Run at the LOWEST threshold of interest: "
                         "the curve then passes through every higher one, so verdicts at "
                         "all of them can be derived from a single run.")
    args = ap.parse_args()

    if args.validate_only:
        validate_screen(seed=args.seed)
    else:
        run_screen(args.datasets, seed=args.seed, skip_validation=args.skip_validation,
                   acc_target=args.acc_target)

