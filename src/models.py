"""Task 4 - the three base learners and the soft-voting ensemble.

All learners are tree ensembles. That is a hard requirement, not a preference:
TreeSHAP computes exact Shapley values only for tree models, and the study's
claims rest on exact attributions rather than KernelSHAP approximations.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_logger  # noqa: E402

log = get_logger("models")

BASE_MODEL_NAMES = ["rf", "xgb", "lgbm"]
ENSEMBLE_NAME = "ensemble"
ALL_MODEL_NAMES = BASE_MODEL_NAMES + [ENSEMBLE_NAME]

DISPLAY_NAMES = {
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "ensemble": "SoftVotingEnsemble",
}


def build_models(seed: int, params: dict | None = None) -> dict:
    """Return {name: unfitted estimator} for the three learners and the ensemble.

    `params` maps a model name ('rf' / 'xgb' / 'lgbm') to a hyperparameter dict,
    normally the tuned best_params_ from Task 5. Unspecified models keep library
    defaults. random_state is always forced to `seed`, overriding anything in
    `params`, so a stale cached parameter file can never reintroduce a fixed seed.
    """
    params = params or {}

    rf = RandomForestClassifier(
        random_state=seed, n_jobs=-1, **_clean(params.get("rf", {})))
    # enable_categorical=False is REQUIRED, not cosmetic. XGBoost >= 3.0 defaults it
    # to True, and shap's TreeExplainer then refuses interventional perturbation with
    # "Categorical split is not yet supported" - even though every feature here is a
    # StandardScaler-scaled float and no categorical split can exist. Declaring it
    # false states the truth about the data rather than suppressing a real error.
    xgb = XGBClassifier(
        random_state=seed, n_jobs=-1, eval_metric="logloss", enable_categorical=False,
        **_clean(params.get("xgb", {})))
    lgbm = LGBMClassifier(
        random_state=seed, n_jobs=-1, verbose=-1, **_clean(params.get("lgbm", {})))

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        n_jobs=-1,
    )
    return {"rf": rf, "xgb": xgb, "lgbm": lgbm, "ensemble": ensemble}


def _clean(p: dict) -> dict:
    """Strip keys that build_models sets itself, so they cannot be double-passed."""
    return {k: v for k, v in p.items()
            if k not in ("random_state", "n_jobs", "eval_metric", "verbose",
                         "enable_categorical")}


def base_estimators_of(ensemble: VotingClassifier) -> dict:
    """Fitted base learners of a fitted VotingClassifier, keyed by name.

    VotingClassifier clones its inputs, so SHAP must be computed on
    `estimators_`, never on the objects handed to the constructor.
    """
    if not hasattr(ensemble, "estimators_"):
        raise ValueError("ensemble is not fitted - no estimators_ attribute")
    names = [n for n, _ in ensemble.estimators]
    return dict(zip(names, ensemble.estimators_))


if __name__ == "__main__":
    m = build_models(seed=42)
    for name, est in m.items():
        log.info("%-9s -> %s", name, est.__class__.__name__)
    log.info("Ensemble members: %s", [n for n, _ in m["ensemble"].estimators])
