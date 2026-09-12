"""Task 5 - per-dataset hyperparameter tuning.

RandomizedSearchCV, n_iter=50, StratifiedKFold(5), scoring='f1', fitted on the
TRAINING split only. Tuning is performed independently for each dataset and the
resulting parameters are never shared across datasets - the whole point of the
study is that the two datasets are structurally different, so borrowing
parameters would confound the comparison.

Tuning is done once per dataset with SEEDS[0] and the resulting parameters are
reused for all five evaluation seeds. The seeds vary the data split and the model
randomness, which is what the variance estimate is meant to capture; re-tuning
per seed would additionally vary the model class and make the variance
uninterpretable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CV_FOLDS,
    N_SEARCH_ITER,
    RESULTS_DIR,
    SEEDS,
    banner,
    get_logger,
    resolve_config,
    set_global_seed,
)
from src.models import BASE_MODEL_NAMES, build_models  # noqa: E402
from src.preprocessing import get_splits  # noqa: E402

log = get_logger("tuning")

SEARCH_SPACES = {
    "rf": {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "xgb": {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "lgbm": {
        "n_estimators": [100, 300, 500],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [20, 50],
    },
}


def _params_path(key: str) -> Path:
    """Tuned-parameter file for a dataset (Phase A) or an experiment config (A4)."""
    return RESULTS_DIR / f"best_params_{key}.json"


# Configs whose training data is byte-for-byte what the Phase A per-dataset search
# already ran on, so re-running the search would burn hours to reproduce a known
# answer. Their parameter files are seeded from the dataset-level result instead.
# Any config NOT listed here is tuned from scratch.
CONFIG_EQUIVALENT_TO_DATASET = {
    "phiusiil_full": "phiusiil",   # dedup=True, all 50 features - identical to Phase A
    "uci_dedup": "uci",            # dedup=True, all 30 features - identical to Phase A
}


class _ProgressScorer:
    """f1 scorer that logs progress and a wall-clock ETA as fits complete.

    RandomizedSearchCV offers no progress callback, so the scorer doubles as one:
    it is invoked exactly once per (candidate, fold) fit. Requires n_jobs=1 on the
    search itself, which is also the memory-safe choice on the 165k-row dataset
    (parallelism is delegated to the estimators instead).
    """

    def __init__(self, total_fits: int, tag: str):
        self.total = total_fits
        self.tag = tag
        self.done = 0
        self.start = time.perf_counter()

    def __call__(self, estimator, X, y):
        score = f1_score(y, estimator.predict(X), zero_division=0)
        self.done += 1
        elapsed = time.perf_counter() - self.start
        rate = elapsed / self.done
        remaining = rate * (self.total - self.done)
        if self.done == 1 or self.done % CV_FOLDS == 0 or self.done == self.total:
            log.info("    %s  fit %3d/%3d  f1=%.4f  elapsed=%6.1fs  ETA=%6.1fs",
                     self.tag, self.done, self.total, score, elapsed, remaining)
        return score


def tune_model(dataset: str, model_name: str, seed: int | None = None,
               force: bool = False, config=None) -> dict:
    """Tune one model on one dataset or config. Returns the best parameter dict.

    `dataset` doubles as the cache key: a plain dataset name for the Phase A
    searches, or a config name when `config` is supplied.
    """
    seed = seed if seed is not None else SEEDS[0]
    cached = load_best_params(dataset)
    if model_name in cached and not force:
        log.info("[%s/%s] using cached best params: %s", dataset, model_name,
                 cached[model_name])
        return cached[model_name]

    set_global_seed(seed)
    if config is not None:
        splits = get_splits(config.dataset, seed, dedup=config.dedup, features=config.features)
    else:
        splits = get_splits(dataset, seed)
    estimator = build_models(seed)[model_name]
    space = SEARCH_SPACES[model_name]

    total_fits = N_SEARCH_ITER * CV_FOLDS
    log.info("[%s/%s] RandomizedSearchCV: n_iter=%d cv=%d -> %d fits on %d training rows",
             dataset, model_name, N_SEARCH_ITER, CV_FOLDS, total_fits, len(splits.y_train))

    scorer = _ProgressScorer(total_fits, f"[{dataset}/{model_name}]")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=N_SEARCH_ITER,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed),
        scoring=scorer,
        random_state=seed,
        n_jobs=1,          # estimators use n_jobs=-1; nesting would oversubscribe
        refit=False,       # E1 refits per seed anyway; refitting here wastes time
        error_score="raise",
    )

    t0 = time.perf_counter()
    search.fit(splits.X_train, splits.y_train)
    wall = time.perf_counter() - t0

    best = dict(search.best_params_)
    log.info("[%s/%s] best f1=%.5f in %.1fs (%.1f min)",
             dataset, model_name, search.best_score_, wall, wall / 60)
    log.info("[%s/%s] best params: %s", dataset, model_name, best)

    cached[model_name] = best
    meta = cached.setdefault("_meta", {})
    meta[model_name] = {
        "best_cv_f1": float(search.best_score_),
        "wall_clock_seconds": round(wall, 2),
        "n_iter": N_SEARCH_ITER,
        "cv_folds": CV_FOLDS,
        "scoring": "f1",
        "tuning_seed": seed,
        "n_train_rows": int(len(splits.y_train)),
    }
    _params_path(dataset).write_text(json.dumps(cached, indent=2, default=_json_safe),
                                     encoding="utf-8")
    return best


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def load_best_params(dataset: str) -> dict:
    """Load cached tuned parameters; {} when no tuning has been run."""
    path = _params_path(dataset)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_params_for_models(dataset: str) -> dict:
    """Tuned parameters keyed by model name, stripped of bookkeeping entries."""
    return {k: v for k, v in load_best_params(dataset).items() if not k.startswith("_")}


def tune_dataset(dataset: str, seed: int | None = None, force: bool = False) -> dict:
    """Tune all three base learners for one dataset."""
    banner(log, f"TASK 5: HYPERPARAMETER TUNING - {dataset}")
    t0 = time.perf_counter()
    out = {}
    for m in BASE_MODEL_NAMES:
        out[m] = tune_model(dataset, m, seed=seed, force=force)
    log.info("[%s] all searches complete in %.1f min", dataset, (time.perf_counter() - t0) / 60)
    return out


# --------------------------------------------------------------------------- #
# Task A4 - per-CONFIG tuning
# --------------------------------------------------------------------------- #
def _seed_params_from_dataset(config_name: str) -> bool:
    """Copy a dataset-level result into a config-level file when they are identical."""
    src_key = CONFIG_EQUIVALENT_TO_DATASET.get(config_name)
    if src_key is None:
        return False
    dst = _params_path(config_name)
    if dst.exists():
        return True
    src = _params_path(src_key)
    if not src.exists():
        return False
    payload = json.loads(src.read_text(encoding="utf-8"))
    meta = payload.setdefault("_meta", {})
    meta["_provenance"] = (
        f"copied from best_params_{src_key}.json: config {config_name!r} trains on exactly "
        f"the data the Phase A {src_key!r} search used (same de-duplication policy, same "
        f"feature set), so the search result carries over unchanged."
    )
    dst.write_text(json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8")
    log.info("[%s] seeded tuned parameters from best_params_%s.json (identical training "
             "data - no re-search needed)", config_name, src_key)
    return True


def tune_config(config, seed: int | None = None, force: bool = False) -> dict:
    """Tune all three base learners for one experiment config (Task A4).

    Every config is tuned against its OWN training split. A config whose training
    data is identical to a Phase A per-dataset search inherits that result rather
    than repeating it; everything else gets a fresh search, because neither a
    changed feature space (phiusiil_leakfree) nor a changed row set (uci_full,
    which carries 11,055 rows rather than the de-duplicated 5,849) is guaranteed
    to keep the same optimum.
    """
    cfg = resolve_config(config) if isinstance(config, str) else config
    banner(log, f"TASK A4: HYPERPARAMETER TUNING - config {cfg.name}")
    log.info("[%s] %s", cfg.name, cfg.describe())

    if not force and _seed_params_from_dataset(cfg.name):
        return get_params_for_models(cfg.name)

    seed = seed if seed is not None else SEEDS[0]
    t0 = time.perf_counter()
    out = {}
    for m in BASE_MODEL_NAMES:
        out[m] = tune_model(cfg.name, m, seed=seed, force=force, config=cfg)
    log.info("[%s] all searches complete in %.1f min", cfg.name, (time.perf_counter() - t0) / 60)
    return out


def get_params_for_config(config) -> dict:
    """Tuned parameters for a config, falling back to its dataset's Phase A result."""
    cfg = resolve_config(config) if isinstance(config, str) else config
    params = get_params_for_models(cfg.name)
    if params:
        return params
    if _seed_params_from_dataset(cfg.name):
        return get_params_for_models(cfg.name)
    log.warning("[%s] no config-level tuned parameters; falling back to the dataset-level "
                "file for %r.", cfg.name, cfg.dataset)
    return get_params_for_models(cfg.dataset)


if __name__ == "__main__":
    import argparse

    from config import CONFIG_NAMES

    ap = argparse.ArgumentParser(description="Task 5 / A4: hyperparameter tuning")
    ap.add_argument("--dataset", default="both", choices=["phiusiil", "uci", "both"])
    ap.add_argument("--config", default=None, choices=[*CONFIG_NAMES, "all"],
                    help="tune per experiment config (Task A4) instead of per dataset")
    ap.add_argument("--model", default="all", choices=["all", *BASE_MODEL_NAMES])
    ap.add_argument("--force", action="store_true", help="re-tune, ignoring the cache")
    args = ap.parse_args()

    if args.config:
        from config import available_configs
        names = available_configs() if args.config == "all" else [args.config]
        for c in names:
            tune_config(c, force=args.force)
    else:
        datasets = ["phiusiil", "uci"] if args.dataset == "both" else [args.dataset]
        for d in datasets:
            if args.model == "all":
                tune_dataset(d, force=args.force)
            else:
                tune_model(d, args.model, force=args.force)
