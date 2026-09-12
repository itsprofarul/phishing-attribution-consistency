"""Global configuration for Phase A of the SHAP cross-dataset consistency study.

Every source of randomness in this project must be seeded from the values here.
There is no unseeded randomness anywhere in the codebase.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Experimental constants (frozen for the paper)
# --------------------------------------------------------------------------- #
SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_BOOTSTRAP = 1000
SHAP_SAMPLE_SIZE = 10000
CV_FOLDS = 5
N_SEARCH_ITER = 50

# Convergence check sample sizes (Task 7)
SHAP_CONVERGENCE_SIZES = [5000, 10000, 20000]
SHAP_CONVERGENCE_TAU_THRESHOLD = 0.95

# Jaccard cut-offs used by the consistency metrics (Task 8)
JACCARD_KS = [5, 10, 15]

# Permutation test size (Task 8)
N_PERMUTATIONS = 10000

# Single-feature accuracy above which a feature is flagged as leakage (Task 3)
LEAKAGE_ACC_THRESHOLD = 0.95

# E2 interpretation thresholds (Task 9)
TAU_STABLE_THRESHOLD = 0.85
TAU_UNSTABLE_THRESHOLD = 0.70

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
UCI_IDS = {
    "phiusiil": 967,   # PhiUSIIL Phishing URL Dataset, 235,795 x 54
    "uci": 327,        # Phishing Websites, 11,055 x 30
}
DATASETS = list(UCI_IDS)

# Published class balances used by the harmonisation assertion (Task 2.1).
# A silent label inversion would invalidate the whole study, so these are hard
# gates rather than warnings.
EXPECTED_PHISHING_RATE = {
    "phiusiil": 0.4281,   # 100,945 of 235,795
    "uci": 0.443,
}
EXPECTED_RATE_TOLERANCE = 0.01   # absolute tolerance on the phishing fraction

# Published shapes, quoted as (rows, n_features) EXCLUDING the label column.
# The cached parquet holds features + label, so it has one column more.
EXPECTED_SHAPE = {
    "phiusiil": (235795, 54),
    "uci": (11055, 30),
}

# String identifiers in PhiUSIIL: not predictive features, dropped in Task 2.2.
#
# FILENAME is listed here because the original brief specified it, but it never
# reaches the frame: UCI assigns it role "Other" (not "Feature"/"ID"), so
# ucimlrepo excludes it from data.features and data.ids is None. Only the other
# four are actually present and actually dropped. See results/feature_inventory.csv
# and Task A6 - this is the whole of the 50-vs-49 feature-count discrepancy.
PHIUSIIL_DROP_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

# Feature suspected of being near-deterministic with the PhiUSIIL label (Task 3).
SUSPECTED_LEAKY_FEATURE = "URLSimilarityIndex"

# --------------------------------------------------------------------------- #
# De-duplication policy (Task A3)
# --------------------------------------------------------------------------- #
# UCI's 30 features are all ternary-coded {-1, 0, 1}. Two websites sharing an
# identical 30-dimensional ternary vector is an expected COLLISION between
# distinct sites that happen to share a coarse feature profile, not a duplicated
# record. Removing them discards real observations and shifts the class balance
# from the published 44.3% to 51.6%, breaking comparability with every prior
# paper on this dataset.
#
# PhiUSIIL carries continuous features (URLCharProb, TLDLegitimateProb, ...),
# where an exact tie across all 50 columns is vanishingly unlikely by chance and
# therefore much more plausibly a genuine repeated record.
#
# Duplicate COUNTS are reported for both datasets regardless of this setting -
# the duplicate rate is a dataset property the paper reports either way.
DEDUPLICATE = {"phiusiil": True, "uci": False}

# --------------------------------------------------------------------------- #
# Leak-free PhiUSIIL feature set (Task A1.4)
# --------------------------------------------------------------------------- #
# Populated by src/leak_analysis.py from the A1.3 progressive-removal experiment.
# None means "not yet determined, or PhiUSIIL could not be de-leaked by feature
# removal" - in which case the phiusiil_leakfree config is unavailable and A4/A5
# must not be run for it.
#
# The block between the markers below is REWRITTEN IN PLACE by
# src/leak_analysis.py --write-config. Do not edit it by hand; edit the analysis
# and re-run it, so that the recorded evidence always matches the experiment that
# produced the feature set.
# >>>BEGIN GENERATED: PHIUSIIL_LEAKFREE (Task A1.4)>>>
# Evidence: A1.3 progressive removal (seed 42): removing URLSimilarityIndex, LineOfCode, IsHTTPS, NoOfSelfRef, NoOfExternalRef, NoOfImage, HasSocialNet, NoOfJS, NoOfCSS, LargestLineLength, LetterRatioInURL, NoOfOtherSpecialCharsInURL, NoOfLettersInURL, HasCopyrightInfo dropped held-out ensemble accuracy from 1.000000 to 0.989191.
# Removed as leaking: URLSimilarityIndex, LineOfCode, IsHTTPS, NoOfSelfRef, NoOfExternalRef, NoOfImage, HasSocialNet, NoOfJS, NoOfCSS, LargestLineLength, LetterRatioInURL, NoOfOtherSpecialCharsInURL, NoOfLettersInURL, HasCopyrightInfo
# Retained: 36 of 50 features.
PHIUSIIL_LEAKFREE_FEATURES: list[str] | None = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "SpacialCharRatioInURL",
    "HasTitle",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "NoOfEmptyRef",
]
PHIUSIIL_LEAKFREE_EVIDENCE: str | None = (
    "A1.3 progressive removal (seed 42): removing URLSimilarityIndex, LineOfCode, IsHTTPS, NoOfSelfRef, NoOfExternalRef, NoOfImage, HasSocialNet, NoOfJS, NoOfCSS, LargestLineLength, LetterRatioInURL, NoOfOtherSpecialCharsInURL, NoOfLettersInURL, HasCopyrightInfo dropped held-out ensemble accuracy from 1.000000 to 0.989191."
)
# <<<END GENERATED<<<

# --------------------------------------------------------------------------- #
# Experiment configurations (Tasks A4/A5)
# --------------------------------------------------------------------------- #
# A "config" is a (dataset, de-duplication policy, feature subset) triple. It is
# the unit E1/E2 are re-run over, so that the leak-controlled and full variants
# are directly comparable within one table.
EXPERIMENT_CONFIGS = {
    "uci_full":          {"dataset": "uci",      "dedup": False, "features": None},
    "uci_dedup":         {"dataset": "uci",      "dedup": True,  "features": None},
    "phiusiil_full":     {"dataset": "phiusiil", "dedup": True,  "features": None},
    "phiusiil_leakfree": {"dataset": "phiusiil", "dedup": True,  "features": "PHIUSIIL_LEAKFREE_FEATURES"},
}
CONFIG_NAMES = list(EXPERIMENT_CONFIGS)


@dataclass(frozen=True)
class ExperimentConfig:
    """One (dataset, de-duplication policy, feature subset) cell of A4/A5."""

    name: str
    dataset: str
    dedup: bool
    features: tuple[str, ...] | None   # None = use every retained feature

    @property
    def feature_tag(self) -> str:
        """Short stable tag distinguishing feature subsets in cache filenames."""
        if self.features is None:
            return "all"
        digest = hashlib.sha1("|".join(self.features).encode()).hexdigest()[:8]
        return f"n{len(self.features)}_{digest}"

    def describe(self) -> str:
        feats = "all features" if self.features is None else f"{len(self.features)} features"
        return (f"{self.name}: dataset={self.dataset} dedup={self.dedup} {feats}")


class ConfigUnavailableError(RuntimeError):
    """Raised when a config depends on a feature set that has not been determined."""


def resolve_config(name: str) -> ExperimentConfig:
    """Materialise a named config, late-binding any feature-set reference.

    `features` may name a module-level constant (e.g. PHIUSIIL_LEAKFREE_FEATURES)
    rather than holding a literal list, so that a config defined here picks up a
    feature set determined later by the leak analysis without an import cycle.
    """
    if name not in EXPERIMENT_CONFIGS:
        raise KeyError(f"Unknown config {name!r}; expected one of {CONFIG_NAMES}")
    spec = EXPERIMENT_CONFIGS[name]
    feats = spec["features"]
    if isinstance(feats, str):
        referenced = globals().get(feats)
        if referenced is None:
            raise ConfigUnavailableError(
                f"Config {name!r} requires {feats}, which is None. Run "
                f"src/leak_analysis.py (Task A1) to determine it first; if the "
                f"progressive-removal experiment failed to de-leak the dataset it "
                f"stays None by design and this config must not be run."
            )
        feats = tuple(referenced)
    elif feats is not None:
        feats = tuple(feats)
    return ExperimentConfig(name=name, dataset=spec["dataset"],
                            dedup=bool(spec["dedup"]), features=feats)


def available_configs() -> list[str]:
    """Config names that can currently be resolved (leak-free may be pending)."""
    out = []
    for n in CONFIG_NAMES:
        try:
            resolve_config(n)
        except ConfigUnavailableError:
            continue
        out.append(n)
    return out

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
SHAP_CACHE_DIR = CACHE_DIR / "shap"
RESULTS_DIR = PROJECT_ROOT / "results"
LEAK_DIR = RESULTS_DIR / "leak_analysis"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOG_PATH = RESULTS_DIR / "phase_a.log"

for _d in (RAW_DIR, CACHE_DIR, SHAP_CACHE_DIR, RESULTS_DIR, LEAK_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def write_csv_atomic(df, path: Path, **kwargs) -> None:
    """DataFrame.to_csv via a temp file and an atomic rename.

    These files are rewritten after every iteration/seed of a multi-hour run, so
    a hard kill part-way through a write would otherwise leave a truncated table
    that looks like valid partial results. Renaming into place guarantees the
    file on disk is always a complete snapshot of some prefix of the run.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def set_global_seed(seed: int) -> None:
    """Seed every global RNG. Per-model random_state is passed explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Logging: console + results/phase_a.log
# --------------------------------------------------------------------------- #
_LOG_CONFIGURED = False


def get_logger(name: str = "phase_a") -> logging.Logger:
    """Return a logger writing to both stdout and results/phase_a.log."""
    global _LOG_CONFIGURED
    root = logging.getLogger("phase_a")
    if not _LOG_CONFIGURED:
        root.setLevel(logging.INFO)
        root.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        root.addHandler(stream)

        fileh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
        _LOG_CONFIGURED = True

    if name in ("phase_a", "", None):
        return root
    return root.getChild(name)


def banner(logger: logging.Logger, text: str, char: str = "=") -> None:
    """Section header, so the log is skimmable after a multi-hour run."""
    line = char * 78
    logger.info(line)
    logger.info(text)
    logger.info(line)


def library_versions() -> dict[str, str]:
    """Resolved versions of every library whose output could shift results."""
    import importlib

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in (
        "numpy", "pandas", "sklearn", "xgboost", "lightgbm",
        "shap", "scipy", "matplotlib", "seaborn", "pyarrow", "ucimlrepo",
    ):
        try:
            versions[mod] = importlib.import_module(mod).__version__
        except Exception as exc:  # pragma: no cover - diagnostics only
            versions[mod] = f"<unavailable: {exc}>"
    return versions


if __name__ == "__main__":
    log = get_logger()
    banner(log, "CONFIGURATION")
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Seeds        : %s", SEEDS)
    log.info("Split        : train=%.2f val=%.2f test=%.2f",
             1 - TEST_SIZE - VAL_SIZE, VAL_SIZE, TEST_SIZE)
    for k, v in library_versions().items():
        log.info("%-12s %s", k, v)
