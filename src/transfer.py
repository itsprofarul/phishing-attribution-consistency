"""Task B1 - zero-shot cross-dataset transfer.

The decisive experiment of Phase B. Genuine phishing-detection signal should
transfer across independently collected datasets; a collection artifact should
not. PhiUSIIL scores 1.0000 in-distribution, so the question is what it scores on
UCI having never seen it.

THE ENCODING PROBLEM (the central methodological decision here)
---------------------------------------------------------------
The brief asks for a "conceptual intersection" of the two feature schemas, but
the two datasets do not merely name features differently - they ENCODE them
differently:

    PhiUSIIL   continuous / counts   URLLength in [13, 6097], NoOfExternalRef in [0, 27516]
    UCI        ternary               every feature in {-1, 0, +1}

Training on PhiUSIIL's raw `URLLength` and testing on UCI's `url_length` would
therefore collapse to chance for a trivial reason - the test features occupy a
range the model never saw - and that collapse would say nothing whatsoever about
leakage. Reporting it as evidence of leakage would be a serious error, because
the experiment would have been rigged by its own preprocessing.

So both datasets are projected onto a SHARED TERNARY SCHEMA, using the binning
rules documented for the UCI dataset (Mohammad, Thabtah & McCluskey, "Phishing
Websites Features"). UCI features are already in that encoding and pass through
unchanged; PhiUSIIL's continuous features are discretised with the same
thresholds. Only then is transfer meaningful: both sides speak the same language,
and any remaining collapse is about signal, not about scale.

POLARITY, verified empirically rather than assumed (see FEATURE_MAP rationale):
UCI uses +1 = legitimate-indicating, 0 = suspicious, -1 = phishing-indicating.
This was confirmed by the per-value phishing rate on all nine mapped features
before any rule was written.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    RESULTS_DIR,
    SEEDS,
    banner,
    get_logger,
    set_global_seed,
    write_csv_atomic,
)
from src.preprocessing import get_xy  # noqa: E402

log = get_logger("transfer")

TRANSFER_DIR = RESULTS_DIR / "transfer"
TRANSFER_DIR.mkdir(parents=True, exist_ok=True)

# UCI ternary convention, confirmed empirically.
LEGIT, SUSPICIOUS, PHISH = 1, 0, -1


# --------------------------------------------------------------------------- #
# B1.1 Conceptual feature mapping
# --------------------------------------------------------------------------- #
@dataclass
class ConceptMap:
    """One shared concept, with the rule projecting PhiUSIIL onto UCI's encoding."""

    concept: str
    uci_column: str
    phiusiil_source: tuple[str, ...]
    rule: Callable[[pd.DataFrame], pd.Series]
    rationale: str
    caveat: str = ""


def _bin_url_length(df: pd.DataFrame) -> pd.Series:
    # UCI rule: <54 legitimate, 54-75 suspicious, >75 phishing.
    v = df["URLLength"]
    return pd.Series(np.where(v < 54, LEGIT, np.where(v <= 75, SUSPICIOUS, PHISH)), index=df.index)


def _bin_subdomain(df: pd.DataFrame) -> pd.Series:
    # UCI rule: <=1 dot legitimate, 2 suspicious, >2 phishing.
    v = df["NoOfSubDomain"]
    return pd.Series(np.where(v <= 1, LEGIT, np.where(v == 2, SUSPICIOUS, PHISH)), index=df.index)


def _bin_anchor_external(df: pd.DataFrame) -> pd.Series:
    # UCI url_of_anchor: share of anchors pointing off-domain.
    # <31% legitimate, 31-67% suspicious, >67% phishing.
    ext = df["NoOfExternalRef"].astype(float)
    total = ext + df["NoOfSelfRef"].astype(float) + df["NoOfEmptyRef"].astype(float)
    ratio = np.divide(ext, total, out=np.zeros(len(df)), where=total > 0)
    return pd.Series(np.where(ratio < 0.31, LEGIT, np.where(ratio <= 0.67, SUSPICIOUS, PHISH)),
                     index=df.index)


FEATURE_MAP: list[ConceptMap] = [
    ConceptMap(
        "ip_in_domain", "having_ip_address", ("IsDomainIP",),
        lambda d: pd.Series(np.where(d["IsDomainIP"] == 1, PHISH, LEGIT), index=d.index),
        "Both encode 'the host is a bare IP address rather than a domain name'. "
        "Direct binary correspondence."),
    ConceptMap(
        "url_length", "url_length", ("URLLength",), _bin_url_length,
        "Same quantity (characters in the URL); PhiUSIIL stores it raw, UCI stores it "
        "binned. Discretised with UCI's published 54/75 thresholds."),
    ConceptMap(
        "subdomain_depth", "having_sub_domain", ("NoOfSubDomain",), _bin_subdomain,
        "Both encode subdomain nesting depth. UCI counts dots after stripping www and "
        "ccTLD; PhiUSIIL counts subdomains directly, so the counts align up to the "
        "stripping convention.",
        caveat="UCI's dot-count and PhiUSIIL's subdomain count can differ by one for "
               "hosts with a ccTLD (e.g. .co.uk)."),
    ConceptMap(
        "ssl_state", "sslfinal_state", ("IsHTTPS",),
        lambda d: pd.Series(np.where(d["IsHTTPS"] == 1, LEGIT, PHISH), index=d.index),
        "Closest available correspondence to UCI's SSL assessment: whether the page is "
        "served over HTTPS at all, which is the dominant component of sslfinal_state.",
        caveat="IMPERFECT. UCI's sslfinal_state additionally requires a trusted issuer "
               "and a certificate at least a year old, and has a 'suspicious' state for "
               "https-with-untrusted-issuer. PhiUSIIL records only the scheme, so the "
               "suspicious state is unreachable from the PhiUSIIL side."),
    ConceptMap(
        "iframe", "iframe", ("NoOfiFrame",),
        lambda d: pd.Series(np.where(d["NoOfiFrame"] == 0, LEGIT, PHISH), index=d.index),
        "Both encode iframe usage on the page.",
        caveat="UCI targets specifically INVISIBLE iframes; PhiUSIIL counts all iframes. "
               "Note also that UCI's iframe feature carries no marginal signal at all "
               "(phishing rate 0.44 at both of its values), so this concept contributes "
               "nothing on the UCI side regardless of the mapping quality."),
    ConceptMap(
        "popup", "popupwindow", ("NoOfPopup",),
        lambda d: pd.Series(np.where(d["NoOfPopup"] == 0, LEGIT, PHISH), index=d.index),
        "Both encode popup-window usage.",
        caveat="UCI targets popups containing a text field; PhiUSIIL counts all popups. "
               "As with iframe, UCI's popupwindow has phishing rate 0.44 at both values - "
               "zero marginal signal on the UCI side."),
    ConceptMap(
        "url_redirect", "redirect", ("NoOfURLRedirect",),
        lambda d: pd.Series(np.where(d["NoOfURLRedirect"] >= 1, 1, 0), index=d.index),
        "Both encode redirection on the way to the page. UCI's `redirect` takes only "
        "{0, 1} in the delivered data, so the PhiUSIIL side is mapped onto that same "
        "{0, 1} domain rather than the usual ternary one.",
        caveat="UCI's published rule describes a 3-way split on redirect count, but the "
               "delivered column is binary; the mapping follows the DATA, not the paper."),
    ConceptMap(
        "form_handler", "sfh", ("HasExternalFormSubmit",),
        lambda d: pd.Series(np.where(d["HasExternalFormSubmit"] == 1, PHISH, LEGIT),
                            index=d.index),
        "Both encode whether a form posts to a different domain than the page - UCI's "
        "Server Form Handler check and PhiUSIIL's external-form-submit flag.",
        caveat="UCI's 'suspicious' state (blank/about:blank handler) has no PhiUSIIL "
               "counterpart and is unreachable from the PhiUSIIL side."),
    ConceptMap(
        "anchor_external_ratio", "url_of_anchor",
        ("NoOfExternalRef", "NoOfSelfRef", "NoOfEmptyRef"), _bin_anchor_external,
        "UCI's url_of_anchor is the share of <a> hrefs pointing off-domain. PhiUSIIL "
        "provides the underlying counts (external / self / empty references), so the "
        "ratio is reconstructed directly and binned with UCI's 31%/67% thresholds."),
]

# Recorded so the paper can state exactly what was discarded and why, rather than
# leaving the shared space unexplained. "Do not force matches" - these are the
# concepts where no defensible correspondence exists.
UNMAPPED_UCI = {
    "shortining_service": "URL-shortener detection; PhiUSIIL has no shortener feature.",
    "having_at_symbol": "presence of '@' in the URL; PhiUSIIL counts only aggregate "
                        "special characters, with no per-character breakdown.",
    "double_slash_redirecting": "'//' position in the path; no PhiUSIIL equivalent.",
    "prefix_suffix": "hyphen in the domain; PhiUSIIL has no hyphen feature.",
    "domain_registration_length": "WHOIS registration length; PhiUSIIL has no WHOIS data.",
    "favicon": "favicon loaded from an EXTERNAL domain. PhiUSIIL's HasFavicon records "
               "presence only, which is a different concept - mapping them would force "
               "a match between 'has a favicon' and 'favicon is foreign'.",
    "port": "non-standard port; no PhiUSIIL equivalent.",
    "https_token": "the literal token 'https' inside the domain name (a deception "
                   "pattern), NOT whether HTTPS is used. PhiUSIIL's IsHTTPS is the "
                   "latter and is already mapped to sslfinal_state.",
    "request_url": "share of loaded OBJECTS (images/scripts) from an external domain. "
                   "PhiUSIIL has NoOfImage/NoOfCSS/NoOfJS but no internal/external "
                   "split for them, so the ratio cannot be reconstructed.",
    "links_in_tags": "share of links inside meta/script/link tags; PhiUSIIL has no "
                     "tag-scoped link counts.",
    "submitting_to_email": "mailto: form action; no PhiUSIIL equivalent.",
    "abnormal_url": "WHOIS hostname consistency; no PhiUSIIL WHOIS data.",
    "on_mouseover": "JavaScript status-bar manipulation; PhiUSIIL records no JS behaviour.",
    "rightclick": "right-click disabling; PhiUSIIL records no JS behaviour.",
    "age_of_domain": "WHOIS domain age; no PhiUSIIL WHOIS data.",
    "dnsrecord": "DNS record existence; no PhiUSIIL DNS data.",
    "web_traffic": "Alexa traffic rank; external reputation, absent from PhiUSIIL.",
    "page_rank": "PageRank; external reputation, absent from PhiUSIIL.",
    "google_index": "Google index membership; external reputation, absent from PhiUSIIL.",
    "links_pointing_to_page": "INBOUND link count. PhiUSIIL's NoOfExternalRef is "
                              "OUTBOUND; mapping them would invert the direction of the "
                              "relationship.",
    "statistical_report": "membership of a published blacklist; external reputation.",
}


def mapping_table(with_polarity: bool = True) -> pd.DataFrame:
    """The B1.1 / t11 supplementary table: every mapped and unmapped feature.

    The per-concept polarity is merged in by default rather than left in a
    separate file. The headline directional result - that the strongest shared
    concept carries OPPOSITE signs on the two datasets - belongs in the same table
    a reader consults to see what was mapped, so it cannot be dropped from the
    exports by accident.
    """
    rows = []
    for m in FEATURE_MAP:
        rows.append({
            "concept": m.concept, "status": "mapped",
            "uci_feature": m.uci_column,
            "phiusiil_source": " + ".join(m.phiusiil_source),
            "rationale": m.rationale, "caveat": m.caveat,
        })
    for col, why in UNMAPPED_UCI.items():
        rows.append({"concept": "", "status": "unmapped_uci", "uci_feature": col,
                     "phiusiil_source": "", "rationale": why, "caveat": ""})

    mapped_p = {c for m in FEATURE_MAP for c in m.phiusiil_source}
    Xp, _ = get_xy("phiusiil")
    for col in Xp.columns:
        if col in mapped_p:
            continue
        rows.append({"concept": "", "status": "unmapped_phiusiil", "uci_feature": "",
                     "phiusiil_source": col,
                     "rationale": "no defensible UCI counterpart (UCI has no comparable "
                                  "page-content/URL-composition feature)", "caveat": ""})
    tbl = pd.DataFrame(rows)
    if with_polarity:
        pol = polarity_agreement()[
            ["concept", "rho_phiusiil", "rho_uci", "direction_agrees", "status"]
        ].rename(columns={"status": "polarity_status"})
        tbl = tbl.merge(pol, on="concept", how="left")
    return tbl


def directional_inconsistency_table() -> pd.DataFrame:
    """Headline directional result, ranked by how much the disagreement matters.

    A sign flip on a feature nothing relies on is a curiosity; a sign flip on the
    single strongest shared concept is a finding. `importance_weight` is the
    larger of the two |rho| values, so the table sorts the inversions that matter
    to the top rather than listing them alphabetically.
    """
    pol = polarity_agreement().copy()
    pol["importance_weight"] = pol[["abs_rho_phiusiil", "abs_rho_uci"]].max(axis=1)
    pol["headline"] = (~pol["direction_agrees"]) & (pol["importance_weight"] >= 0.5)
    return pol.sort_values(["headline", "importance_weight"], ascending=False).reset_index(drop=True)


def to_shared_schema(dataset: str, dedup: bool | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Project one dataset onto the shared ternary schema."""
    X, y = get_xy(dataset, dedup=dedup)
    out = pd.DataFrame(index=X.index)
    if dataset == "uci":
        for m in FEATURE_MAP:
            out[m.concept] = X[m.uci_column].astype(int)
    elif dataset == "phiusiil":
        for m in FEATURE_MAP:
            out[m.concept] = m.rule(X).astype(int)
    else:
        raise KeyError(f"no shared-schema projection defined for {dataset!r}")
    return out, y


def describe_shared_schema() -> pd.DataFrame:
    """Per-concept value distribution and phishing rate on BOTH datasets.

    This is the sanity check that the projection actually aligns the two sides: a
    concept whose value distribution is wildly different across datasets, or whose
    polarity flips, cannot support transfer and must be reported as such.
    """
    rows = []
    for ds in ("phiusiil", "uci"):
        X, y = to_shared_schema(ds, dedup=False if ds == "uci" else None)
        for c in X.columns:
            for v in sorted(X[c].unique()):
                mask = X[c] == v
                rows.append({"dataset": ds, "concept": c, "value": int(v),
                             "n": int(mask.sum()), "share": float(mask.mean()),
                             "phishing_rate": float(y[mask].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# B1.2 The four evaluations
# --------------------------------------------------------------------------- #
TRANSFER_METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]


def _shared_splits(dataset: str, seed: int):
    """70/15/15 split of the shared-schema projection, matching the Phase A protocol."""
    from sklearn.model_selection import train_test_split

    from config import TEST_SIZE, VAL_SIZE

    X, y = to_shared_schema(dataset, dedup=False if dataset == "uci" else None)
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy()
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        Xv, yv, test_size=TEST_SIZE, stratify=yv, random_state=seed, shuffle=True)
    X_tr, _, y_tr, _ = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_tmp,
        random_state=seed, shuffle=True)
    return (X_tr, y_tr), (X_te, y_te), list(X.columns)


def run_transfer_matrix(seeds: list[int] | None = None) -> pd.DataFrame:
    """All four train/test combinations, `seeds` repetitions each.

    Every cell is evaluated on the TARGET'S OWN TEST SPLIT, so the in-distribution
    and cross-distribution rows for a given target are scored on identical rows.
    That makes the contrast a property of the training source alone, and it lets
    B1.3 compare attributions computed on exactly the same instances.
    """
    from sklearn.preprocessing import StandardScaler

    from src.evaluate import score_predictions
    from src.models import ENSEMBLE_NAME, build_models
    from src.tuning import get_params_for_models

    seeds = seeds or SEEDS
    banner(log, "B1.2 ZERO-SHOT TRANSFER MATRIX")

    rows = []
    for seed in seeds:
        set_global_seed(seed)
        prepared = {}
        for ds in ("phiusiil", "uci"):
            (X_tr, y_tr), (X_te, y_te), names = _shared_splits(ds, seed)
            scaler = StandardScaler().fit(X_tr)
            ens = build_models(seed, get_params_for_models(ds))[ENSEMBLE_NAME]
            t0 = time.perf_counter()
            ens.fit(scaler.transform(X_tr), y_tr)
            prepared[ds] = {"model": ens, "scaler": scaler, "names": names,
                            "X_te": X_te, "y_te": y_te, "fit_s": time.perf_counter() - t0,
                            "n_train": len(y_tr)}

        for src in ("phiusiil", "uci"):
            for tgt in ("phiusiil", "uci"):
                s, t = prepared[src], prepared[tgt]
                # The source's OWN scaler is applied to the target: a zero-shot
                # model has no access to target statistics, so refitting the
                # scaler on the target would leak distributional information.
                Xt = s["scaler"].transform(t["X_te"])
                y_pred = s["model"].predict(Xt)
                y_proba = s["model"].predict_proba(Xt)[:, 1]
                scores = score_predictions(t["y_te"], y_pred, y_proba)
                majority = float(max(t["y_te"].mean(), 1 - t["y_te"].mean()))
                rows.append({
                    "train": src, "test": tgt,
                    "direction": f"{src}->{tgt}",
                    "in_distribution": src == tgt,
                    "seed": seed, **scores,
                    "majority_class_accuracy": majority,
                    "accuracy_above_majority": scores["accuracy"] - majority,
                    "n_train": s["n_train"], "n_test": len(t["y_te"]),
                })
                log.info("[seed %d] %-22s acc=%.4f f1=%.4f mcc=%+.4f auc=%.4f "
                         "(majority %.4f, delta %+.4f)", seed, f"{src}->{tgt}",
                         scores["accuracy"], scores["f1"], scores["mcc"],
                         scores["roc_auc"], majority,
                         scores["accuracy"] - majority)

    df = pd.DataFrame(rows)
    write_csv_atomic(df, TRANSFER_DIR / "transfer_per_seed.csv")
    return df


def summarise_transfer(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std across seeds, with a percentile bootstrap CI per metric."""
    from src.consistency import bootstrap_ci

    rows = []
    for direction, g in df.groupby("direction"):
        rec = {"direction": direction, "train": g["train"].iloc[0],
               "test": g["test"].iloc[0], "in_distribution": bool(g["in_distribution"].iloc[0]),
               "n_seeds": len(g), "n_test": int(g["n_test"].iloc[0]),
               "majority_class_accuracy": float(g["majority_class_accuracy"].iloc[0])}
        for m in TRANSFER_METRICS:
            v = g[m].to_numpy(dtype=float)
            rec[f"{m}_mean"] = float(v.mean())
            rec[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            ci = bootstrap_ci(np.mean, v, n=1000, seed=42)
            rec[f"{m}_ci_lower"] = ci["ci_lower"]
            rec[f"{m}_ci_upper"] = ci["ci_upper"]
        rec["accuracy_above_majority"] = rec["accuracy_mean"] - rec["majority_class_accuracy"]
        rows.append(rec)
    order = {"phiusiil->phiusiil": 0, "phiusiil->uci": 1, "uci->uci": 2, "uci->phiusiil": 3}
    out = pd.DataFrame(rows)
    return out.sort_values("direction", key=lambda s: s.map(order)).reset_index(drop=True)


def interpret_transfer(summary: pd.DataFrame) -> list[str]:
    """The B1.2 interpretation block the brief requires."""
    lines = []
    get = lambda d: summary.loc[summary["direction"] == d].iloc[0]  # noqa: E731

    pp, pu = get("phiusiil->phiusiil"), get("phiusiil->uci")
    uu, up = get("uci->uci"), get("uci->phiusiil")

    lines.append(f"PhiUSIIL in-distribution : {pp['accuracy_mean']:.4f} "
                 f"+/- {pp['accuracy_std']:.4f}")
    lines.append(f"PhiUSIIL -> UCI          : {pu['accuracy_mean']:.4f} "
                 f"+/- {pu['accuracy_std']:.4f}  (majority baseline "
                 f"{pu['majority_class_accuracy']:.4f}, delta "
                 f"{pu['accuracy_above_majority']:+.4f}, MCC {pu['mcc_mean']:+.4f})")
    lines.append(f"UCI in-distribution      : {uu['accuracy_mean']:.4f} "
                 f"+/- {uu['accuracy_std']:.4f}")
    lines.append(f"UCI -> PhiUSIIL          : {up['accuracy_mean']:.4f} "
                 f"+/- {up['accuracy_std']:.4f}  (majority baseline "
                 f"{up['majority_class_accuracy']:.4f}, delta "
                 f"{up['accuracy_above_majority']:+.4f}, MCC {up['mcc_mean']:+.4f})")
    lines.append("")

    drop = pp["accuracy_mean"] - pu["accuracy_mean"]
    acc_failed = pu["accuracy_above_majority"] <= 0.02
    auc = pu["roc_auc_mean"]

    # Accuracy and AUC answer different questions, and on this data they disagree,
    # so the verdict must consider both. Accuracy at a fixed 0.5 threshold punishes
    # a miscalibrated operating point; AUC is threshold-free and asks only whether
    # the model RANKS target phishing above target legitimate. A model can fail the
    # first while passing the second, and calling that "no transferable signal"
    # would overstate the leakage claim.
    if acc_failed and auc < 0.65:
        lines.append(f"VERDICT: PhiUSIIL->UCI accuracy ({pu['accuracy_mean']:.4f}) is at or "
                     f"below the majority-class baseline "
                     f"({pu['majority_class_accuracy']:.4f}) AND ROC-AUC is {auc:.4f}, near "
                     f"chance. Against {pp['accuracy_mean']:.4f} in-distribution (a drop of "
                     f"{drop:.4f}), the PhiUSIIL-trained model carries essentially no "
                     f"transferable phishing signal - strong evidence that its performance "
                     f"is collection-specific rather than phishing-specific.")
    elif acc_failed:
        lines.append(f"VERDICT (MIXED - this partially WEAKENS the leakage claim and must "
                     f"be reported as such):")
        lines.append(f"     Accuracy transfers badly: {pu['accuracy_mean']:.4f} against a "
                     f"{pu['majority_class_accuracy']:.4f} majority baseline "
                     f"({pu['accuracy_above_majority']:+.4f}), down from "
                     f"{pp['accuracy_mean']:.4f} in-distribution.")
        lines.append(f"     BUT ROC-AUC is {auc:.4f}, far above the 0.5 chance level, and "
                     f"MCC is {pu['mcc_mean']:+.4f}. The model still RANKS UCI phishing "
                     f"above UCI legitimate well beyond chance; what fails is the decision "
                     f"threshold, not the ordering.")
        lines.append(f"     Reading: the shared-schema signal PhiUSIIL learns is partly "
                     f"genuine and partly collection-specific. The collapse in accuracy is "
                     f"real, but attributing it wholly to leakage would be wrong - a "
                     f"calibration failure under distribution shift produces the same "
                     f"accuracy collapse with the ranking left intact.")
    else:
        lines.append(f"VERDICT: PhiUSIIL->UCI accuracy ({pu['accuracy_mean']:.4f}) sits "
                     f"{pu['accuracy_above_majority']:+.4f} ABOVE the majority-class "
                     f"baseline (ROC-AUC {auc:.4f}). Genuine signal does transfer, which "
                     f"WEAKENS the pure leakage reading and must be reported as such.")

    if pu["mcc_mean"] < -0.02:
        lines.append(f"     MCC is NEGATIVE ({pu['mcc_mean']:+.4f}): the model is not merely "
                     f"uninformed on UCI but anti-correlated - consistent with the "
                     f"inverted-polarity concepts identified in B1.1.")

    lines.append("")
    lines.append(f"IMPORTANT SCOPE NOTE: the in-distribution cell here is "
                 f"{pp['accuracy_mean']:.4f}, NOT the 1.0000 reported in E1. This matrix is "
                 f"computed on the 9-concept SHARED schema, which excludes "
                 f"URLSimilarityIndex and the other leaking features by construction. The "
                 f"transfer contrast is therefore between shared-schema in-distribution and "
                 f"shared-schema cross-distribution; quoting it against the full-feature "
                 f"1.0000 would compare two different feature spaces.")
    return lines


def polarity_agreement() -> pd.DataFrame:
    """Do the two datasets agree on WHICH value of each shared concept means phishing?

    Transfer presupposes that a concept relates to the label in the same direction
    on both sides. Where the direction flips, a source-trained model is not merely
    uninformed about the target - it is actively wrong, and its transfer accuracy
    can fall BELOW chance. Distinguishing those two failure modes matters for the
    interpretation, so the per-concept direction is measured rather than assumed.
    """
    from scipy.stats import spearmanr

    rho = {}
    for ds in ("phiusiil", "uci"):
        X, y = to_shared_schema(ds, dedup=False if ds == "uci" else None)
        rho[ds] = {c: float(spearmanr(X[c], y).statistic) if X[c].nunique() > 1 else 0.0
                   for c in X.columns}

    # A concept whose rho is ~0 on one side has no direction to agree ABOUT, so the
    # no-signal case is tested BEFORE the sign comparison. Testing sign first would
    # score a vacuous match (e.g. +0.003 vs +0.52) as agreement and inflate the
    # headline "concepts that agree" count.
    NO_SIGNAL = 0.02
    rows = []
    for m in FEATURE_MAP:
        rp, ru = rho["phiusiil"][m.concept], rho["uci"][m.concept]
        if abs(ru) < NO_SIGNAL and abs(rp) < NO_SIGNAL:
            status, agree = "NO_SIGNAL_BOTH", False
        elif abs(ru) < NO_SIGNAL:
            status, agree = "NO_SIGNAL_UCI", False
        elif abs(rp) < NO_SIGNAL:
            status, agree = "NO_SIGNAL_PHIUSIIL", False
        elif np.sign(rp) == np.sign(ru):
            status, agree = "agrees", True
        else:
            status, agree = "INVERTED", False
        rows.append({
            "concept": m.concept, "rho_phiusiil": rp, "rho_uci": ru,
            "direction_agrees": bool(agree),
            "abs_rho_phiusiil": abs(rp), "abs_rho_uci": abs(ru),
            "status": status,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task B1: cross-dataset transfer")
    ap.add_argument("--stage", default="mapping",
                    choices=["mapping", "polarity", "matrix"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.stage == "matrix":
        df = run_transfer_matrix(seeds=args.seeds)
        summary = summarise_transfer(df)
        write_csv_atomic(summary, TRANSFER_DIR / "transfer_summary.csv")
        banner(log, "B1.2 TRANSFER SUMMARY (mean +/- std across seeds)", "-")
        cols = ["direction", "n_test", "accuracy_mean", "accuracy_std", "f1_mean",
                "mcc_mean", "roc_auc_mean", "majority_class_accuracy",
                "accuracy_above_majority"]
        for line in summary[cols].round(4).to_string(index=False).splitlines():
            log.info(line)
        banner(log, "B1.2 INTERPRETATION")
        for line in interpret_transfer(summary):
            log.info(line)
        (TRANSFER_DIR / "transfer_notes.json").write_text(json.dumps({
            "summary": summary.to_dict("records"),
            "interpretation": interpret_transfer(summary),
            "b1_4_prior_work": {
                "framework": "CDEPF (ScienceDirect, 2026)",
                "reported": "94.4% on PhiUSIIL->UCI transfer",
                "their_setting": "AFTER domain adaptation",
                "our_setting": "zero-shot: no target labels, no target statistics, "
                               "source scaler applied unchanged to the target",
                "comparability": "NOT directly comparable. Domain adaptation fits to the "
                                 "target distribution, which is precisely the capability "
                                 "this experiment withholds in order to test whether the "
                                 "learned signal is collection-specific. Quoting the two "
                                 "numbers side by side without this distinction would be "
                                 "a serious error.",
            },
        }, indent=2, default=float), encoding="utf-8")
        log.info("-> %s", TRANSFER_DIR / "transfer_summary.csv")
        raise SystemExit(0)

    if args.stage == "polarity":
        banner(log, "B1.1 POLARITY AGREEMENT ACROSS THE SHARED SCHEMA")
        pol = polarity_agreement()
        write_csv_atomic(pol, TRANSFER_DIR / "polarity_agreement.csv")
        for line in pol.round(4).to_string(index=False).splitlines():
            log.info(line)
        bad = pol[pol["status"] == "INVERTED"]
        log.info("")
        log.info("concepts whose direction AGREES : %d / %d",
                 int(pol["direction_agrees"].sum()), len(pol))
        if len(bad):
            log.warning("INVERTED concepts (%d): %s", len(bad), list(bad["concept"]))
            log.warning("A source-trained model using these is not merely uninformed on "
                        "the target - it is actively anti-correlated, so transfer accuracy "
                        "can fall BELOW chance. This must be separated from a plain "
                        "'no signal transfers' reading.")

        dit = directional_inconsistency_table()
        write_csv_atomic(dit, TRANSFER_DIR / "directional_inconsistency.csv")
        head = dit[dit["headline"]]
        if len(head):
            banner(log, "HEADLINE DIRECTIONAL INCONSISTENCY", "-")
            for _, r in head.iterrows():
                log.warning("%s: rho_uci=%+.4f vs rho_phiusiil=%+.4f - the sign is "
                            "INVERTED on the strongest-weighted shared concept "
                            "(|rho|max=%.4f).", r["concept"], r["rho_uci"],
                            r["rho_phiusiil"], r["importance_weight"])
            log.warning("This is not a marginal disagreement: the two datasets encode "
                        "OPPOSITE relationships between this concept and the label, so "
                        "any model transferring between them is actively misled by its "
                        "most informative shared input.")
        log.info("-> %s", TRANSFER_DIR / "polarity_agreement.csv")
        log.info("-> %s", TRANSFER_DIR / "directional_inconsistency.csv")
        raise SystemExit(0)

    banner(log, "B1.1 CONCEPTUAL FEATURE MAPPING")
    tbl = mapping_table()
    write_csv_atomic(tbl, TRANSFER_DIR / "feature_mapping.csv")
    n_map = int((tbl["status"] == "mapped").sum())
    log.info("mapped concepts      : %d", n_map)
    log.info("unmapped UCI features: %d", int((tbl["status"] == "unmapped_uci").sum()))
    log.info("unmapped PhiUSIIL    : %d", int((tbl["status"] == "unmapped_phiusiil").sum()))
    for m in FEATURE_MAP:
        log.info("  %-22s %-20s <- %s%s", m.concept, m.uci_column,
                 " + ".join(m.phiusiil_source), "  [CAVEAT]" if m.caveat else "")
    log.info("-> %s", TRANSFER_DIR / "feature_mapping.csv")

    banner(log, "B1.1 SHARED-SCHEMA ALIGNMENT CHECK", "-")
    desc = describe_shared_schema()
    write_csv_atomic(desc, TRANSFER_DIR / "shared_schema_distributions.csv")
    piv = desc.pivot_table(index=["concept", "value"], columns="dataset",
                           values=["share", "phishing_rate"])
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        for line in piv.round(3).to_string().splitlines():
            log.info(line)
    log.info("-> %s", TRANSFER_DIR / "shared_schema_distributions.csv")
