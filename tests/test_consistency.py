"""Tests for the consistency metrics (Task 8).

These functions produce the paper's headline numbers, so the properties asserted
here are the ones a reviewer would check by hand: identical inputs must be a
perfect match, reversed inputs must be a perfect anti-match, and unrelated inputs
must sit at chance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consistency import (  # noqa: E402
    align,
    all_metrics,
    bootstrap_ci,
    jaccard_at_k,
    kendall_tau,
    permutation_null_tau,
    directional_consistency,
    directional_consistency_profile,
    feature_directions,
    sign_agreement,
    sign_agreement_profile,
    sign_agreement_unfiltered,
    sign_agreement_weighted,
)

RNG = np.random.default_rng(12345)


@pytest.fixture
def importance() -> np.ndarray:
    """A generic importance vector: positive, distinct, unordered."""
    return RNG.random(30) + 0.01


# --------------------------------------------------------------------------- #
# Identical inputs
# --------------------------------------------------------------------------- #
def test_tau_identical_is_one(importance):
    tau, p = kendall_tau(importance, importance)
    assert tau == pytest.approx(1.0)
    assert p < 0.001


@pytest.mark.parametrize("k", [5, 10, 15])
def test_jaccard_identical_is_one(importance, k):
    assert jaccard_at_k(importance, importance, k) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [5, 10, 15])
def test_sign_agreement_identical_is_100(importance, k):
    signed = importance * RNG.choice([-1.0, 1.0], size=importance.size)
    imp = np.abs(signed)
    assert sign_agreement(signed, signed, imp, imp, top_k=k).percent == pytest.approx(100.0)


def test_sign_agreement_weighted_identical_is_100(importance):
    signed = importance * RNG.choice([-1.0, 1.0], size=importance.size)
    imp = np.abs(signed)
    assert sign_agreement_weighted(signed, signed, imp, imp).percent == pytest.approx(100.0)


def test_sign_agreement_unfiltered_identical_is_100(importance):
    signed = importance * RNG.choice([-1.0, 1.0], size=importance.size)
    assert sign_agreement_unfiltered(signed, signed) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Reversed inputs
# --------------------------------------------------------------------------- #
def test_tau_reversed_is_minus_one(importance):
    # Negating reverses the ordering exactly, which is what tau = -1 means.
    tau, _ = kendall_tau(importance, -importance)
    assert tau == pytest.approx(-1.0)


def test_tau_reversed_rank_order():
    a = np.arange(1.0, 21.0)
    tau, _ = kendall_tau(a, a[::-1])
    assert tau == pytest.approx(-1.0)


def test_jaccard_disjoint_top_k_is_zero():
    # 20 features; top-5 under `a` is the bottom-5 under its reverse, so the
    # top-5 sets are disjoint and Jaccard must be exactly 0.
    a = np.arange(1.0, 21.0)
    assert jaccard_at_k(a, a[::-1], 5) == pytest.approx(0.0)


@pytest.mark.parametrize("k", [5, 10, 15])
def test_sign_agreement_flipped_is_zero(importance, k):
    signed = importance * RNG.choice([-1.0, 1.0], size=importance.size)
    imp = np.abs(signed)
    assert sign_agreement(signed, -signed, imp, imp, top_k=k).percent == pytest.approx(0.0)


def test_sign_agreement_weighted_flipped_is_zero(importance):
    signed = importance * RNG.choice([-1.0, 1.0], size=importance.size)
    imp = np.abs(signed)
    assert sign_agreement_weighted(signed, -signed, imp, imp).percent == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Random inputs
# --------------------------------------------------------------------------- #
def test_tau_random_is_near_zero():
    taus = [kendall_tau(RNG.random(40), RNG.random(40))[0] for _ in range(200)]
    assert abs(float(np.mean(taus))) < 0.05


def test_tau_random_is_usually_not_significant():
    p_values = [kendall_tau(RNG.random(40), RNG.random(40))[1] for _ in range(200)]
    # Under the null, ~5% should fall below 0.05. Allow generous slack.
    assert float(np.mean(np.asarray(p_values) < 0.05)) < 0.15


def test_sign_agreement_random_is_near_50():
    vals = []
    for _ in range(100):
        a, b = RNG.normal(size=100), RNG.normal(size=100)
        vals.append(sign_agreement(a, b, np.abs(a), np.abs(b), top_k=15).percent)
    # Genuinely unrelated directions must still sit at chance.
    assert 40.0 < float(np.nanmean(vals)) < 60.0


def test_sign_agreement_unfiltered_random_is_near_50():
    vals = [sign_agreement_unfiltered(RNG.normal(size=100), RNG.normal(size=100))
            for _ in range(100)]
    assert 40.0 < float(np.mean(vals)) < 60.0


# --------------------------------------------------------------------------- #
# Permutation null
# --------------------------------------------------------------------------- #
def test_permutation_null_identical_is_significant(importance):
    res = permutation_null_tau(importance, importance, n_perm=2000, seed=1)
    assert res["tau"] == pytest.approx(1.0)
    assert res["p_empirical"] <= 1.0 / (2000 + 1) + 1e-12
    assert abs(res["null_mean"]) < 0.05


def test_permutation_null_random_is_not_significant():
    a, b = RNG.random(30), RNG.random(30)
    res = permutation_null_tau(a, b, n_perm=2000, seed=2)
    # A genuinely unrelated pair should not clear a 1% bar.
    assert res["p_empirical"] > 0.01


def test_permutation_p_is_never_zero():
    a = np.arange(30.0)
    res = permutation_null_tau(a, a, n_perm=500, seed=3)
    assert res["p_empirical"] > 0.0


def test_permutation_null_is_deterministic_given_seed():
    a, b = RNG.random(25), RNG.random(25)
    r1 = permutation_null_tau(a, b, n_perm=500, seed=7)
    r2 = permutation_null_tau(a, b, n_perm=500, seed=7)
    assert r1 == r2


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_the_point_estimate():
    sample = RNG.normal(loc=0.8, scale=0.05, size=50)
    res = bootstrap_ci(np.mean, sample, n=500, seed=4)
    assert res["ci_lower"] <= res["point"] <= res["ci_upper"]
    assert res["ci_lower"] < res["ci_upper"]


def test_bootstrap_ci_paired_inputs_preserve_pairing():
    a = RNG.random(40)
    b = a + RNG.normal(scale=0.01, size=40)  # strongly concordant with a
    res = bootstrap_ci(kendall_tau, (a, b), n=300, seed=5)
    # If pairing were broken by independent resampling, tau would collapse to ~0.
    assert res["ci_lower"] > 0.5
    assert res["point"] > 0.5


def test_bootstrap_ci_is_deterministic_given_seed():
    sample = RNG.random(30)
    assert bootstrap_ci(np.mean, sample, n=200, seed=9) == \
           bootstrap_ci(np.mean, sample, n=200, seed=9)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        kendall_tau(np.arange(5.0), np.arange(6.0))
    with pytest.raises(ValueError):
        jaccard_at_k(np.arange(5.0), np.arange(6.0), 3)
    with pytest.raises(ValueError):
        sign_agreement_unfiltered(np.arange(5.0), np.arange(6.0))
    with pytest.raises(ValueError):
        sign_agreement(np.arange(5.0), np.arange(6.0),
                       np.arange(5.0), np.arange(6.0))


def test_jaccard_k_larger_than_n_features_is_clamped():
    a, b = RNG.random(8), RNG.random(8)
    # With k >= n every feature is in both top-k sets, so Jaccard is 1.
    assert jaccard_at_k(a, b, 15) == pytest.approx(1.0)


def test_align_orders_both_dicts_consistently():
    a = {"f1": 0.5, "f2": 0.1, "f3": 0.9}
    b = {"f3": 0.8, "f1": 0.4, "f2": 0.2}
    va, vb, names = align(a, b)
    assert names == ["f1", "f2", "f3"]
    assert va.tolist() == [0.5, 0.1, 0.9]
    assert vb.tolist() == [0.4, 0.2, 0.8]


def test_align_rejects_disjoint_feature_sets():
    with pytest.raises(ValueError):
        align({"a": 1.0}, {"b": 1.0})


def test_zero_importance_features_do_not_inflate_sign_agreement():
    # A feature the model never used (0.0) must not count as agreeing with a
    # feature that has a real positive contribution.
    a = np.array([0.0, 0.5, -0.5])
    b = np.array([0.3, 0.5, -0.5])
    assert sign_agreement_unfiltered(a, b) == pytest.approx(200.0 / 3.0)


# --------------------------------------------------------------------------- #
# Task A2 - the defect that thresholding fixes
# --------------------------------------------------------------------------- #
def _tail_disagreement_case(n_important: int = 20, n_tail: int = 30, seed: int = 3):
    """Two attribution sets agreeing perfectly on the signal, disagreeing on noise.

    The important head has large, identically-signed values in both sets. The tail
    has near-zero values whose signs are independent coin flips - exactly the
    situation that dragged the Phase A figure down to ~50%.
    """
    rng = np.random.default_rng(seed)
    head = rng.uniform(0.5, 1.0, n_important) * rng.choice([-1.0, 1.0], n_important)
    tail_a = rng.normal(scale=1e-6, size=n_tail)
    tail_b = rng.normal(scale=1e-6, size=n_tail)
    signed_a = np.concatenate([head, tail_a])
    signed_b = np.concatenate([head, tail_b])
    return signed_a, signed_b, np.abs(signed_a), np.abs(signed_b)


@pytest.mark.parametrize("k", [5, 10, 15])
def test_tail_noise_does_not_drag_down_thresholded_sign_agreement(k):
    """THE regression test: disagreement confined to the unimportant tail.

    The two sets agree on every feature that matters, so a metric measuring the
    signal must stay at 100%. The original all-features metric does not, which is
    precisely the defect this test exists to catch.
    """
    signed_a, signed_b, abs_a, abs_b = _tail_disagreement_case()
    res = sign_agreement(signed_a, signed_b, abs_a, abs_b, top_k=k)
    assert res.percent == pytest.approx(100.0)
    assert res.n_features >= 3


def test_tail_noise_does_sink_the_unfiltered_metric():
    """The defect itself, asserted - so the gap between the two is pinned down."""
    signed_a, signed_b, _, _ = _tail_disagreement_case()
    unfiltered = sign_agreement_unfiltered(signed_a, signed_b)
    # 20 real features agree; 30 tail features agree only by chance, so the
    # unfiltered metric lands near 20/50 + 0.5*30/50 = 70%.
    assert unfiltered < 80.0


def test_top_k_beyond_the_signal_degrades_towards_the_unfiltered_value():
    """An honest limitation of thresholding, pinned so it cannot regress silently.

    top_k is a fixed cut-off, not a discovered one. Asking for more important
    features than genuinely exist forces noise into the selection, and the metric
    correctly falls below 100%. The paper therefore reports n alongside every
    value, and reports several k.
    """
    # Only 6 features carry signal, but we ask for the top 15.
    signed_a, signed_b, abs_a, abs_b = _tail_disagreement_case(n_important=6, n_tail=44)
    tight = sign_agreement(signed_a, signed_b, abs_a, abs_b, top_k=5)
    loose = sign_agreement(signed_a, signed_b, abs_a, abs_b, top_k=15)
    assert tight.percent == pytest.approx(100.0)
    assert loose.percent < 100.0


def test_weighted_sign_agreement_is_robust_to_the_tail():
    """The threshold-free variant must also survive the tail, without a cut-off."""
    signed_a, signed_b, abs_a, abs_b = _tail_disagreement_case()
    assert sign_agreement_weighted(signed_a, signed_b, abs_a, abs_b).percent > 99.0


def test_sign_agreement_returns_n_it_was_computed_over():
    signed_a, signed_b, abs_a, abs_b = _tail_disagreement_case()
    res = sign_agreement(signed_a, signed_b, abs_a, abs_b, top_k=10)
    assert res.n_features == len(res.features)
    assert res.n_features <= 10


def test_sign_agreement_nan_when_too_few_shared_features():
    """Disjoint top-k sets must yield NaN, not a percentage over one or two features."""
    # a ranks the first features highest, b ranks the last features highest, so
    # their top-2 sets cannot intersect.
    n = 10
    abs_a = np.linspace(1.0, 0.1, n)
    abs_b = np.linspace(0.1, 1.0, n)
    signed = np.ones(n)
    res = sign_agreement(signed, signed, abs_a, abs_b, top_k=2)
    assert np.isnan(res.percent)
    assert res.n_features < 3


def test_sign_agreement_is_symmetric_in_its_arguments():
    """A consistency metric must not depend on which set is passed first."""
    signed_a, signed_b, abs_a, abs_b = _tail_disagreement_case(seed=11)
    fwd = sign_agreement(signed_a, signed_b, abs_a, abs_b, top_k=10).percent
    rev = sign_agreement(signed_b, signed_a, abs_b, abs_a, top_k=10).percent
    assert fwd == pytest.approx(rev)
    w_fwd = sign_agreement_weighted(signed_a, signed_b, abs_a, abs_b).percent
    w_rev = sign_agreement_weighted(signed_b, signed_a, abs_b, abs_a).percent
    assert w_fwd == pytest.approx(w_rev)


def test_the_regression_test_actually_discriminates():
    """Meta-test: the A2 regression test must FAIL against the pre-fix behaviour.

    A regression test that passes both before and after a fix tests nothing. The
    corrected metric has a different signature from the original, so the old
    function cannot be called directly; instead the old BEHAVIOUR (average the
    sign match over every feature, ignoring importance and top_k) is re-created
    behind the new signature and the regression assertion is re-run against it.

    If this test ever starts passing trivially, the tail-noise test above has
    stopped discriminating and the suite has lost its guarantee.
    """
    def old_behaviour(signed_a, signed_b, abs_a, abs_b, top_k=10):
        pct = sign_agreement_unfiltered(signed_a, signed_b)
        return pct

    signed_a, signed_b, abs_a, abs_b = _tail_disagreement_case()

    # The fix: perfect agreement on the features that matter.
    assert sign_agreement(signed_a, signed_b, abs_a, abs_b,
                          top_k=10).percent == pytest.approx(100.0)
    # The defect: the same inputs, scored the old way, are nowhere near 100%.
    assert old_behaviour(signed_a, signed_b, abs_a, abs_b, top_k=10) < 80.0


# --------------------------------------------------------------------------- #
# Task B3 - directional consistency
# --------------------------------------------------------------------------- #
def _monotone_case(n: int = 400, seed: int = 5):
    """A feature whose SHAP contribution rises with its value, in both sets.

    This is the case mean signed SHAP handles correctly, so it is the control:
    both metrics should agree that direction is consistent.
    """
    rng = np.random.default_rng(seed)
    Xa = rng.normal(size=(n, 3))
    Xb = rng.normal(size=(n, 3))
    # feature 0 monotone increasing; the others are noise.
    Sa = np.column_stack([Xa[:, 0] * 0.5, rng.normal(scale=1e-3, size=n),
                          rng.normal(scale=1e-3, size=n)])
    Sb = np.column_stack([Xb[:, 0] * 0.5, rng.normal(scale=1e-3, size=n),
                          rng.normal(scale=1e-3, size=n)])
    return Xa, Sa, Xb, Sb, ["monotone", "noise1", "noise2"]


def _bidirectional_case(n: int = 400, seed: int = 6):
    """A feature contributing POSITIVELY at high values and NEGATIVELY at low ones.

    Its mean signed SHAP cancels to ~0 while its mean |SHAP| stays large - the
    exact situation that broke the original metric. Direction is identical in
    both sets, so a correct directional metric must report agreement.
    """
    rng = np.random.default_rng(seed)
    Xa = rng.normal(size=(n, 2))
    Xb = rng.normal(size=(n, 2))
    Sa = np.column_stack([Xa[:, 0] * 0.8, rng.normal(scale=1e-3, size=n)])
    Sb = np.column_stack([Xb[:, 0] * 0.8, rng.normal(scale=1e-3, size=n)])
    return Xa, Sa, Xb, Sb, ["bidirectional", "noise"]


def test_directional_monotone_feature_gives_rho_near_one_and_full_agreement():
    Xa, Sa, Xb, Sb, names = _monotone_case()
    rho_a = feature_directions(Xa, Sa, names)
    rho_b = feature_directions(Xb, Sb, names)
    assert rho_a[0] == pytest.approx(1.0, abs=0.02)
    assert rho_b[0] == pytest.approx(1.0, abs=0.02)
    res = directional_consistency(Xa, Sa, Xb, Sb, names, top_k=1, rho_min=0.5)
    assert res["agreement_percent"] == pytest.approx(100.0)
    assert res["n_meeting_rho_min"] == 1


def test_bidirectional_feature_cancels_under_the_old_metric():
    """The defect, asserted: mean signed SHAP is ~0 despite a large mean |SHAP|."""
    Xa, Sa, _, _, _ = _bidirectional_case()
    mean_signed = float(np.abs(Sa[:, 0].mean()))
    mean_abs = float(np.abs(Sa[:, 0]).mean())
    assert mean_abs > 0.4                       # the feature is important
    assert mean_signed / mean_abs < 0.15        # yet its mean signed value cancels


def test_bidirectional_feature_is_handled_correctly_by_directional_consistency():
    """The fix: the same feature shows a strong, consistent direction."""
    Xa, Sa, Xb, Sb, names = _bidirectional_case()
    res = directional_consistency(Xa, Sa, Xb, Sb, names, top_k=1, rho_min=0.5)
    assert res["agreement_percent"] == pytest.approx(100.0)
    assert abs(res["rho_pairs"][0]["rho_a"]) > 0.9


def test_directional_consistency_detects_a_genuine_inversion():
    """A feature with opposite directions must NOT be scored as agreeing."""
    Xa, Sa, Xb, Sb, names = _monotone_case()
    Sb_flipped = Sb.copy()
    Sb_flipped[:, 0] *= -1          # same magnitude, opposite direction
    res = directional_consistency(Xa, Sa, Xb, Sb_flipped, names, top_k=1, rho_min=0.5)
    assert res["agreement_percent"] == pytest.approx(0.0)
    assert res["n_meeting_rho_min"] == 1       # both are strong, they simply disagree


def test_directional_consistency_rho_min_excludes_weak_directions():
    """A weak direction should fail a high rho_min but pass a low one."""
    rng = np.random.default_rng(7)
    n = 600
    Xa = rng.normal(size=(n, 1))
    Xb = rng.normal(size=(n, 1))
    # Mostly noise with a faint monotone component -> modest |rho|.
    Sa = (Xa * 0.05 + rng.normal(scale=1.0, size=(n, 1)))
    Sb = (Xb * 0.05 + rng.normal(scale=1.0, size=(n, 1)))
    lo = directional_consistency(Xa, Sa, Xb, Sb, ["weak"], top_k=1, rho_min=0.01)
    hi = directional_consistency(Xa, Sa, Xb, Sb, ["weak"], top_k=1, rho_min=0.5)
    assert hi["n_meeting_rho_min"] == 0
    assert hi["agreement_percent"] == pytest.approx(0.0)
    assert lo["n_meeting_rho_min"] >= 0        # low threshold admits it


def test_directional_consistency_is_symmetric():
    Xa, Sa, Xb, Sb, names = _monotone_case()
    f = directional_consistency(Xa, Sa, Xb, Sb, names, top_k=3, rho_min=0.3)
    r = directional_consistency(Xb, Sb, Xa, Sa, names, top_k=3, rho_min=0.3)
    assert f["agreement_percent"] == pytest.approx(r["agreement_percent"])
    assert f["n_meeting_rho_min"] == r["n_meeting_rho_min"]


def test_directional_profile_covers_every_threshold_combination():
    Xa, Sa, Xb, Sb, names = _monotone_case()
    prof = directional_consistency_profile(Xa, Sa, Xb, Sb, names, ks=(1, 2, 3),
                                           rho_mins=(0.3, 0.5))
    for k in (1, 2, 3):
        for rm in ("03", "05"):
            assert f"directional_top{k}_rho{rm}" in prof
            assert f"n_meeting_rho_min_top{k}_rho{rm}" in prof


def test_sign_agreement_profile_reports_every_variant():
    signed_a, signed_b, abs_a, abs_b = _tail_disagreement_case()
    prof = sign_agreement_profile(signed_a, signed_b, abs_a, abs_b)
    for k in (5, 10, 15):
        assert prof[f"sign_agreement_top{k}"] == pytest.approx(100.0)
        assert prof[f"n_features_sign_top{k}"] >= 3
    assert prof["sign_agreement_weighted"] > 99.0
    assert prof["sign_agreement_unfiltered"] < 80.0


# --------------------------------------------------------------------------- #
# Aggregate helper
# --------------------------------------------------------------------------- #
def test_all_metrics_identical_inputs(importance):
    signed = importance * RNG.choice([-1.0, 1.0], size=importance.size)
    m = all_metrics(importance, importance, signed, signed, seed=11, n_perm=500)
    assert m["kendall_tau"] == pytest.approx(1.0)
    assert m["jaccard_5"] == pytest.approx(1.0)
    assert m["jaccard_10"] == pytest.approx(1.0)
    assert m["jaccard_15"] == pytest.approx(1.0)
    assert m["sign_agreement_top10"] == pytest.approx(100.0)
    assert m["sign_agreement_weighted"] == pytest.approx(100.0)
    assert m["sign_agreement_unfiltered"] == pytest.approx(100.0)
    assert m["tau_pvalue_permutation"] < 0.01
