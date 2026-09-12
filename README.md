# Cross-dataset attribution disagreement in phishing URL detection

Code and results supporting the paper *Explanations that do not transfer: cross-dataset
attribution disagreement and benchmark validity in phishing URL detection*.

The study measures whether SHAP attributions computed on one phishing benchmark agree with
those computed on another, interpreted against a **within-dataset reference value**
obtained under identical conditions. It also screens six public phishing benchmarks for
label leakage and identifies a measurement artifact in a commonly used directional
agreement statistic.

Everything is seeded, cached, and logged. There is no unseeded randomness in the codebase.

**Archived release:** https://doi.org/10.5281/zenodo.22727658

---

## Headline results

| Quantity | Value |
|---|---|
| Within-dataset attribution agreement (Kendall's τ) | 0.7683 – 0.9308 |
| Cross-dataset agreement, clean benchmark pairs | 0.1754 – 0.2667 |
| Independent benchmarks carrying a near-oracle feature | 1 of 5 |
| PhiUSIIL sufficiency at k = 3 | 1.0000 (100% of the full 50-feature model) |
| PhiUSIIL comprehensiveness cost of removing the same top 3 | 0.0016 |
| Naive vs corrected directional agreement | 44–51% vs 80–100% |

**Cross-dataset consistency**, against a within-dataset ceiling of 0.8244 (`uci_full`):

| Pairing | Shared concepts | Kendall's τ | Jaccard@5 |
|---|---|---|---|
| UCI vs uci379 | 9 | 0.2667 ± 0.0465 | 0.667 |
| UCI vs mendeley_hannousse | 11 | 0.1754 ± 0.0990 | 0.429 |
| UCI vs phiusiil_leakfree | 7 | −0.4667 ± 0.1086 | 0.429 |

Both quantities are Kendall's τ between two SHAP rankings. They differ only in whether the
rankings originate from two halves of one dataset or from two different datasets, which is
what makes the comparison like-for-like.

---

## Datasets

**No dataset files are included in this repository.** All six benchmark configurations are
downloaded from their original sources by the acquisition scripts. Each dataset remains
under the license set by its publisher.

| Dataset | Source | Identifier |
|---|---|---|
| PhiUSIIL Phishing URL | UCI ML Repository, id 967 | `10.1016/j.cose.2023.103545` |
| Phishing Websites | UCI ML Repository, id 327 | `10.24432/C51W2X` |
| Website Phishing | UCI ML Repository, id 379 | see acquisition report |
| Web page phishing detection | Mendeley Data | `10.17632/c2gw7fy2j4.3` |
| Phishing URL dataset (Tan) | Mendeley Data | see acquisition report |

Verify each identifier against `data/raw/external/acquisition_report.json`, which records
the versions and access dates actually used.

```bash
python src/data_loader.py --force        # PhiUSIIL and UCI, via ucimlrepo
python -m src.external_datasets          # Mendeley sources, with provenance recording
```

Both cache to `data/` (git-ignored) and record source URL, DOI, size, format, and access
date.

---

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # POSIX
```

Verified on **Windows 10, CPU only, Python 3.13.5**. SHAP output varies across library
versions, so these are pinned exactly in `requirements.txt`.

| package | version | | package | version |
|---|---|---|---|---|
| python | 3.13.5 | | shap | 0.52.0 |
| numpy | 2.5.3 | | scipy | 1.18.1 |
| pandas | 3.0.5 | | matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 | | seaborn | 0.13.2 |
| xgboost | 3.4.1 | | pyarrow | 25.0.1 |
| lightgbm | 4.7.0 | | ucimlrepo | 0.0.7 |
| numba | 0.67.0 | | pytest | 9.1.1 |

---

## Run

```bash
python run_phase_a.py                    # performance and within-dataset baselines
python run_addendum.py                   # leak characterization, corrected metrics, sweeps
python -m src.transfer                   # zero-shot cross-dataset transfer
python -m src.benchmark_screen           # screening across all six configurations
python -m src.cross_dataset              # cross-dataset consistency, all pairings
python -m src.faithfulness               # sufficiency, comprehensiveness, injection
python -m src.explanation_cost           # cost profile
python -m src.paper_export               # tables and figures
python -m pytest tests/ -q               # 51 tests
```

Useful flags:

```bash
python run_phase_a.py --skip-tuning        # reuse cached best params
python run_phase_a.py --dataset uci        # one dataset
python run_addendum.py --tasks A2 A3 A6    # a subset
python run_addendum.py --skip-progressive  # skip the expensive greedy removal loop
```

**A1 must run before A4.** The leak-controlled feature set does not exist until progressive
removal has determined it, and `phiusiil_leakfree` is skipped automatically while
`PHIUSIIL_LEAKFREE_FEATURES` is `None`.

Every module is runnable standalone and importable. Stages resume from cached intermediate
results, so an interruption costs at most one seed rather than a restart.

**Runtime.** Approximately 20 hours end to end on an 8-core CPU with no GPU. TreeSHAP on
the leak-controlled PhiUSIIL configuration dominates: explanation cost rises 7.1× as forest
node count rises 21×, while model fitting time stays flat at 20–33 s. A cached replay of
the same work completes in roughly 12 minutes.

---

## Experiment configurations

An experiment *config* is a (dataset, de-duplication policy, feature subset) triple — the
unit E1 and E2 are re-run over, so leak-controlled and full variants sit in one comparable
table.

| config | dataset | rows | features |
|---|---|---|---|
| `uci_full` | UCI | 11,055 (all) | 30 |
| `uci_dedup` | UCI | 5,849 (de-duplicated) | 30 |
| `phiusiil_full` | PhiUSIIL | 234,987 | 50 |
| `phiusiil_leakfree` | PhiUSIIL | 234,987 | 36 |

```python
SEEDS = [42, 43, 44, 45, 46]     TEST_SIZE = 0.15      VAL_SIZE = 0.15
N_BOOTSTRAP = 1000               SHAP_SAMPLE_SIZE = 10000
CV_FOLDS = 5                     N_SEARCH_ITER = 50
```

---

## Layout

```
config.py                    seeds, paths, constants, logging
src/data_loader.py           fetch, cache, diagnostics
src/external_datasets.py     Mendeley acquisition, ARFF support, provenance
src/preprocessing.py         label harmonization, dedup policy, splits, scaling
src/leakage_check.py         single-feature predictive power
src/leak_analysis.py         distributions, thresholds, progressive removal
src/benchmark_screen.py      reusable screening function and verdict logic
src/models.py                RF / XGB / LGBM + soft-voting ensemble
src/tuning.py                RandomizedSearchCV, per configuration
src/evaluate.py              E1 performance
src/experiment_e2.py         E2 within-dataset ceiling
src/shap_utils.py            TreeSHAP, caching, convergence check
src/consistency.py           tau, permutation null, Jaccard, directional, bootstrap
src/transfer.py              zero-shot cross-dataset transfer
src/cross_dataset.py         cross-dataset consistency across pairings
src/faithfulness.py          sufficiency, comprehensiveness, spurious injection
src/explanation_cost.py      cost profile with contention annotation
src/feature_inventory.py     feature inventory reconciliation
src/paper_export.py          publication tables and figures
run_phase_a.py               Phase A orchestration
run_addendum.py              addendum orchestration
tests/test_consistency.py    51 tests on the metric functions

results/paper_tables/        t1-t16, the tables cited in the paper
figures/paper/               f1-f8
```

Caching: `data/raw/*.parquet` (downloads), `data/cache/*_prepared_{dedup,full}.parquet`
(harmonized frames), `data/cache/shap/*.npy` (SHAP arrays), `results/best_params_*.json`
(tuning, per configuration). Delete a cache file to force recomputation.

---

## Reusing the screening procedure

The benchmark screen is the component most likely to be useful independently. It accepts
any labeled tabular dataset and returns a verdict:

```python
from src.benchmark_screen import screen_benchmark

report = screen_benchmark(X, y, dataset_name="my_benchmark", seed=42)

report["n_features_above_95"]   # features that alone exceed 0.95 accuracy
report["full_model_accuracy"]
report["minimal_leaking_set"]
report["verdict"]               # 'clean' | 'repairable' | 'unrepairable'
```

**Read `n_features_above_95` before the verdict.** The verdict depends on an accuracy
threshold, and this study shows it to be fragile in three independent ways: one dataset's
verdict flips between thresholds of 0.97 and 0.99; verdicts shift under hyperparameter
regime; and a de-leaked configuration crosses the verdict line when re-tuned on its own
feature space. The near-oracle count moves under none of these.

Every verdict is exported alongside `verdict_threshold_used`, so a label never appears
without its cutoff.

---

## Methodological notes

These are the non-obvious choices. Each corresponds to a decision defended in the paper.

**Label harmonization is a hard gate.** All datasets are mapped to
`1 = phishing, 0 = legitimate`, and the resulting balance is asserted against published
figures. A silent inversion would flip every SHAP sign while leaving accuracy untouched,
making it undetectable downstream. The assertion is checked against the
**pre-de-duplication** rate, because that is what published figures describe.

**SHAP is computed in probability space.** `model_output='probability'` forces
`feature_perturbation='interventional'`, which requires a background dataset: a 100-row
stratified sample of the *training* split, never of the explained data.
`check_additivity=True` everywhere.

**Ensemble SHAP is the equal-weight mean of the base learners' SHAP values.** This is
exact, not an approximation: soft voting averages base probabilities, Shapley values are
linear in the model output, and all three learners are explained in the same probability
space against the same background. Implemented explicitly in
`shap_utils._ensemble_shap_values` rather than trusting a library shortcut.

**Kendall's tau-b, not tau-a.** SHAP importance vectors contain ties at exactly zero for
features the model never split on; tau-b corrects for them. A permutation null (10,000
shuffles) supplements the analytic p-value, which assumes no ties and an asymptotic
approximation — both shaky for 30-50 features.

**Mean signed SHAP is the wrong direction statistic.** A first implementation averaged
sign agreement over all features and returned approximately 50% on both datasets. Two
diagnoses were tested. The obvious one — that unimportant features contribute noise — is
wrong: restricting the metric to important features *lowers* the figure further, to 20% on
UCI top-10.

The actual cause is cancellation. A feature that pushes toward phishing at some values and
away at others has its positive and negative contributions average out, so its *mean
signed* SHAP sits near zero however important it is. On UCI half A, `url_of_anchor` has
mean |SHAP| = 0.1504 but mean signed SHAP of only +0.0033; the median
`|mean signed| / mean |SHAP|` over the top-10 is **0.042**, versus 0.145 in the tail. The
head is *more* cancelled than the tail, so thresholding concentrates the metric on
precisely the features whose signs are least determinate.

`shap_utils.directional_importance` supplies a statistic that does not cancel: the Spearman
correlation between a feature's value and its SHAP value — "when this feature goes up, does
it push toward phishing?". Measured that way, two UCI halves agree on **100%** of the
top-10 directions, with |rho| >= 0.58 for every top-10 feature in both halves. The sole
top-15 disagreement, `age_of_domain`, has |rho| = 0.28 / 0.11 — genuinely weak, so the
metric flags real uncertainty rather than matching spuriously.

The naive statistic is retained and reported as `sign_agreement_unfiltered`. The gap
between it and the corrected value is itself a finding, and is reported in the paper.

**De-duplication is per-dataset policy, not a global step.**
`config.DEDUPLICATE = {'phiusiil': True, 'uci': False}`. UCI's 30 features are all
ternary-coded `{-1, 0, 1}`, so two websites sharing an identical 30-dimensional vector is
an expected *collision* between distinct sites with the same coarse profile, not a
duplicated record. Removing them discards 47.1% of the data (11,055 to 5,849) and shifts
class balance from the published 44.3% to 51.6%, breaking comparability with prior work.
PhiUSIIL carries continuous features, where an exact tie across all 50 columns is
vanishingly unlikely by chance — its duplicate rate is 0.343% and removal is retained
there.

Duplicate counts are computed and logged for both datasets regardless of whether removal is
applied. `uci_dedup` is retained as a sensitivity variant and reported alongside the
primary `uci_full`.

**Retaining UCI duplicates inflates apparent accuracy, and both things are true at once.**
With duplicates retained, 67.6% of test rows have an exact feature-twin in training and
ensemble accuracy is 0.973; de-duplicated, twins fall to 2.3% and accuracy is 0.953. The
rows are genuine observations, so deleting them is wrong — but a test row whose feature
vector already appears in training is memorizable, so `uci_full` overstates generalization
*to unseen feature profiles*. Both are reported, each labeled with the question it answers.

**Bootstrap resampling preserves pairing.** For paired inputs, both importance vectors are
indexed by a *shared* bootstrap index; resampling independently would destroy exactly the
correspondence being measured.

**E2 halves are independent pipelines.** Each half gets its own `StandardScaler` and its
own ensemble; the halves are disjoint, so the two SHAP profiles come from genuinely
independent data.

**Hyperparameters are tuned per configuration.** Parameters selected on a leaking feature
space are not valid on a de-leaked one. The consequence is quantified in
`t15_hyperparameter_regime_reconciliation.csv`: the same 36-feature configuration scores
0.989191 under inherited parameters and 0.992709 when re-tuned — a difference that crosses
the 0.99 verdict line.

**`enable_categorical=False` is required on XGBoost >= 3.0.** The default flipped to
`True`, and SHAP then refuses interventional TreeSHAP with "Categorical split is not yet
supported" despite every feature being a scaled float. Declaring it false states the truth
about the data; it does not suppress a real error.

**Timings are annotated by concurrent load.** A sampler records competing jobs with
date-aware matching. Only measurements taken without competing CPU load enter the cost
analysis; the eligibility flag is exported alongside every timing.

---

## Selected findings

**`URLSimilarityIndex` is a label artifact, not a strong feature.** The feature is constant
on the legitimate class: all 134,850 legitimate rows take exactly 100.0, while only 0.78%
of phishing rows do. The single rule `URLSimilarityIndex < 100 -> phishing` scores 99.67%
on all 234,987 rows, with precision 1.000 and recall 0.992. Two further PhiUSIIL features
exceed 0.95 single-feature accuracy: `NoOfExternalRef` (0.961) and `LineOfCode` (0.961).
Evidence in `results/urlsimilarity_label_artifact.json` and
`figures/leakage_urlsimilarity.png`.

**The leakage is diffuse, not concentrated.** Greedy removal by attribution rank required
fourteen removals to bring accuracy below 0.99. After the first three, the procedure works
through page-structure counts and then a URL character-ratio feature. The label is
reconstructable from many weakly redundant signals rather than two or three strong ones.

**Sufficiency and comprehensiveness restate this directly.** A model trained on PhiUSIIL's
top three attributed features alone reproduces the full 50-feature model exactly
(accuracy 1.0000). Removing those same three costs 0.0016. Sufficient yet not necessary.

**A planted shortcut cannot outcompete the natural one.** On clean UCI, an injected
50%-strength label proxy reaches attribution rank 3 of 31. On PhiUSIIL, the identical
injection reaches only rank 26 of 51. Genuine-feature rank correlation stays at rho ~ 0.99
throughout — attribution is stable, simply anchored to a leak. Stability and validity are
separable properties.

**The within-dataset ceiling is dataset-dependent.** PhiUSIIL reaches tau = 0.9190 with the
leak present and 0.9308 without it; UCI reaches 0.8244 and 0.7683 de-duplicated. A
cross-dataset tau cannot be judged against a single assumed ceiling.

---

## Outputs

| File | Contents |
|---|---|
| `results/paper_tables/t1`-`t16` | every table cited in the paper |
| `results/paper_tables/t2_benchmark_screening.csv` | population contrast across six benchmarks |
| `results/paper_tables/t8_cross_dataset_consistency.csv` | the headline result |
| `results/paper_tables/t11_feature_mapping.csv` | frozen concept mapping with polarity |
| `results/paper_tables/t15_hyperparameter_regime_reconciliation.csv` | regime-dependent verdicts |
| `results/urlsimilarity_label_artifact.json` | the leakage evidence, quantified |
| `results/dedup_accuracy_diagnostic.json` | duplicate-driven accuracy inflation |
| `results/sign_agreement_diagnostic_v2.json` | the cancellation diagnosis |
| `figures/paper/f6_within_vs_cross.png` | within-dataset ceiling against cross-dataset agreement |
| `figures/paper/f7_group_shap_mass.png` | why the rankings disagree |

---

## Known limitations

- **Attribution transfer under source-trained models is not implemented.** Whether a model
  carries its explanation to a new benchmark or re-derives it from target data is stated as
  future work in the paper.
- **The zero-shot transfer verdict is mixed.** PhiUSIIL to UCI accuracy falls below the
  majority baseline, but ROC-AUC remains 0.8402: ordering survives while calibration fails.
  Consistent with the leakage account, but not establishing it.
- **The explanation-cost diagnostic is validated within a dataset only.** Cross-dataset
  comparison of node count is confounded by dataset size.
- **Tree ensembles only.** TreeSHAP gives exact Shapley values for tree models; conclusions
  may not transfer to deep or transformer-based detectors.

---

## Citation

```bibtex
@article{natarajan2026attribution,
  title   = {Explanations that do not transfer: cross-dataset attribution
             disagreement and benchmark validity in phishing URL detection},
  author  = {Natarajan, Arul Kumar},
  journal = {Journal of Computer Virology and Hacking Techniques},
  year    = {2026},
  note    = {Under review}
}
```

Archived release: https://doi.org/10.5281/zenodo.22727658

---

## License

Code released under the MIT License. Datasets are not redistributed here and remain subject
to the terms set by their respective publishers.

## Contact

Arul Kumar Natarajan — Samarkand International University of Technology, Uzbekistan
