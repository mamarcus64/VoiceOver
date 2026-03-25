# Pilot Study: Smile Detection from Facial Action Units

## Overview

This pilot study evaluates whether OpenFace Action Unit (AU) features beyond AU12 alone can better predict human-labeled smile quality in a corpus of historical interview footage. The goal is to inform the main study's smile detection pipeline by identifying an optimal feature combination and decision thresholds grounded in human annotations.

---

## Data

**Source.** OpenFace `FeatureExtraction` output at 30 fps per video (`threadward_results/{video_id}/result.csv`). Each row contains 17 continuous AU regression scores (`AU##_r`) and 18 binary AU classifier outputs (`AU##_c`).

**Candidate smile moments.** The upstream pipeline detects candidate segments per video using AU12 as the sole signal:

1. Gaussian-smooth AU12_r (σ = 0.133 s, ≈ 4 frames at 30 fps).
2. Threshold: smoothed AU12_r > 1.0; keep runs ≥ 0.5 s.
3. Compute per-segment summary statistics (mean_r, peak_r, mass_r) on the raw (unsmoothed) AU12 within each run.
4. Post-filter: retain segments with mean_r ≥ 1.5; merge gaps ≤ 1.0 s; drop merged segments < 0.5 s.

All 795 tasks in this pilot satisfy the mean_r ≥ 1.5 criterion. New AU combinations are therefore evaluated on this same pre-filtered pool, making comparisons with the AU12-only baseline directly valid.

**Human annotations.** Six annotators (Leonard: 795 tasks; Christina: 99; Marcus: 88; Julia: 79; Emily: 40; Gabor: 31) labeled each candidate moment as one of: *genuine*, *polite*, *masking*, or *not_a_smile*. For the binary analysis, the first three categories are collapsed to **smile** and the last is **not_a_smile**. Approximately 75 tasks overlap across 3–6 annotators; the remaining tasks were labeled by Leonard alone.

**Consensus label.** For each task, the binary consensus is *smile* if ≥ 50% of annotators said smile, *not_a_smile* otherwise. Label distribution: **500 smile**, **295 not_a_smile** (N = 795).

**Weighting.** Two analysis modes are reported throughout:
- *Unweighted*: each task contributes equally (weight = 1).
- *Weighted*: each task contributes proportionally to its number of annotators (weight = N_annotators), giving higher influence to tasks with stronger consensus evidence.

---

## AU Feature Extraction

For each annotated task (defined by `smile_start`, `smile_end`), frames in the interval `[smile_start, smile_end]` were sliced from the corresponding OpenFace CSV. Per-AU summary statistics were computed on the raw (unsmoothed) signal within that window:

| Statistic | Formula |
|-----------|---------|
| `AU##_r_mean` | mean of AU##_r across frames in window |
| `AU##_r_peak` | max of AU##_r |
| `AU##_r_std` | standard deviation of AU##_r |
| `AU##_r_mass` | Σ(AU##_r) / 30 fps (area under curve, in AU·s) |
| `AU##_c_mean` | fraction of frames where binary classifier fires |

All 17 `AU##_r` and 18 `AU##_c` columns were extracted. The full feature matrix is saved at `analysis/au_features_dataset.csv`.

---

## Results

### Univariate AU discriminability

| AU | Anatomy | AUC (unweighted) | AUC (weighted) |
|----|---------|-----------------|----------------|
| AU12 | Lip Corner Puller | 0.713 | — |
| AU06 | Cheek Raiser | 0.687 | 0.716 |
| AU25 | Lips Part | 0.680 | — |
| AU09 | Nose Wrinkler | 0.666 | 0.726 |
| AU07 | Lid Tightener | 0.585 | — |
| AU14 | Dimpler | 0.583 | — |

AU12 remains the strongest single feature. AU06 (the canonical Duchenne smile marker) and AU25 (lips parting in open smiles) are the next most informative.

### Pairwise combinations

The best pairwise additive scores (AU_i_mean + AU_j_mean) by AUC:

| Combination | AUC (unweighted) | AUC (weighted) |
|-------------|-----------------|----------------|
| AU09 + AU12 | 0.740 | 0.710 |
| AU12 + AU25 | 0.736 | 0.713 |
| AU06 + AU25 | 0.727 | 0.732 |
| AU06 + AU12 (Duchenne) | 0.723 | 0.727 |
| AU06 × AU12 (Duchenne product) | 0.722 | 0.732 |

The classical Duchenne composite (AU06 + AU12) improves over AU12 alone by 0.010 AUC. The multiplicative form (AU06 × AU12) performs similarly and reaches slightly higher F1 at its optimal threshold. Notably, AU09 (Nose Wrinkler) combined with AU12 yields the best pairwise AUC in the unweighted setting, suggesting that nasal activity is an independent predictor of genuine smiles in this corpus.

### Logistic regression over all AUs

A logistic regression trained on all 17 `AU##_r_mean` features (z-scored per feature) with 5-fold cross-validation for L2 regularization achieves:

| | Unweighted | Weighted |
|-|-----------|---------|
| AUC | **0.793** | **0.797** |

This represents a gain of +0.080 AUC over AU12 alone. The top positive predictors (standardized coefficients, unweighted) are:

| Feature | Coef | Interpretation |
|---------|------|----------------|
| AU12 (Lip Corner Puller) | +0.270 | Primary smile signal |
| AU25 (Lips Part) | +0.220 | Open-mouth smiles |
| AU06 (Cheek Raiser) | +0.196 | Duchenne component |
| AU09 (Nose Wrinkler) | +0.140 | Nasal engagement |

Top negative predictors (features associated with *not_a_smile*):

| Feature | Coef | Interpretation |
|---------|------|----------------|
| AU01 (Inner Brow Raiser) | −0.123 | Concern/surprise expression |
| AU17 (Chin Raiser) | −0.120 | Chin tension, non-smile |
| AU45 (Blink) | −0.120 | Occlusion / low confidence frames |
| AU04 (Brow Lowerer) | −0.108 | Frowning |

> **Note on optimism.** AUC values are computed on the training set. The logistic model was not evaluated on a held-out test set in this pilot. Expect 5–10 AUC points of inflation relative to a properly cross-validated estimate.

---

## Suggestions for the Main Study

### 1. Detection pipeline

Retain the existing AU12 pre-filter (mean_r ≥ 1.5) as the first stage to limit the candidate pool. Apply the logistic classifier as a second-stage scorer on the same segments. This preserves scientific comparability with the current system.

### 2. Feature scoring

For each candidate segment, compute `AU##_r_mean` for all 17 AUs from the raw OpenFace CSV. Apply the logistic model:

```
z_i = (AU##_r_mean_i − μ_i) / σ_i          (standardize using pilot scaler)
score = sigmoid(Σ coef_i · z_i + intercept)
```

The pilot scaler and coefficient values are saved at `pilot_analysis/pilot_logistic_model.json`. If the main study corpus has substantially different demographics or recording conditions, re-fit the scaler and re-estimate the coefficients on newly annotated main-study tasks.

### 3. Threshold selection

The following operating points were derived from the pilot (unweighted logistic, in-sample):

| Target recall (P[smile detected | true smile]) | Logistic threshold | Specificity | Precision |
|---|---|---|---|
| 70% sensitivity | 0.636 | 74.9% | 82.8% |
| 80% sensitivity | 0.582 | 58.0% | 76.4% |
| 90% sensitivity | 0.508 | 42.7% | 72.7% |

For comparison, AU12 alone at 90% sensitivity operates at 32.2% specificity and 69.2% precision.

**Recommended default for annotation tasks:** 80% sensitivity (threshold = 0.582). This recovers 80% of true smiles while discarding ~42% of non-smiles that AU12 alone would pass, without being so strict as to systematically miss subtle expressions.

**Recommended for analysis (purity over completeness):** 70% sensitivity (threshold = 0.636). Approximately 83% of detected moments are genuine smiles at this operating point.

### 4. Multi-annotator sampling

Tasks annotated by ≥ 3 raters showed stronger consensus signal. For the main study, prioritize ≥ 2 annotators per task for any subset used to evaluate or re-calibrate the model. The weighted-annotator AUC (0.797) tracks the unweighted result (0.793) closely here, but larger divergences in smaller datasets can indicate label noise.

### 5. Validation

Before deploying the logistic model corpus-wide, annotate a held-out random sample of ~150 main-study tasks (stratified by logistic score quartile) and compute AUC on those held-out labels. This gives an unbiased estimate of out-of-sample performance and will reveal whether the pilot thresholds transfer.

---

## Figures

All figures are in `VoiceOver/pilot_analysis/`. Generated by `pilot_analysis/make_figures.py`.

- **fig1_roc.pdf** — ROC curves: AU12 alone, Duchenne (AU06+AU12), AU09+AU12, Logistic (all AUs); unweighted and weighted panels.
- **fig2_auc_ranking.pdf** — AUC for all 17 univariate AU features plus logistic; sorted bar chart.
- **fig3_logistic_coef.pdf** — Standardized logistic coefficients; unweighted and weighted.

---

## Reproducibility

```bash
# From VoiceOver/
python3 scripts/generate_task_manifest.py          # produces data/smile_task_manifest.json
python3 scripts/build_au_feature_dataset.py        # produces analysis/au_features_dataset.csv
python3 scripts/au_roc_sweep.py                    # produces analysis/au_roc_sweep/
python3 pilot_analysis/make_figures.py             # produces pilot_analysis/fig*.pdf
```

All scripts require `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`.
