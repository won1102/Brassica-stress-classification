# Transcriptome-Based Stress Classification Pipeline

This repository contains the Python code used for XGBoost-SHAP feature selection and Top-N logistic regression evaluation in *Brassica rapa* abiotic-stress classification.

The workflow corresponds to the feature-selection and minimal-marker modeling steps described in Sections 2.3–2.4 of the associated manuscript. Gene-level redundancy removal was performed as an annotation-based curation step between feature selection and Top-N evaluation.

## Pipeline overview

```text
Train-only TPM feature matrix
        |
        v
1. feature_selection.py
   - log2(TPM + 1)
   - repeated stratified 5-fold CV x 5 repeats
   - undersampling within each training fold
   - MinMax scaling fitted within each training fold
   - XGBoost training
   - SHAP values calculated on validation folds
   - mean absolute SHAP ranking
   - cumulative SHAP cutoff at 90%
   - Top-10% stability frequency
   - stability frequency >= 0.40
        |
        v
Stable candidate ranking
        |
        v
Gene-level redundancy removal
   - map stable candidate features to NCBI gene IDs using the reference GFF annotation
   - treat features mapped to the same gene ID as redundant
   - retain the highest SHAP-ranked feature for each gene
        |
        v
Curated final ranking
        |
        v
2. topn_evaluation.py
   - evaluate ranked Top-N feature sets
   - stratified 5-fold logistic regression CV
   - MinMax scaling fitted within each CV training fold
   - class_weight="balanced"
   - fit each evaluated Top-N model on the full training set
   - evaluate the fixed held-out test set
```

## Repository structure

```text
.
├── feature_selection.py
├── topn_evaluation.py
├── requirements.txt
└── README.md
```

## Input data

### 1. Expression feature matrix

`feature_selection.py` and `topn_evaluation.py` use TPM expression matrices with the following structure:

```text
sample | treatment | classification | feature_1 | feature_2 | ...
```

Required metadata columns:

- `sample`: unique sample identifier
- `treatment`: treatment label retained as metadata
- `classification`: binary class label
  - `1` = target stress
  - `0` = non-target class

All remaining columns are treated as expression features.

Expression values are expected to be TPM values. The scripts apply:

```text
log2(TPM + 1)
```

before model fitting.

For `feature_selection.py`, the input matrix must contain training samples only. Held-out test samples must not be included in feature selection.

### 2. Curated final feature ranking

`topn_evaluation.py` requires a ranked feature table prepared after stability filtering and gene-level redundancy removal.

The table must contain:

```text
Feature
```

and a supported ranking column such as `Rank` or `Final_rank`.

`Feature` must retain the original expression-matrix feature ID so that selected features can be matched directly to the TPM matrix.

### 3. Fixed test sample file

`topn_evaluation.py` requires an Excel file containing at least:

```text
sample
```

Samples listed in this file are separated from the training data before cross-validation and used as the fixed held-out test set.

## Installation

Tested with Python 3.13.0.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1. XGBoost-SHAP feature selection

Run separately for each target-stress classifier using a train-only feature matrix.

```bash
python feature_selection.py \
    --data-path path/to/train_only_feature_matrix.xlsx \
    --out-dir path/to/feature_selection_results
```

Default settings:

```text
Seed                       = 42
Cross-validation           = 5 folds x 5 repeats
Cumulative SHAP cutoff     = 0.90
Stability definition       = Top 10% of features in each validation fold
Stability-frequency cutoff = 0.40

XGBoost:
max_depth                  = 3
n_estimators               = 200
learning_rate              = 0.05
subsample                  = 0.8
colsample_bytree           = 0.8
min_child_weight           = 2
reg_lambda                 = 3.0
reg_alpha                  = 0.1
```

The stability threshold can also be supplied explicitly:

```bash
--stability-threshold 0.40
```

Main outputs:

```text
Repeated5fold_SHAP_importance.xlsx
Repeated5fold_performance.xlsx
SHAP_90pct_candidate_genes.xlsx
SHAP_stable_candidate_genes.xlsx
SHAP_feature_selection_parameters.json
Cumulative_SHAP_curve.png
```

`Repeated5fold_SHAP_importance.xlsx` contains the mean absolute SHAP importance, SHAP rank, cumulative SHAP contribution, and Top-10% stability frequency for each input feature.

`SHAP_90pct_candidate_genes.xlsx` contains the ranked features up to the first feature at which cumulative SHAP contribution reaches 0.90.

`SHAP_stable_candidate_genes.xlsx` contains candidates satisfying `StabilityFreq_Top10pct >= 0.40` within the 90% cumulative-SHAP candidate set.

## Gene-level redundancy removal

Before Top-N evaluation, stable candidate features were mapped to NCBI gene IDs using the reference GFF annotation.

Features mapped to the same gene ID were treated as redundant. For each duplicated gene, the feature with the highest final mean absolute SHAP rank was retained.

This curation step is documented here and is not implemented as a separate script in this repository. The resulting curated ranking was used as input for Top-N evaluation.

The original expression-matrix feature ID should be retained in the `Feature` column so that `topn_evaluation.py` can match selected features directly to the TPM matrix.

## Step 2. Top-N logistic regression evaluation

```bash
python topn_evaluation.py \
    --data-path path/to/train_test_feature_matrix.xlsx \
    --ranking-path path/to/curated_final_ranking.xlsx \
    --fixed-test-path path/to/fixed_test_samples.xlsx \
    --out-dir path/to/topn_results
```

By default, the script evaluates the following Top-N checkpoints where available:

```text
3, 5, 7, 9, 10, 15, 20
```

If more than 20 candidate features are available, additional checkpoints are added at 10-feature intervals:

```text
30, 40, 50, ...
```

The complete candidate-feature set is always included automatically.

For example:

```text
15 candidates -> 3, 5, 7, 9, 10, 15
24 candidates -> 3, 5, 7, 9, 10, 15, 20, 24
35 candidates -> 3, 5, 7, 9, 10, 15, 20, 30, 35
```

Custom Top-N values can also be supplied explicitly:

```bash
--feature-counts 3 5 7 9 10 15 20
```

Requested values larger than the number of ranked features are skipped automatically.

Top-N refers to the first N features in the curated final ranking. No SHAP cutoff, stability filtering, or gene-level redundancy removal is repeated in this step.

Logistic regression settings:

```text
Input transformation = log2(TPM + 1)
Cross-validation     = stratified 5-fold
solver               = liblinear
max_iter             = 500
class_weight         = balanced
seed                 = 42
```

For each evaluated Top-N, the script saves:

```text
Top_N/
├── selected_genes.csv
├── CV_fold_results.csv
├── CV_oof_predictions.csv
├── CV_summary.csv
├── Test_summary.csv
├── Test_predictions.csv
├── Test_confusion_matrix.csv
├── logistic_model.pkl
├── scaler.pkl
└── coefficients.csv
```

An overall comparison is saved as:

```text
Overall_Summary.csv
```

## Marker-panel selection and test-set usage

The fixed test samples are excluded from feature selection and cross-validation.

Marker-panel size is determined from the cross-validation results. In the associated analysis, the final panel was chosen as the smallest evaluated Top-N set achieving a mean cross-validation F1 score >= 0.90.

Test metrics are reported for the evaluated Top-N models but are not used to select or optimize the marker-panel size.

## Reproducibility

The main random seed is:

```text
42
```

Scaling is fitted using training data only:

- XGBoost-SHAP feature selection: `MinMaxScaler` is fitted on the undersampled training partition of each CV fold and applied to its validation partition.
- Top-N logistic regression CV: `MinMaxScaler` is fitted on each CV training fold and applied to the corresponding validation fold.
- Final Top-N model: `MinMaxScaler` is fitted on the full training set and applied to the fixed test set.

SHAP values used for feature importance are calculated on validation samples rather than on the samples used to fit each XGBoost model.

## Notes

- `treatment` is retained as sample metadata and is not used as a model feature.
- Feature IDs used for model fitting are preserved throughout the pipeline.
- Stability filtering is performed after the 90% cumulative SHAP candidate set is defined.
- Gene-level redundancy is defined using gene IDs from the reference GFF annotation.
- Cold and Heat analyses are run separately because their candidate features and final rankings differ.

## Citation

Citation information for the associated article will be added after publication.
