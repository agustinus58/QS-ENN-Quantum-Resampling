# Fold-Level Results

This directory contains the unrounded fold-level outputs used in the QS-ENN experiments and statistical comparisons.

## Evaluation protocol

Each dataset file contains results from:

- 3 independent validation seeds: `11`, `23`, and `37`
- 10 stratified folds per seed
- 30 paired outer-fold evaluations per experimental arm
- 12 experimental arms
- 360 rows per dataset file

The validation seeds are independent of the design-selection seeds used during method development.

## Files

| File | Dataset |
|---|---|
| `folds_breast.csv` | Breast Cancer Wisconsin |
| `folds_cervical.csv` | Cervical Cancer |
| `folds_frankfurt.csv` | Frankfurt Hospital Diabetes |
| `folds_hcv.csv` | HCV (Hepatitis C) |
| `folds_heart.csv` | Heart Disease |
| `folds_liver.csv` | Liver Disease |
| `folds_lung.csv` | Lung Cancer |
| `folds_obesity.csv` | Obesity |
| `folds_pima.csv` | PIMA Indian Diabetes |
| `folds_typhoid.csv` | Typhoid |

## Experimental arms

The fold-level files contain the following experimental arms:

- `KNN + SMOTE-ENN`
- `QKNN + No Resampling`
- `QKNN + SMOTE`
- `QKNN + Borderline-SMOTE`
- `QKNN + SMOTE-ENN`
- `QKNN + SMOTE-Tomek`
- `QKNN + Quantum-SMOTE`
- `QKNN + Quantum-SMOTEV2`
- `QKNN + QS-ENN`
- `Ablation: Without Borderline Allocation`
- `Ablation: Pillar 2 = Neighbor`
- `Ablation: Without Q-ENN`

In the original raw output files, a few arm labels may appear in Indonesian:

- `QKNN + Tanpa Resampling` = `QKNN + No Resampling`
- `ablasi: tanpa borderline` = `Ablation: Without Borderline Allocation`
- `ablasi: P2 = tetangga` = `Ablation: Pillar 2 = Neighbor`
- `ablasi: tanpa Q-ENN` = `Ablation: Without Q-ENN`

These label differences do not affect the numerical results.

## Columns

| Column | Description |
|---|---|
| `dataset` | Dataset name |
| `key` | Internal dataset key |
| `rep` | Validation repetition index |
| `fold` | Fold index |
| `seed` | Validation random seed |
| `arm` | Experimental method or ablation variant |
| `multiclass` | Whether the dataset is multiclass |
| `Acc` | Accuracy |
| `Recall_1` | Minority-class recall; macro recall for multiclass data |
| `Prec_1` | Minority-class precision; macro precision for multiclass data |
| `F1_1` | Minority-class F1-score; macro F1 for multiclass data |
| `Gmean` | Geometric mean |
| `AUC` | Area under the ROC curve |
| `BalAcc` | Balanced accuracy |
| `n_synth` | Number of synthetic samples generated |
| `n_removed` | Total number of samples removed during editing |
| `n_removed_maj_orig` | Number of original majority samples removed |
| `n_removed_min_orig` | Number of original minority samples removed |
| `n_removed_synth` | Number of synthetic samples removed |
| `n_min_before` | Minority count before resampling/editing |
| `n_maj_before` | Majority count before resampling/editing |
| `n_min_after` | Minority count after resampling/editing |
| `n_maj_after` | Majority count after resampling/editing |
| `ir_before` | Imbalance ratio before resampling/editing |
| `ir_after` | Imbalance ratio after resampling/editing |
| `cum_var` | Cumulative PCA variance retained in the corresponding fold |

## Statistical traceability

The statistical comparisons reported in the study are calculated from these unrounded paired fold-level values. Reported table values are rounded only for presentation.

Paired comparisons use the same repetition and fold indices across methods. This preserves fold-wise pairing for Wilcoxon signed-rank testing and Holm-adjusted multiple-comparison analysis.

## Multiclass note

`folds_obesity.csv` corresponds to the multiclass Obesity dataset. Its Recall, Precision, F1-score, and related summary metrics use macro averaging where applicable.

## Typhoid note

Typhoid fold-level results are retained for transparency and reproducibility. Because the post-cleaning discrimination was approximately chance level (`AUC ≈ 0.495`), Typhoid is excluded from comparative and ablation claims in the accompanying study.

## Reproducibility

These files should be kept unchanged as the numerical record underlying the reported aggregate tables and statistical tests. Re-running the experiment using `qsenn.py` and the corresponding dataset notebooks should regenerate the same fold-level structure under the stated software environment and random seeds.
