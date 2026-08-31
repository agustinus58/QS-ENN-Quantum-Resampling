# Datasets

The datasets used in the QS-ENN experiments are listed below. The links point to the public source or repository from which each dataset can be obtained.

| Dataset | File | Source |
|---|---|---|
| PIMA Indian Diabetes | `pima.csv` | [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Liver Disease (ILPD) | `liver.csv` | [UCI Machine Learning Repository](https://doi.org/10.24432/C5D02C) |
| Breast Cancer Wisconsin (Diagnostic) | `breast_cancer.csv` | [UCI Machine Learning Repository](https://doi.org/10.24432/C5DW2B) |
| Obesity Classification | `obesity.csv` | [Kaggle — Obesity Classification Dataset](https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset) |
| Heart Disease | `heart_disease.csv` | [Kaggle — Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| HCV (Hepatitis C) | `hcv.csv` | [UCI Machine Learning Repository](https://doi.org/10.24432/C5D612) |
| Cervical Cancer (Risk Factors) | `cervical_cancer.csv` | [UCI Machine Learning Repository](https://doi.org/10.24432/C5Z310) |
| Lung Cancer | `lung_cancer.csv` | [Kaggle — Lung Cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) |
| Frankfurt Hospital Diabetes | `frankfurt_diabetes.csv` | [Kaggle — Diabetes Dataset](https://www.kaggle.com/datasets/johndasilva/diabetes) |
| Typhoid | `typhoid.csv` | [Mendeley Data](https://doi.org/10.17632/9hp4tgxz8f.1) |

## Data preprocessing

No observations are removed using IQR-based row deletion. Tukey IQR bounds are used only for exploratory diagnostics where applicable.

Potential target-leakage variables are excluded before model fitting according to the experimental protocol.

All preprocessing operations that learn parameters from the data are fitted exclusively on the training fold and then applied to the corresponding held-out fold. These operations include median imputation, Min–Max scaling, PCA, and the post-PCA rescaling used before quantum encoding.

Up to five principal components are retained. When fewer than five predictors remain, the available components are zero-padded to match the five-qubit input without adding information.

## Dataset-specific leakage controls

- **Cervical Cancer:** `Hinselmann`, `Schiller`, and `Citology` are excluded because they are concurrent screening-test variables closely related to the `Biopsy` target.
- **Typhoid:** `Blood Culture`, `Widal Test`, `ESR`, and `WBC Count` are excluded because they are diagnostic or near-deterministic variables for the target used in this study.
- **Obesity:** `ID` is excluded as an identifier, and `BMI` is excluded because it is definitional for the obesity class label used in the experiment.

