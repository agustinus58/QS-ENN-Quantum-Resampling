# QS-ENN-Quantum-Resampling
# QS-ENN — Quantum SMOTE with Geodesic Synthesis and Protective Editing

Companion code for the QS-ENN study. A single shared module (`qsenn.py`) contains the core implementation and is used by ten lightweight dataset-specific notebooks.

## Repository Structure

```text
qsenn.py                     shared implementation
notebooks/
  00_aggregation.ipynb       Tables 1/2/3/5 + applicability analysis
  01_cervical.ipynb          one notebook per dataset
  02_pima.ipynb              ...
  10_obesity.ipynb
```

Each dataset notebook calls `run_all([KEY])` and displays the corresponding outputs. Core logic is centralized in `qsenn.py` to avoid duplicated implementations across datasets and to keep all experiments consistent.

## Usage

Place `qsenn.py` in the project directory or Google Drive folder, then run it from each notebook:

```python
%run /content/drive/MyDrive/HASIL_QSENN/qsenn.py

init(
    data_dir="/content/drive/MyDrive/DRIVES3/DATA/",
    out_dir="/content/drive/MyDrive/HASIL_QSENN/",
    profil="sedang",
)

run_all(["pima"])
```

Outside Google Colab, `qsenn.py` can also be imported as a standard Python module.

## Recommended Workflow

1. **Run preflight checks.** Execute the first cell of each notebook or call `preflight()` for the full dataset registry. This checks file names, target columns, and features that may indicate target leakage.
2. **Run dataset experiments.** The ten dataset notebooks can be executed independently or in parallel sessions. Results are written to `OUT_DIR`, and completed datasets can be skipped automatically.
3. **Aggregate the results.** Run `00_aggregation.ipynb` to generate Tables 1, 2, 3, and 5 and the leave-one-dataset-out applicability analysis.

## Main Functions

| Function | Purpose |
|---|---|
| `init(data_dir, out_dir, profil)` | Configure data/output paths and computational profile |
| `preflight(keys)` | Validate dataset registry and screen for potential target leakage |
| `load_dataset(key)` | Load and preprocess one dataset |
| `run_dataset(key)` | Run the complete pipeline for one dataset |
| `run_all(keys, force)` | Batch execution with resume support |
| `make_tables(df)` | Generate Tables 2/3/5 and traceability checks |
| `applicability_lodo()` | Run leave-one-dataset-out applicability analysis |
| `run_nested_cv(key)` | Run equal-budget nested cross-validation |
| `run_representation(key)` | Run diagnostic sensitivity analysis across K and feature-map settings |
| `run_finite_shot(key)` | Run finite-shot SWAP-test fidelity diagnostics |

## Methodological Notes

### Fold-isolated preprocessing

Median imputation, Min–Max scaling, and PCA are fitted exclusively on the training fold and then applied to the held-out fold using the same fitted objects. Retained variance is therefore reported as mean ± standard deviation across folds rather than as a single value.

### IQR handling

No dataset rows are removed using IQR-based filtering in the evaluation pipeline. For Quantum-SMOTEV2, IQR-based angular-outlier detection is used only as part of its sample-generation procedure. Dataset size (`n`) and imbalance ratio (`IR`) therefore correspond to the data after leakage-variable exclusions and without IQR-based row deletion.

### Target-leakage control

For the Cervical Cancer dataset, `Hinselmann`, `Schiller`, and `Citology` are excluded because they are concurrent screening variables closely related to the `Biopsy` target. The `Dx*` variables are retained as prior-diagnosis risk factors.

### Delta conventions

Two sign conventions are used intentionally:

- **Baseline comparison:** Δ = QS-ENN − baseline. Positive values favor QS-ENN.
- **Ablation analysis:** Δ = ablated variant − complete QS-ENN. Positive values indicate that removing or replacing the component improves the metric.

### Quantum encoder

`fast_statevectors` computes the feature map using vectorized NumPy operations. Its output was verified against the PennyLane qnode implementation to approximately `1e-16` across 12 combinations of circuit depth and entanglement pattern. The vectorized implementation is approximately 60× faster for the tested configuration.

### Multiclass handling

The Obesity dataset uses `qs_enn_mc`. Safe-level fidelity is defined as the fraction of same-class neighbors, synthetic states are directed toward the centroid of their own class, and protective editing preserves original samples from all non-majority classes. For binary datasets, this formulation reduces to the standard binary QS-ENN definition.

Obesity is evaluated using macro-averaged metrics. Quantum-SMOTE and Quantum-SMOTEV2 are originally defined for binary settings; for Obesity, this codebase uses the corresponding multiclass generalization implemented without QS-ENN Pillars 1 and 3.

## Random Seeds

- **Design selection:** 42, 202, 777
- **Primary validation:** 11, 23, 37
- **Diagnostics:** 11

These seed groups are kept separate throughout the codebase so that the runs used for design selection are not reused as the reported primary validation results.
