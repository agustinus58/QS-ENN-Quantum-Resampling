"""qsenn — Quantum SMOTE with Geodesic Synthesis and Protective Editing.

Shared implementation for the QS-ENN experiments. Dataset-specific notebooks
configure paths and call the public functions in this module.

The module provides fold-isolated preprocessing, quantum feature encoding,
QS-ENN and comparison resamplers, repeated cross-validation, ablation studies,
equal-budget nested validation, representation diagnostics, finite-shot
simulation, statistical testing, and table-generation utilities.

Typical usage
-------------
    %run /content/drive/MyDrive/HASIL_QSENN/qsenn.py
    init(data_dir="/content/drive/MyDrive/DRIVES3/DATA/")
    df = run_dataset("pima")
"""


from __future__ import annotations

import os, glob, time, contextlib
import numpy as np
import pandas as pd

try:
    import pennylane as qml
except Exception:
    qml = None

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, balanced_accuracy_score)
from scipy.stats import wilcoxon, rankdata, spearmanr, kendalltau
from scipy.spatial.distance import cdist
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================
N_COMPONENTS = 5
DEPTH        = 2
n_qubits     = N_COMPONENTS
K_NEIGHBORS  = 3

QSENN_K_SAFELEVEL, QSENN_K_SYNTH, QSENN_K_ENN, QSENN_TARGET = 5, 5, 3, 1.0
QS_N_CLUSTERS, QS_ANGLE_FRAC, QS_TARGET = 3, 0.15, 1.0
QSV2_SPLIT_FACTOR, QSV2_NUM_BINS, QSV2_BOOST_FACTOR = 10, 5, 1.5

FMAP_PARAMS = {"s": np.ones(n_qubits),
               "beta":  np.linspace(0.5, 1.0, DEPTH),
               "gamma": np.linspace(0.3, 0.7, DEPTH)}

DESIGN_SEEDS = [42, 202, 777]
VAL_SEEDS    = [11, 23, 37]
DIAG_SEED    = 11

DATA_DIR = ""
OUT_DIR  = "./hasil_qsenn/"

PROFIL   = "medium"
_BUDGET_FAST = dict(nested_seeds=[11], inner_splits=3, nested_folds=5,
                    rep_seeds=[11], rep_folds=5, shot_seeds=[11],
                    shot_folds=5, shot_realisasi=3)
_BUDGET_MEDIUM = dict(nested_seeds=[11], inner_splits=5, nested_folds=10,
                      rep_seeds=[11], rep_folds=10, shot_seeds=[11],
                      shot_folds=10, shot_realisasi=5)
_BUDGET_FULL = dict(nested_seeds=[11, 23, 37], inner_splits=5, nested_folds=10,
                    rep_seeds=[11, 23, 37], rep_folds=10, shot_seeds=[11, 23, 37],
                    shot_folds=10, shot_realisasi=10)
_BUDGETS = {
    "fast": _BUDGET_FAST, "medium": _BUDGET_MEDIUM, "full": _BUDGET_FULL,
    "cepat": _BUDGET_FAST, "sedang": _BUDGET_MEDIUM, "penuh": _BUDGET_FULL,
}
BUDGET = _BUDGETS[PROFIL]


def init(data_dir=None, out_dir=None, profil=None, profile=None, verbose=True):
    """Configure data/output directories and the computation-budget profile. Call once before running experiments."""
    global DATA_DIR, OUT_DIR, PROFIL, BUDGET
    if profile is not None:
        profil = profile
    if data_dir is not None:
        DATA_DIR = data_dir
    if out_dir is not None:
        OUT_DIR = out_dir
    elif data_dir is not None:
        OUT_DIR = (data_dir.replace("/DATA/", "/HASIL_QSENN/")
                   if "/DATA/" in data_dir else os.path.join(data_dir, "hasil_qsenn"))
    if profil is not None:
        PROFIL = profil; BUDGET = _BUDGETS[profil]
    os.makedirs(OUT_DIR, exist_ok=True)
    if verbose:
        print(f"DATA_DIR : {DATA_DIR}")
        print(f"OUT_DIR  : {OUT_DIR}")
        print(f"Profile  : {PROFIL} -> {BUDGET}")
        done = sorted(os.path.basename(f)[6:-4] for f in glob.glob(outp("folds_*.csv"))
                      if "ALL" not in f)
        print(f"Completed: {done if done else 'none'}")
    return OUT_DIR


def outp(fname):
    return os.path.join(OUT_DIR, fname)


try:
    from IPython.display import display          # noqa: F401
except Exception:
    def display(*args, **kwargs):
        for a in args:
            print(a)



# ============================================================

# ============================================================
# ============================================================

#     AND further methodological alignment with the reference literature)

#     were previously fit ONCE on the full/raw dataset before any
#     train/test split existed. This helper fits each of them on the
#     TRAINING fold only and transforms the test fold with those same
#     fitted objects/statistics.

#     METHODOLOGICAL CHANGES from an earlier version of this pipeline:
#     (1) Raw-feature IQR-based row removal was dropped entirely. The
#         reference methods this study compares against (Quantum-SMOTE,
#         Mohanty et al. 2024; Quantum-SMOTEV2, Mohanty et al. 2025) do
#         not remove rows via IQR in their own data preparation; their
#         only IQR use identifies "Angular Outliers" among already-
#         synthesized minority points purely to ADD more synthetic
#         samples (never to delete rows), which quantum_smotev2() in
#         this notebook already replicates faithfully.
#     (2) Median imputation is now a single, class-AGNOSTIC statistic
#         (not computed separately per class). A shuffled-label control
#         test revealed that class-conditional imputation can itself
#         act as a leakage-like channel on datasets with heavy
#         missingness: rows sharing a missing-value pattern are all
#         filled with their (random) class's median, letting a
#         classifier detect class membership from the imputed value
#         alone, independent of any real signal. Using one training-fold
#         median for all classes removes this artifact (verified: mean
#         AUC on shuffled labels drops from ~0.75 to ~0.55, consistent
#         with chance-level noise at this sample size).
# ============================================================
N_COMPONENTS = 5

def make_fold_data(X_raw, y_arr, tr, te, n_components=N_COMPONENTS):
    """Fit median imputation, Min-Max scaling, PCA, and post-PCA scaling on the training fold only, then transform the held-out fold with the fitted objects."""


    Xtr_raw, Xte_raw = X_raw[tr].astype(float), X_raw[te].astype(float)
    ytr_raw, yte_raw = y_arr[tr], y_arr[te]

    imputer = SimpleImputer(strategy="median")
    Xtr_imp = imputer.fit_transform(Xtr_raw)
    Xte_imp = imputer.transform(Xte_raw)

    scaler_raw = MinMaxScaler()
    Xtr_scaled = scaler_raw.fit_transform(Xtr_imp)
    Xte_scaled = scaler_raw.transform(Xte_imp)

    ncomp = min(n_components, Xtr_scaled.shape[1])
    pca = PCA(n_components=ncomp, random_state=42)
    Xtr_pca_raw = pca.fit_transform(Xtr_scaled)
    Xte_pca_raw = pca.transform(Xte_scaled)

    scaler_pca = MinMaxScaler()
    Xtr_pca = scaler_pca.fit_transform(Xtr_pca_raw)
    Xte_pca = scaler_pca.transform(Xte_pca_raw)

    return Xtr_pca, Xte_pca, ytr_raw, yte_raw, pca.explained_variance_ratio_


EVR_LOG = []

def log_evr(evr, tag=""):
    EVR_LOG.append(dict(tag=tag, cum_var=float(np.sum(evr))))
    return evr

def evr_summary():
    if not EVR_LOG:
        return "no folds have been run"
    v = np.array([r["cum_var"] for r in EVR_LOG]) * 100
    return f"Retained variance {len(v)} fold: {v.mean():.1f}% +- {v.std(ddof=1):.1f}%"


# ============================================================
# QUANTUM FEATURE MAP + KERNEL
# ============================================================
# ============================================================

# ============================================================
n_qubits = N_COMPONENTS  # fixed architecture parameter (was X_pca.shape[1]; PCA is now fit per-fold)
DEPTH    = 2

FMAP_PARAMS = {
    "s":     np.ones(n_qubits),
    "beta":  np.linspace(0.5, 1.0, DEPTH),
    "gamma": np.linspace(0.3, 0.7, DEPTH),
}

def ring_entanglement(wires):
    wires = list(wires); L = len(wires)
    for i in range(L):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % L]])

def feature_block(x, wires, params=FMAP_PARAMS):
    """Apply RY(pi*s*x), ring-CNOT entanglement, and RZ(beta*x + gamma*x^2) for DEPTH layers."""
    wires = list(wires)
    s, beta, gamma = params["s"], params["beta"], params["gamma"]
    for l in range(DEPTH):
        for i, w in enumerate(wires):
            qml.RY(np.pi * s[i] * float(x[i]), wires=w)
        ring_entanglement(wires)
        for i, w in enumerate(wires):
            qml.RZ(beta[l] * float(x[i]) + gamma[l] * float(x[i]) ** 2, wires=w)


dev = qml.device("default.qubit", wires=n_qubits, shots=None) if qml is not None else None


def _needs_qml(fn):
    def _wrap(*a, **k):
        if qml is None:
            raise RuntimeError(f"{fn.__name__} requires PennyLane; "
                               "use fast_statevectors instead.")
        return fn(*a, **k)
    return _wrap

@(qml.qnode(dev) if qml is not None else _needs_qml)
def _kernel_probs(x1, x2):
    feature_block(x1, wires=range(n_qubits))
    qml.adjoint(feature_block)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))


@(qml.qnode(dev) if qml is not None else _needs_qml)
def _statevector(x):
    feature_block(x, wires=range(n_qubits))
    return qml.state()

def quantum_statevectors(X):
    """Encode each row of X as |phi(x)> and return a complex array of shape (N, 2**n_qubits)."""
    return np.array([np.asarray(_statevector(x)) for x in X])

def fidelity_distmat(PsiA, PsiB):
    """Return quantum distance 1 - |<phi_a|phi_b>|^2 from statevectors using NumPy operations."""
    M = PsiA.conj() @ PsiB.T
    return np.maximum(1.0 - np.abs(M) ** 2, 0.0)

quantum_distance_cache = {}
def _key(a): return tuple(np.round(np.asarray(a, dtype=float), 6))

def quantum_distance_cached(a, b):
    ka, kb = _key(a), _key(b)
    if (ka, kb) in quantum_distance_cache: return quantum_distance_cache[(ka, kb)]
    if (kb, ka) in quantum_distance_cache: return quantum_distance_cache[(kb, ka)]
    d = 1.0 - float(_kernel_probs(a, b)[0])
    quantum_distance_cache[(ka, kb)] = d
    return d


# ============================================================
# VECTORIZED ENCODER
# ============================================================
# ============================================================


# ============================================================
_BITS = np.array([[(j >> (n_qubits - 1 - w)) & 1 for w in range(n_qubits)]
                  for j in range(2 ** n_qubits)])
_SGN  = np.where(_BITS == 0, -1.0, 1.0)


def _ent_pairs(ent, n=None):
    n = n_qubits if n is None else n
    if ent == "ring":   return [(c, (c + 1) % n) for c in range(n)]
    if ent == "linear": return [(c, c + 1) for c in range(n - 1)]
    if ent == "full":   return [(c, t) for c in range(n) for t in range(c + 1, n)]
    if ent == "none":   return []
    raise ValueError(ent)


def _perm_inv(ent, n=None):
    """Return the inverse computational-basis permutation for one CNOT layer."""
    n = n_qubits if n is None else n
    D = 2 ** n
    img = np.arange(D)
    for c, t in _ent_pairs(ent, n):
        nxt = np.empty(D, int)
        for j in range(D):
            nxt[j] = (j ^ (1 << (n - 1 - t))) if ((j >> (n - 1 - c)) & 1) else j
        img = nxt[img]
    inv = np.empty(D, int); inv[img] = np.arange(D)
    return inv


_PERM_CACHE = {}
def _perm_for(ent, n=None):
    n = n_qubits if n is None else n
    if (ent, n) not in _PERM_CACHE:
        _PERM_CACHE[(ent, n)] = _perm_inv(ent, n)
    return _PERM_CACHE[(ent, n)]


def fast_statevectors(X, depth=None, ent="ring", params=None, n=None):
    """Vectorized NumPy implementation of the fixed quantum feature map for an entire batch."""
    n     = n_qubits if n is None else n
    depth = DEPTH if depth is None else depth
    p     = FMAP_PARAMS if params is None else params
    sv, beta, gamma = p["s"], p["beta"], p["gamma"]
    D    = 2 ** n
    bits = _BITS if n == n_qubits else np.array(
        [[(j >> (n - 1 - w)) & 1 for w in range(n)] for j in range(D)])
    sgn  = _SGN if n == n_qubits else np.where(bits == 0, -1.0, 1.0)
    perm = _perm_for(ent, n)

    X = np.ascontiguousarray(np.asarray(X, float))
    N = len(X)
    psi = np.zeros((N, D), complex); psi[:, 0] = 1.0
    for l in range(depth):
        for i in range(n):                                   # RY
            th = (np.pi * sv[i] * X[:, i]) / 2.0
            c_, s_ = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
            v = psi.reshape(N, 2 ** i, 2, 2 ** (n - 1 - i))
            a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
            psi = np.stack([c_ * a0 - s_ * a1, s_ * a0 + c_ * a1], axis=2).reshape(N, D)
        psi = psi[:, perm]
        PHI = beta[l] * X[:, :n] + gamma[l] * X[:, :n] ** 2   # RZ (diagonal)
        psi = psi * np.exp(1j * 0.5 * (PHI @ sgn.T))
    return psi


# ============================================================
# DISTANCE COMPONENTS + KNN
# ============================================================
# ============================================================

# ============================================================
def _chi2_cdist(A, B, eps=1e-10):
    out = np.zeros((len(A), len(B)))
    for j in range(len(B)):
        diff = A - B[j]
        out[:, j] = np.sum(diff * diff / (A + B[j] + eps), axis=1)
    return out

def _polar_cdist(A, B, eps=1e-12):
    nA = np.linalg.norm(A, axis=1, keepdims=True) + eps
    nB = np.linalg.norm(B, axis=1, keepdims=True) + eps
    cos = np.clip((A @ B.T) / (nA * nB.T), -1.0, 1.0)
    return np.arccos(cos) / np.pi

def classical_distmat(A, B, metric, VI=None):
    if metric == "euclidean":   return cdist(A, B, "euclidean")
    if metric == "manhattan":   return cdist(A, B, "cityblock")
    if metric == "canberra":    return cdist(A, B, "canberra")
    if metric == "chebyshev":   return cdist(A, B, "chebyshev")
    if metric == "chi-square":  return _chi2_cdist(A, B)
    if metric == "polar":       return _polar_cdist(A, B)
    if metric == "mahalanobis": return cdist(A, B, "mahalanobis", VI=VI)
    raise ValueError(metric)

def hybrid_component(A, B, metric, VI=None):
    if metric == "euclidean":
        return np.zeros((len(A), len(B)))
    return classical_distmat(A, B, metric, VI=VI)

def quantum_base_distmat(A, B, symmetric=False):
    nA, nB = len(A), len(B); D = np.zeros((nA, nB))
    if symmetric:
        for i in range(nA):
            for j in range(i + 1, nB):
                d = quantum_distance_cached(A[i], B[j]); D[i, j] = D[j, i] = d
    else:
        for i in range(nA):
            for j in range(nB):
                D[i, j] = quantum_distance_cached(A[i], B[j])
    return D

def knn_proba_pos_from_distmat(D, y_train, k, pos_label=None, exclude_self=False):
    """Estimate positive-class probability from a precomputed distance matrix using the k nearest neighbors."""


    if pos_label is None:


        _c, _n = np.unique(np.asarray(y_train), return_counts=True)
        if len(_c) == 2 and _n.max() / _n.min() < 1.05:
            print("WARNING knn_proba: pos_label was not provided and the classes are "
                  "nearly balanced, so inferring the smaller class is not "
                  "reliable. Provide pos_label explicitly.")
        pos_label = pos_neg_of(y_train)[0]
    Dq = D.copy()
    if exclude_self: np.fill_diagonal(Dq, np.inf)
    y_train = np.asarray(y_train)
    k_eff = min(k, Dq.shape[1] - (1 if exclude_self else 0))
    proba = np.empty(Dq.shape[0])
    for i in range(Dq.shape[0]):
        idx = np.argpartition(Dq[i], k_eff)[:k_eff]
        proba[i] = np.mean(y_train[idx] == pos_label)
    return proba


def sk_proba_pos(clf, X, pos_label=None):
    """Return the probability of the requested positive class from a fitted scikit-learn classifier."""


    if pos_label is None:



        # pemanggil menyebutkannya secara eksplisit.
        pos_label = globals().get("POS", list(clf.classes_)[-1])
    classes = list(clf.classes_)
    return clf.predict_proba(X)[:, classes.index(pos_label)]


knn_proba1_from_distmat = knn_proba_pos_from_distmat

def compute_mahalanobis_VI(X, reg=1e-6):
    cov = np.cov(X, rowvar=False) + reg * np.eye(X.shape[1])
    return np.linalg.inv(cov)


# ============================================================
# QS-ENN + QUANTUM-SMOTE PRIMITIVES
# ============================================================
# ============================================================

# ============================================================
from sklearn.cluster import KMeans
QSENN_K_SAFELEVEL, QSENN_K_SYNTH, QSENN_K_ENN, QSENN_TARGET = 5, 5, 3, 1.0
QS_N_CLUSTERS, QS_ANGLE_FRAC, QS_TARGET = 3, 0.15, 1.0     # param Quantum-SMOTE

def swap_fidelity_distmat(PsiA, PsiB):
    """Return the QKNN fidelity distance equivalent to a noiseless SWAP-test overlap estimate."""
    return fidelity_distmat(PsiA, PsiB)

def phase_align(psi_ref, psi):
    ov = np.vdot(psi_ref, psi)
    return psi if np.abs(ov) < 1e-12 else psi * np.exp(-1j * np.angle(ov))

def slerp_state(psi_i, psi_j, lam):
    psi_j = phase_align(psi_i, psi_j)
    c = np.clip(np.real(np.vdot(psi_i, psi_j)), -1.0, 1.0); theta = np.arccos(c)
    out = psi_i.copy() if theta < 1e-7 else \
          (np.sin((1-lam)*theta)*psi_i + np.sin(lam*theta)*psi_j) / np.sin(theta)
    return out / (np.linalg.norm(out) + 1e-12)

# ---------- QS-ENN ----------
def quantum_safe_levels(Psi, y_, min_label, k):
    D = fidelity_distmat(Psi, Psi); np.fill_diagonal(D, np.inf)
    y_ = np.asarray(y_); keff = min(k, len(y_)-1); r = np.zeros(len(y_))
    for i in range(len(y_)):
        r[i] = np.mean(y_[np.argpartition(D[i], keff)[:keff]] == min_label)
    return r

def protective_qenn(Psi, y_, is_original, min_label, maj_label, k):
    D = fidelity_distmat(Psi, Psi); np.fill_diagonal(D, np.inf)
    y_ = np.asarray(y_); keff = min(k, len(y_)-1); keep = np.ones(len(y_), bool)
    for i in range(len(y_)):
        idx = np.argpartition(D[i], keff)[:keff]
        pred = min_label if np.mean(y_[idx] == min_label) > 0.5 else maj_label
        if pred == y_[i]: continue
        keep[i] = True if (y_[i] == min_label and is_original[i]) else False
    return keep

def qs_enn(Psi_tr, y_tr, k_sl=QSENN_K_SAFELEVEL, k_syn=QSENN_K_SYNTH,
                 k_enn=QSENN_K_ENN, target_ratio=QSENN_TARGET, rng=None):
    rng = rng if rng is not None else np.random.RandomState(0); y_tr = np.asarray(y_tr)
    cl, ct = np.unique(y_tr, return_counts=True)
    min_label, maj_label = int(cl[ct.argmin()]), int(cl[ct.argmax()])
    n_min, n_maj = int(ct.min()), int(ct.max())
    mmask = (y_tr == min_label); Psi_min = Psi_tr[mmask]
    if len(Psi_min) < 2:
        return Psi_tr.copy(), y_tr.copy(), dict(n_synth=0, n_removed=0)
    r = quantum_safe_levels(Psi_tr, y_tr, min_label, k_sl)[mmask]; el = r > 0.0
    G = max(int(round(target_ratio*n_maj)) - n_min, 0)
    w = (1.0 - r) * el
    if w.sum() <= 0: w = el.astype(float); w = np.ones_like(r) if w.sum() <= 0 else w
    w = w / w.sum(); g = np.floor(G*w).astype(int); rem = int(G - g.sum())
    if rem > 0:
        for a in rng.choice(len(g), rem, p=w): g[a] += 1
    Dmm = fidelity_distmat(Psi_min, Psi_min); np.fill_diagonal(Dmm, np.inf)
    ke = min(k_syn, len(Psi_min)-1); synth = []
    for si in range(len(Psi_min)):
        if g[si] == 0: continue
        nb = np.argpartition(Dmm[si], ke)[:ke]
        for _ in range(int(g[si])):
            synth.append(slerp_state(Psi_min[si], Psi_min[int(rng.choice(nb))], float(rng.uniform(0,1))))
    synth = np.array(synth) if synth else np.empty((0, Psi_tr.shape[1]), complex)
    Psi_all = np.vstack([Psi_tr, synth]) if len(synth) else Psi_tr.copy()
    y_all   = np.concatenate([y_tr, np.full(len(synth), min_label)])
    is_orig = np.concatenate([np.ones(len(y_tr), bool), np.zeros(len(synth), bool)])
    keep = protective_qenn(Psi_all, y_all, is_orig, min_label, maj_label, k_enn)
    return Psi_all[keep], y_all[keep], dict(n_synth=int(len(synth)), n_removed=int((~keep).sum()))


def quantum_smote(Psi_tr, y_tr, X_feat, n_clusters=QS_N_CLUSTERS,
                  angle_frac=QS_ANGLE_FRAC, target_ratio=QS_TARGET, rng=None):
    rng = rng if rng is not None else np.random.RandomState(0); y_tr = np.asarray(y_tr)
    cl, ct = np.unique(y_tr, return_counts=True)
    ml = int(cl[ct.argmin()]); n_min, n_maj = int(ct.min()), int(ct.max())
    k = min(n_clusters, len(X_feat))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_feat)
    cent_states = quantum_statevectors(km.cluster_centers_)
    lab = km.labels_
    midx = np.where(y_tr == ml)[0]
    G = max(int(round(target_ratio*n_maj)) - n_min, 0)
    base = G // len(midx); extra = G - base*len(midx)
    g_per = np.full(len(midx), base); g_per[:extra] += 1
    synth = []
    for s, gi in zip(midx, g_per):
        if gi == 0: continue
        c_state = cent_states[lab[s]]
        ov = np.clip(np.abs(np.vdot(Psi_tr[s], c_state)), 0.0, 1.0)
        theta = np.arccos(ov)
        for t in range(int(gi)):
            lam = min(angle_frac*(t+1), 0.9)
            synth.append(slerp_state(Psi_tr[s], c_state, lam))
    synth = np.array(synth) if synth else np.empty((0, Psi_tr.shape[1]), complex)
    Psi_all = np.vstack([Psi_tr, synth]) if len(synth) else Psi_tr.copy()
    y_all = np.concatenate([y_tr, np.full(len(synth), ml)])
    return Psi_all, y_all, dict(n_synth=int(len(synth)), n_clusters=k)


# ============================================================
# QUANTUM-SMOTEV2
# ============================================================
# ============================================================


#     and relevant quantum oversampling competitor than the original
#     Quantum-SMOTE baseline above.

#     Reimplementation note: the original paper encodes data by amplitude
#     embedding and applies its rotation as an RX gate on every qubit of a
#     compact swap-test circuit. This notebook's shared representation
#     instead uses fidelity-derived angles and SLERP for every quantum
#     resampling method (QS-ENN and Quantum-SMOTE V1 above), so
#     Quantum-SMOTEV2 is reimplemented on that same shared representation,
#     following the same convention already used for the V1 baseline. The
#     three ingredients that distinguish V2 from V1 are reproduced exactly:
#       (a) a SINGLE data centroid for the whole minority class, removing
#           V1's K-means clustering step;
#       (b) the rotation-angle magnitude selected by the paper's own
#           piecewise rule (Algorithm "Angle of rotation calculation
#           logic"), relative to each point's own angular distance to that
#           centroid, scaled by a split_factor;
#       (c) Angular Outlier (AOL) detection via Tukey's IQR rule on the
#           resulting angular-distance distribution, and AOL "boosting":
#           extra synthetic points placed preferentially in
#           under-represented outlier bins using a wider rotation
#           magnitude, exactly as specified in the paper.
#     One deviation from the literal circuit is necessary and is flagged
#     explicitly: V2's RX-gate rotation is applied in a fixed axis
#     unrelated to any second reference point (it nudges the point by a
#     small amount, keeping it close to itself); since this notebook's
#     shared representation only defines "rotate by angle theta" via SLERP
#     between two states, that same small-angle rotation is approximated
#     here as a low-lambda SLERP toward a randomly chosen fellow minority
#     state, which preserves the paper's stated intent -- "the synthetic
#     data points are positioned in proximity to the original minority
#     data point" -- rather than migrating the point toward the centroid
#     the way V1's rotation does.
# ============================================================
QSV2_SPLIT_FACTOR, QSV2_TARGET, QSV2_NUM_BINS, QSV2_BOOST_FACTOR = 10, 1.0, 5, 1.5

def _v2_centroid_state(Psi_set):
    """Return the global minority centroid state as the leading eigenvector of the density matrix."""

    rho = (Psi_set.T @ Psi_set.conj()) / len(Psi_set)
    rho = (rho + rho.conj().T) / 2.0
    w, v = np.linalg.eigh(rho)
    c = v[:, -1]
    return c / (np.linalg.norm(c) + 1e-12)

def _slerp_by_angle(psi_a, psi_b, ang):
    """Rotate psi_a toward psi_b by an absolute geodesic angle in radians using SLERP."""


    ov = np.clip(np.abs(np.vdot(psi_a, psi_b)), 0.0, 1.0)
    theta_ab = float(np.arccos(ov))
    if theta_ab < 1e-9:
        return psi_a.copy()
    lam = float(min(max(ang, 0.0) / theta_ab, 1.0))
    return slerp_state(psi_a, psi_b, lam)


def _v2_rotation_angle(ang_dist, sf, rng):
    """Implement the Quantum-SMOTEV2 angle-of-rotation rule."""
    if ang_dist > np.pi / 2:
        return abs(np.pi / 2 - ang_dist) / sf
    elif ang_dist < 0:  # not reachable under exact statevector fidelity; kept for fidelity to the paper
        return abs((np.pi / 2 - ang_dist) * rng.uniform(0.5, 1.0)) / sf
    else:
        return rng.uniform(0.0, ang_dist) / sf

def quantum_smotev2(Psi_tr, y_tr, split_factor=QSV2_SPLIT_FACTOR, target_ratio=QSV2_TARGET,
                     num_bins=QSV2_NUM_BINS, boost_factor=QSV2_BOOST_FACTOR, rng=None):
    rng = rng if rng is not None else np.random.RandomState(0)
    y_tr = np.asarray(y_tr)
    cl, ct = np.unique(y_tr, return_counts=True)
    ml = int(cl[ct.argmin()]); n_min, n_maj = int(ct.min()), int(ct.max())
    mmask = (y_tr == ml); Psi_min = Psi_tr[mmask]
    if len(Psi_min) < 2:
        return Psi_tr.copy(), y_tr.copy(), dict(n_synth=0, n_boosted=0)

    # (a) single global minority centroid -- no K-means, unlike V1
    c_state = _v2_centroid_state(Psi_min)

    def _angle_to_centroid(psi):
        ov = np.clip(np.abs(np.vdot(psi, c_state)), 0.0, 1.0)
        return float(np.arccos(ov))
    theta_min = np.array([_angle_to_centroid(p) for p in Psi_min])

    G = max(int(round(target_ratio * n_maj)) - n_min, 0)
    base = G // len(Psi_min); extra = G - base * len(Psi_min)
    g_per = np.full(len(Psi_min), base); g_per[:extra] += 1  # uniform allocation (as in the paper, pre-AOL)

    synth, synth_theta = [], []
    for si in range(len(Psi_min)):
        if g_per[si] == 0:
            continue
        for _ in range(int(g_per[si])):
            ang = _v2_rotation_angle(theta_min[si], split_factor, rng)
            partner = Psi_min[int(rng.randint(len(Psi_min)))]
            new_state = _slerp_by_angle(Psi_min[si], partner, ang)
            synth.append(new_state)
            synth_theta.append(_angle_to_centroid(new_state))
    synth = np.array(synth) if synth else np.empty((0, Psi_tr.shape[1]), complex)
    synth_theta = np.array(synth_theta) if len(synth_theta) else np.empty(0)

    # (c) Angular Outlier detection + boosting, following the paper's Algorithms
    #     "Generate Outlier Datasets" and "Boosting Outlier Dataset"
    n_boosted = 0
    if len(synth) and num_bins > 0:
        all_theta = np.concatenate([theta_min, synth_theta])
        Q1, Q3 = np.percentile(all_theta, [25, 75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        out_mask = (synth_theta < lower) | (synth_theta > upper)
        if out_mask.sum() > 0:
            out_idx = np.where(out_mask)[0]
            counts, edges = np.histogram(synth_theta[out_idx], bins=num_bins)
            total_outlier = counts.sum()
            threshold = total_outlier / num_bins if num_bins else 0
            half_threshold = threshold / 2
            boosted = []
            for b in range(num_bins):
                if 0 < counts[b] < half_threshold:
                    in_bin = out_idx[(synth_theta[out_idx] >= edges[b]) & (synth_theta[out_idx] < edges[b + 1])]
                    n_boost = max(int(np.floor(threshold / counts[b])), 1)
                    for bi in in_bin:
                        for _ in range(n_boost):
                            ang = _v2_rotation_angle(theta_min[min(bi, len(theta_min) - 1)], split_factor, rng) * boost_factor
                            partner = Psi_min[int(rng.randint(len(Psi_min)))]
                            boosted.append(_slerp_by_angle(synth[bi], partner, ang))
            if boosted:
                synth = np.vstack([synth, np.array(boosted)])
                n_boosted = len(boosted)

    Psi_all = np.vstack([Psi_tr, synth]) if len(synth) else Psi_tr.copy()
    y_all = np.concatenate([y_tr, np.full(len(synth), ml)])
    return Psi_all, y_all, dict(n_synth=int(len(synth) - n_boosted), n_boosted=int(n_boosted))


# ============================================================

# ============================================================
# ============================================================


# ============================================================
from sklearn.cluster import KMeans

def centroid_state(Psi_set):
    """Return the leading eigenvector of the density matrix, which maximizes average fidelity to the input states."""

    rho = (Psi_set.T @ Psi_set.conj()) / len(Psi_set)
    rho = (rho + rho.conj().T) / 2.0
    w, v = np.linalg.eigh(rho)
    c = v[:, -1]
    return c / (np.linalg.norm(c) + 1e-12)

def qs_enn_variant(Psi_tr, y_tr, use_borderline=True, synth_mode="centroid_min", use_qenn=True,
               k_sl=QSENN_K_SAFELEVEL, k_syn=QSENN_K_SYNTH, k_enn=QSENN_K_ENN,
               target_ratio=QSENN_TARGET, X_feat=None, n_clusters=3,
               lam_max=1.0, rng=None, use_geodesic=True):
    """Run binary QS-ENN or an ablation variant. use_geodesic is retained for backward compatibility and does not change the current SLERP implementation."""
    rng = rng if rng is not None else np.random.RandomState(0)
    y_tr = np.asarray(y_tr)
    cl, ct = np.unique(y_tr, return_counts=True)
    min_label, maj_label = int(cl[ct.argmin()]), int(cl[ct.argmax()])
    n_min, n_maj = int(ct.min()), int(ct.max())
    mmask = (y_tr == min_label); Psi_min = Psi_tr[mmask]
    if len(Psi_min) < 2:
        return Psi_tr.copy(), y_tr.copy(), dict(n_synth=0, n_removed=0)


    if use_borderline:
        r = quantum_safe_levels(Psi_tr, y_tr, min_label, k_sl)[mmask]
        el = r > 0.0
        w = (1.0 - r) * el
        if w.sum() <= 0:
            w = el.astype(float)
            if w.sum() <= 0: w = np.ones(len(Psi_min))
    else:
        w = np.ones(len(Psi_min))
    w = w / w.sum()

    G = max(int(round(target_ratio * n_maj)) - n_min, 0)
    g = np.floor(G * w).astype(int); rem = int(G - g.sum())
    if rem > 0:
        for a in rng.choice(len(g), rem, p=w): g[a] += 1


    if synth_mode == "neighbor":
        Dmm = fidelity_distmat(Psi_min, Psi_min); np.fill_diagonal(Dmm, np.inf)
        ke = min(k_syn, len(Psi_min) - 1)
    elif synth_mode in ("centroid_min", "centroid_global"):
        c_state = centroid_state(Psi_min)
    elif synth_mode == "centroid_cluster":
        if X_feat is None:
            raise ValueError("synth_mode='centroid_cluster' requires X_feat")
        Xm = X_feat[mmask]
        kk = min(n_clusters, len(Xm))
        km = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(Xm)
        lab = km.labels_
        c_states = np.array([centroid_state(Psi_min[lab == j]) if (lab == j).sum() > 0
                             else centroid_state(Psi_min) for j in range(kk)])
    else:
        raise ValueError(f"unknown synth_mode: {synth_mode}")

    synth = []
    for si in range(len(Psi_min)):
        if g[si] == 0: continue
        if synth_mode == "neighbor":
            nb = np.argpartition(Dmm[si], ke)[:ke]
        for _ in range(int(g[si])):
            lam = float(rng.uniform(0, lam_max))
            if synth_mode == "neighbor":
                partner = Psi_min[int(rng.choice(nb))]
            elif synth_mode in ("centroid_min", "centroid_global"):
                partner = c_state
            else:
                partner = c_states[lab[si]]
            synth.append(slerp_state(Psi_min[si], partner, lam))
    synth = np.array(synth) if synth else np.empty((0, Psi_tr.shape[1]), complex)

    Psi_all = np.vstack([Psi_tr, synth]) if len(synth) else Psi_tr.copy()
    y_all   = np.concatenate([y_tr, np.full(len(synth), min_label)])
    is_orig = np.concatenate([np.ones(len(y_tr), bool), np.zeros(len(synth), bool)])


    keep = protective_qenn(Psi_all, y_all, is_orig, min_label, maj_label, k_enn) if use_qenn \
           else np.ones(len(y_all), bool)


    rm = ~keep
    y_keep = y_all[keep]
    n_min_after = int((y_keep == min_label).sum())
    n_maj_after = int((y_keep == maj_label).sum())
    stats = dict(
        n_synth=int(len(synth)),
        n_removed=int(rm.sum()),
        n_removed_maj_orig=int((rm & is_orig & (y_all == maj_label)).sum()),
        n_removed_min_orig=int((rm & is_orig & (y_all == min_label)).sum()),
        n_removed_synth=int((rm & ~is_orig).sum()),
        n_min_before=int(n_min), n_maj_before=int(n_maj),
        n_min_after=n_min_after, n_maj_after=n_maj_after,
        ir_before=float(n_maj) / max(int(n_min), 1),
        ir_after=float(n_maj_after) / max(n_min_after, 1),
    )
    assert stats["n_removed_min_orig"] == 0, "Pillar 3 invariant violated: an original minority sample was removed"
    return Psi_all[keep], y_all[keep], stats


# ============================================================
# CANONICAL STATISTICS (Wilcoxon + Holm)
# ============================================================
# ============================================================


#             Delta = QS-ENN  -  baseline


# ============================================================
from scipy.stats import wilcoxon, rankdata

DELTA_BASELINE_CONVENTION = "Delta = QS-ENN - baseline; Delta > 0 => QS-ENN performs better"
DELTA_ABLATION_CONVENTION = ("Delta = ablated variant - full QS-ENN; "
                             "Delta > 0 => removing the component improves the metric")


def wilcoxon_full(a, b):
    """Compute a paired Wilcoxon signed-rank comparison and return delta, rank sums, W, z, p, and effect size r."""


    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    dz = d[d != 0]; n = len(dz)
    if n == 0:
        return dict(N=0, Tplus=0.0, Tminus=0.0, W=0.0, z=np.nan, p=1.0, r=0.0,
                    delta=float(np.mean(d)))
    rk = rankdata(np.abs(dz))
    Tp = float(rk[dz > 0].sum()); Tm = float(rk[dz < 0].sum())
    W = min(Tp, Tm)
    mu = n * (n + 1) / 4.0
    _, tc = np.unique(np.abs(dz), return_counts=True)
    tie = np.sum(tc ** 3 - tc)
    sg = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    z = (W - mu) / sg if sg > 0 else np.nan
    try:
        p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue
    except Exception:
        p = np.nan
    return dict(N=n, Tplus=Tp, Tminus=Tm, W=W, z=z, p=p,
                r=(abs(z) / np.sqrt(n) if np.isfinite(z) else np.nan),
                delta=float(np.mean(d)))


def holm(pv):
    """Apply the Holm-Bonferroni step-down correction to a sequence of p-values."""
    p = np.asarray(pv, float); m = len(p)
    o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def stat_table(df, ref_arm, others, metric, mode="baseline",
               index=("rep", "fold"), title=""):
    """Build a Wilcoxon-plus-Holm comparison table using the requested baseline or ablation delta convention."""


    Pv = df.pivot_table(index=list(index), columns="arm", values=metric)
    if ref_arm not in Pv.columns:
        raise KeyError(f"Reference arm '{ref_arm}' is not available. Available: {list(Pv.columns)}")
    recs = []
    for o in others:
        if o not in Pv.columns:
            continue
        x, yv = (Pv[ref_arm].values, Pv[o].values) if mode == "baseline" \
                else (Pv[o].values, Pv[ref_arm].values)
        r_ = wilcoxon_full(x, yv)
        recs.append({"Comparator": o.strip(), "Δ": round(r_["delta"], 4),
                     "N": r_["N"], "T+": r_["Tplus"], "T−": r_["Tminus"], "W": r_["W"],
                     "z": round(r_["z"], 3) if np.isfinite(r_["z"]) else np.nan,
                     "p": round(r_["p"], 5) if np.isfinite(r_["p"]) else np.nan,
                     "effect size r": round(r_["r"], 3) if np.isfinite(r_["r"]) else np.nan})
    t = pd.DataFrame(recs)
    if len(t):
        t["p (Holm)"] = np.round(holm(t["p"].fillna(1.0).values), 5)
        t["significant"] = np.where(t["p (Holm)"] < 0.05, "✅", "—")
    if title:
        print(f"\n=== {title} — metric: {metric} ===")
        display(t)
    return t


# ============================================================
# GEOMETRY DESCRIPTORS + PROPOSITION 1
# ============================================================
# ============================================================


# ============================================================
def centroid_of(Psi_set):
    rho = (Psi_set.T @ Psi_set.conj()) / len(Psi_set)
    rho = (rho + rho.conj().T) / 2.0
    w, v = np.linalg.eigh(rho)
    return v[:, -1] / (np.linalg.norm(v[:, -1]) + 1e-12), float(w[-1])


def fold_descriptors(Ptr, ytr, k_sl=None, k_enn=None, rng=None):
    """Compute quantum-neighborhood geometry descriptors for one training fold."""
    k_sl  = QSENN_K_SAFELEVEL if k_sl  is None else k_sl
    k_enn = QSENN_K_ENN       if k_enn is None else k_enn
    rng   = np.random.RandomState(0) if rng is None else rng
    pos_f, neg_f = pos_neg_of(ytr)
    ytr = np.asarray(ytr)
    Pmin, Pmaj = Ptr[ytr == pos_f], Ptr[ytr == neg_f]

    if len(Pmin) < 2 or len(Pmaj) < 2:



        return {k: np.nan for k in
                ["minority_purity", "minority_outlier_fraction", "class_overlap",
                 "centroid_concentration", "fraction_toward_majority_centroid",
                 "synthetic_contamination", "editing_asymmetry"]}

    r = quantum_safe_levels(Ptr, ytr, pos_f, k_sl)[ytr == pos_f]
    mu, lam1 = centroid_of(Pmin)
    nu, _    = centroid_of(Pmaj)


    ov_mu_nu = float(np.abs(np.vdot(mu, nu)))
    ov_i_nu  = np.abs(Pmin.conj() @ nu)
    contaminated = float(np.mean(ov_i_nu < ov_mu_nu))

    overlap_angle = float(np.arccos(np.clip(ov_mu_nu, 0.0, 1.0)))


    Pres, yres, st = qs_enn_variant(Ptr, ytr, rng=rng)
    n_before = len(ytr)
    Pall = np.vstack([Ptr, np.empty((0, Ptr.shape[1]), complex)])
    Psyn_est = Pres[n_before:] if len(Pres) > n_before else Pres[:0]
    if len(Psyn_est):
        D = fidelity_distmat(Psyn_est, Ptr)
        keff = max(1, min(5, D.shape[1] - 1))
        lost = 0
        for i in range(len(Psyn_est)):
            idx = np.argpartition(D[i], keff)[:keff]
            if np.mean(ytr[idx] == pos_f) <= 0.5:
                lost += 1
        synth_contam = lost / len(Psyn_est)
    else:
        synth_contam = np.nan
    enn_asym = (st["n_removed_maj_orig"] / st["n_removed"]) if st["n_removed"] else np.nan

    return dict(
        minority_purity = float(np.mean(r)),
        minority_outlier_fraction = float(np.mean(r == 0)),
        class_overlap = overlap_angle,
        centroid_concentration = lam1,
        fraction_toward_majority_centroid = contaminated,
        synthetic_contamination = float(synth_contam) if synth_contam == synth_contam else np.nan,
        editing_asymmetry = float(enn_asym) if enn_asym == enn_asym else np.nan,
    )


desc_rows = []

desc_df = pd.DataFrame(desc_rows)


# ============================================================

# ============================================================
# ============================================================

# ============================================================
import contextlib
from scipy.stats import kendalltau

_FID_EXACT = fidelity_distmat

def fidelity_finite_shot(F_exact, shots, rng):
    """Estimate fidelity with finite-shot SWAP-test sampling: F_hat = 2*p_hat - 1."""
    if shots is None:
        return F_exact
    p = np.clip((1.0 + np.asarray(F_exact)) / 2.0, 0.0, 1.0)
    Fh = 2.0 * rng.binomial(shots, p) / float(shots) - 1.0
    return np.clip(Fh, 0.0, 1.0)


@contextlib.contextmanager
def finite_shot_mode(shots, rng):
    """Temporarily replace exact fidelity with finite-shot estimates throughout the QS-ENN pipeline."""

    global fidelity_distmat
    if shots is None:
        yield; return
    def noisy(A, B):
        F = 1.0 - _FID_EXACT(A, B)
        return 1.0 - fidelity_finite_shot(F, shots, rng)
    fidelity_distmat = noisy
    try:
        yield
    finally:
        fidelity_distmat = _FID_EXACT


# ============================================================
# DATASET REGISTRY + load_dataset()
# ============================================================
# ============================================================



# ============================================================
REGISTRY = {
    "cervical": dict(
        name="Cervical Cancer",
        file=["cervical_cancer.csv", "risk_factors_cervical_cancer.csv", "kag_risk_factors_cervical_cancer.csv"],
        target=["Biopsy"],



        drop=["Hinselmann", "Schiller", "Citology"],
        binarize=None, na=["?"]),

    "pima": dict(name="PIMA Indian Diabetes",
                 file=["diabetes.csv", "pima_diabetes.csv", "pima-indians-diabetes.csv"],
                 target=["Outcome", "Class"], drop=[], binarize=None, na=None),

    "frankfurt": dict(name="Frankfurt Hospital Diabetes",
                      file=["frankfurt_diabetes.csv", "diabetes_frankfurt.csv", "Diabetes.csv"],
                      target=["Outcome"], drop=[], binarize=None, na=None),

    "liver": dict(name="Liver Disease (ILPD)",
                  file=["indian_liver_patient.csv", "ilpd.csv", "liver.csv"],
                  target=["Dataset", "Selector", "Outcome"], drop=[],
                  binarize=("map", {1: 1, 2: 0}), na=None),

    "breast": dict(name="Breast Cancer Wisconsin",
                   file=["breast_cancer.csv", "data.csv", "wdbc.csv"],
                   target=["diagnosis", "Diagnosis"], drop=["id", "Unnamed: 32"],
                   binarize=("map", {"M": 1, "B": 0}), na=None),

    "heart": dict(name="Heart Disease",
                  file=["heart_disease.csv", "heart_disease_uci.csv", "heart.csv"],
                  target=["num", "target"], drop=["id", "dataset"],
                  binarize=("gt", 0), na=None),

    "hcv": dict(name="HCV (Hepatitis C)",
                file=["hcvdat0.csv", "HepatitisCdata.csv", "HCV-Egy-Data.csv",
                      "hcv.csv", "hcv_data.csv"],
                target=["Category"], drop=["Unnamed: 0"],
                binarize=("not_contains", "Blood Donor"), na=None),

    "lung": dict(name="Lung Cancer",
                 file=["lung_cancer.csv", "survey lung cancer.csv"],
                 target=["LUNG_CANCER"], drop=[],
                 binarize=("map", {"YES": 1, "NO": 0}), na=None),

    "typhoid": dict(name="Typhoid",
                    file=["typhoid.csv", "typhoid_data.csv", "typhoid_dataset.csv"],
                    target=["Final Output", "Diagnosis", "Result"],
                    drop=["Blood Culture", "Blood Culture Result", "Widal Test", "Widal",
                          "ESR", "ESR_increase", "WBC Count"],
                    binarize=None, na=None),

    "obesity": dict(name="Obesity",
                    file=["obesity.csv", "ObesityDataSet_raw_and_data_sinthetic.csv"],
                    target=["Label", "NObeyesdad", "Obesity"],
                    drop=["ID", "BMI"], binarize=None, na=None),
}

import os


_KEYWORDS = {
    "cervical":  ["cervical", "serviks", "biopsy", "risk_factor"],
    "pima":      ["pima", "indian", "diabetes"],
    "frankfurt": ["frankfurt", "diabetes"],
    "liver":     ["liver", "ilpd", "hati"],
    "breast":    ["breast", "wdbc", "wisconsin", "payudara"],
    "heart":     ["heart", "cardio", "jantung"],
    "hcv":       ["hcv", "hepatitis", "hepa"],
    "lung":      ["lung", "paru", "survey"],
    "typhoid":   ["typhoid", "tifoid", "tipes"],
    "obesity":   ["obesity", "obesitas", "nobeyesdad"],
}


def list_csv(folder=None):
    """List CSV files available in the configured data directory."""
    folder = DATA_DIR if folder is None else folder
    try:
        return sorted(f for f in os.listdir(folder or ".")
                      if f.lower().endswith(".csv"))
    except Exception:
        return []


def _pick_file(cand, key=None, verbose=True):
    """Locate a dataset file using explicit candidates first and keyword matching as a fallback."""
    for f in ([cand] if isinstance(cand, str) else cand):
        for base in (DATA_DIR, "", "./"):
            p = base + f
            if os.path.exists(p):
                return p
    if key is None:
        return None

    kk = _KEYWORDS.get(key, [key])
    skor = []
    for f in list_csv():
        fl = f.lower()
        n = sum(1 for k in kk if k in fl)
        if n:
            skor.append((n, -len(f), f))
    if not skor:
        return None
    skor.sort(reverse=True)
    pilih = skor[0][2]
    if verbose:
        lain = [f for _, _, f in skor[1:3]]
        print(f"[{key}] file {cand} was not found; using '{pilih}' "
              f"berdasarkan kecocokan nama."
              + (f" Alternatif: {lain}." if lain else "")
              + f" Tetapkan REGISTRY['{key}']['file'] if this is incorrect.")
    return (DATA_DIR + pilih) if os.path.exists(DATA_DIR + pilih) else pilih


def load_dataset(key, verbose=True):
    """Load and prepare one dataset without modifying global state; return features, labels, metadata, and class statistics."""


    cfg = REGISTRY[key]
    path = _pick_file(cfg["file"], key=key, verbose=verbose)
    if path is None:
        raise FileNotFoundError(
            f"[{key}] none of the candidate files were found: {cfg['file']} "
            f"(dicari di {DATA_DIR!r} dan direktori kerja)")

    d = pd.read_csv(path, na_values=cfg["na"])
    if "Gender" in d.columns:
        d["Gender"] = (d["Gender"].astype(str).str.strip().str.lower()
                       .map({"female": 1, "male": 0}).fillna(d["Gender"]))


    tgt = next((t for t in cfg["target"] if t in d.columns), None)
    if tgt is None:
        generik = ["Final Output", "Outcome", "outcome", "target", "Target",
                   "Class", "class", "label", "Label", "Diagnosis", "Result",
                   "Status", "y"]
        cocok = [c for c in generik if c in d.columns]
        if len(cocok) == 1:
            tgt = cocok[0]
            print(f"[{key}] configured candidates {cfg['target']} were not found; using "
                  f"generic column '{tgt}'. Tetapkan REGISTRY['{key}']['target'] "
                  f"if this is incorrect.")
        elif len(cocok) > 1:
            tgt = cocok[0]
            print(f"[{key}] multiple generic columns matched {cocok}; using "
                  f"'{tgt}'. CHECK this choice; if incorrect, set "
                  f"REGISTRY['{key}']['target'].")
        else:


            tgt = d.columns[-1]
            print(f"[{key}] WARNING: no configured candidate or generic target name "
                  f"matched. Falling back to the LAST column: '{tgt}'.")
            print(f"{'':12} Available columns: {list(d.columns)}")
            print(f"{'':12} Check the class distribution below; if incorrect, "
                  f"tetapkan REGISTRY['{key}']['target'] = ['nama_benar'].")

    dropped = [c for c in cfg["drop"] if c in d.columns]
    d = d.drop(columns=dropped)


    yr = d[tgt].copy()
    rule = cfg["binarize"]
    if rule is not None:
        kind, arg = rule
        if kind == "map":


            asli_kosong = int(yr.isna().sum())
            ymap = yr.map(arg)
            baru_kosong = int(ymap.isna().sum())
            if baru_kosong > asli_kosong:
                nilai = sorted(pd.unique(yr.dropna()))[:10]
                print(f"[{key}] WARNING: mapping rule {arg} does not match "
                      f"the values in the file ({nilai}). Mapping was skipped; labels "
                      f"are encoded as provided. Check the class distribution below.")
            else:
                yr = ymap
        elif kind == "gt":
            yr = (pd.to_numeric(yr, errors="coerce") > arg).astype(int)
        elif kind == "not_contains":
            cocok_str = yr.astype(str).str.contains(arg, case=False, na=False)
            if not cocok_str.any():
                print(f"[{key}] WARNING: no value contains "
                      f"'{arg}'. The not_contains rule was skipped.")
            else:
                yr = (~cocok_str).astype(int)
    if yr.isna().any():
        keep = ~yr.isna()
        print(f"[{key}] {int((~keep).sum())} rows were removed because the target label was missing.")
        d, yr = d[keep], yr[keep]

    Xd = d.drop(columns=[tgt])
    obj = Xd.select_dtypes(include=["object"]).columns.tolist()
    if obj:
        Xd = pd.get_dummies(Xd, columns=obj, drop_first=True)
    Xd = Xd.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xd = Xd.drop(columns=Xd.columns[Xd.isna().all()])

    yy = LabelEncoder().fit_transform(yr)
    cls, cnt = np.unique(yy, return_counts=True)
    meta = dict(key=key, name=cfg["name"], path=path, target=tgt, dropped=dropped,
                n=len(yy), n_feat=Xd.shape[1], n_class=len(cls),
                counts=dict(zip(cls.tolist(), cnt.tolist())),
                IR=float(cnt.max() / cnt.min()),
                multiclass=len(cls) > 2, missing_pct=float(Xd.isna().mean().mean() * 100))


    meta['leakage_flags'] = []
    if verbose and meta["n_class"] == 2 and meta["n"] >= 30:
        for j, nm in enumerate(Xd.columns):
            col = Xd.values[:, j].astype(float)
            ok = ~np.isnan(col)
            if ok.sum() < 20 or len(np.unique(col[ok])) < 2:
                continue
            try:
                a = roc_auc_score(yy[ok], col[ok])
            except Exception:
                continue
            a = max(a, 1 - a)
            if a > 0.90:
                meta['leakage_flags'].append((nm, round(float(a), 3)))

    if verbose:
        print(f"[{key:10}] n={meta['n']:5d} features={meta['n_feat']:3d} "
              f"classes={meta['n_class']} IR={meta['IR']:6.2f} "
              f"NaN={meta['missing_pct']:4.1f}%  target='{tgt}'")
        print(f"{'':12} class distribution: {meta['counts']}  removed={dropped}")
        if meta['leakage_flags']:
            print(f"{'':12} !! POSSIBLE LEAKAGE: single features with AUC > 0.90: "
                  f"{meta['leakage_flags']}")
            print(f"{'':12}    Check whether these are diagnostic tests rather than risk "
                  f"factors. If so, add them to REGISTRY['{key}']['drop'].")
    return dict(X=Xd.values.astype(float), y=yy, features=Xd.columns.tolist(), meta=meta)


# ============================================================
# PREFLIGHT
# ============================================================
# ============================================================


# ============================================================
def preflight(keys=None):
    keys = list(REGISTRY) if keys is None else keys
    rows = []
    for k in keys:
        try:
            r = load_dataset(k, verbose=False); m = r["meta"]
            rows.append(dict(status="OK", **{x: m[x] for x in
                        ["key", "name", "n", "n_feat", "n_class", "IR", "missing_pct"]},
                        target=m["target"], dibuang=len(m["dropped"])))
        except Exception as e:
            rows.append(dict(status="FAILED", key=k, name=REGISTRY[k]["name"],
                             n=np.nan, n_feat=np.nan, n_class=np.nan, IR=np.nan,
                             missing_pct=np.nan, target=str(e)[:90], dibuang=np.nan))
    t = pd.DataFrame(rows)
    display(t)
    ok = (t["status"] == "OK").sum()
    print(f"\n{ok}/{len(t)} dataset dapat dimuat.")
    if ok < len(t):
        print("Fix FAILED entries in REGISTRY (the 'target' column contains the error message).")


    print("\n=== Single-feature leakage screen (AUC > 0.90) ===")
    print("A single feature that nearly determines the target is often a confirmatory test,")
    print("not a risk factor. Check it manually before retaining it.\n")
    from sklearn.metrics import roc_auc_score
    for k in keys:
        try:
            r = load_dataset(k, verbose=False)
        except Exception:
            continue
        X, yy = r["X"], r["y"]
        if r["meta"]["multiclass"]:
            print(f"  [{k:10}] multiclass — leakage screen skipped"); continue
        flags = []
        for j, nm in enumerate(r["features"]):
            col = X[:, j]
            ok_ = ~np.isnan(col)
            if ok_.sum() < 20 or len(np.unique(col[ok_])) < 2:
                continue
            try:
                a = roc_auc_score(yy[ok_], col[ok_])
            except Exception:
                continue
            a = max(a, 1 - a)
            if a > 0.90:
                flags.append((nm, round(a, 3)))
        print(f"  [{k:10}] {flags if flags else 'clean'}")
    return t


# ============================================================
# MULTICLASS SUPPORT
# ============================================================
# ============================================================


# ============================================================
def pos_neg_of(y_fold):
    """Determine positive/minority and negative/majority labels from the current training fold."""

    c_, n_ = np.unique(np.asarray(y_fold), return_counts=True)
    return int(c_[np.argmin(n_)]), int(c_[np.argmax(n_)])


def safe_levels_mc(Psi, y, k):
    y = np.asarray(y); n = len(Psi)
    D = fidelity_distmat(Psi, Psi); np.fill_diagonal(D, np.inf)
    ke = max(1, min(k, n - 1))
    return np.array([float(np.mean(y[np.argpartition(D[i], ke)[:ke]] == y[i]))
                     for i in range(n)])


def qs_enn_mc(Psi, y, k_sl=None, k_enn=None, target_ratio=None,
              use_borderline=True, synth_mode="centroid_min",
              use_qenn=True, lam_max=1.0, rng=None):
    k_sl   = QSENN_K_SAFELEVEL if k_sl is None else k_sl
    k_enn  = QSENN_K_ENN if k_enn is None else k_enn
    target_ratio = QSENN_TARGET if target_ratio is None else target_ratio
    rng = np.random.RandomState(0) if rng is None else rng
    y = np.asarray(y)
    cls, cnt = np.unique(y, return_counts=True)
    maj = int(cls[np.argmax(cnt)]); target_n = int(round(target_ratio * cnt.max()))

    r_all = safe_levels_mc(Psi, y, k_sl)
    Pn, yn = [], []
    for c in cls:
        if c == maj:
            continue
        m = (y == c); Pc = Psi[m]
        need = max(0, target_n - int(m.sum()))
        if need == 0 or len(Pc) == 0:
            continue
        w = (1.0 - r_all[m]) if use_borderline else np.ones(int(m.sum()))
        w = np.ones_like(w) / len(w) if w.sum() <= 1e-12 else w / w.sum()
        mu = centroid_state(Pc)
        for si in rng.choice(len(Pc), size=need, p=w):
            if synth_mode == "neighbor" and len(Pc) > 1:
                j = int(rng.randint(len(Pc)))
                tgt = Pc[j] if j != si else Pc[(j + 1) % len(Pc)]
            else:
                tgt = mu
            Pn.append(slerp_state(Pc[si], tgt, float(rng.uniform(0, lam_max))))
            yn.append(c)

    P_all = np.vstack([Psi] + ([np.array(Pn)] if Pn else []))
    y_all = np.concatenate([y, np.array(yn, dtype=y.dtype)]) if yn else y.copy()
    is_orig = np.zeros(len(y_all), bool); is_orig[:len(y)] = True

    keep = np.ones(len(y_all), bool)
    if use_qenn and len(y_all) > k_enn + 1:
        D = fidelity_distmat(P_all, P_all); np.fill_diagonal(D, np.inf)
        ke = max(1, min(k_enn, len(y_all) - 1))
        for i in range(len(y_all)):
            if is_orig[i] and y_all[i] != maj:
                continue
            lab, ct = np.unique(y_all[np.argpartition(D[i], ke)[:ke]], return_counts=True)
            if lab[np.argmax(ct)] != y_all[i]:
                keep[i] = False

    rm = ~keep
    n_after = {int(c): int((y_all[keep] == c).sum()) for c in cls}


    stats = dict(n_synth=len(yn), n_removed=int(rm.sum()),
                 n_removed_maj_orig=int((rm & is_orig & (y_all == maj)).sum()),
                 n_removed_min_orig=int((rm & is_orig & (y_all != maj)).sum()),
                 n_removed_synth=int((rm & ~is_orig).sum()),
                 n_min_before=int(cnt.min()), n_maj_before=int(cnt.max()),
                 n_min_after=int(min(n_after.values())),
                 n_maj_after=int(max(n_after.values())),
                 ir_before=float(cnt.max() / cnt.min()),
                 ir_after=float(max(n_after.values()) / max(min(n_after.values()), 1)))
    assert stats["n_removed_min_orig"] == 0, "Pillar 3 invariant violated"
    return P_all[keep], y_all[keep], stats


def knn_proba_mc(D, y_tr, K, classes):
    y_tr = np.asarray(y_tr); ke = max(1, min(K, D.shape[1]))
    out = np.zeros((D.shape[0], len(classes)))
    for i in range(D.shape[0]):
        idx = np.argpartition(D[i], ke - 1)[:ke]
        for j, c in enumerate(classes):
            out[i, j] = np.mean(y_tr[idx] == c)
    return out


from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, balanced_accuracy_score)

def score_any(y_true, prob, classes, pos_label=None):
    """Compute the common evaluation metrics for binary or multiclass predictions."""


    classes = np.asarray(classes)
    if len(classes) == 2:
        pos = classes[1] if pos_label is None else pos_label
        neg = classes[0] if pos_label is None else classes[classes != pos_label][0]
        p = prob[:, list(classes).index(pos)] if prob.ndim > 1 else np.asarray(prob)
        yp = np.where(p >= 0.5, pos, neg)
        rp = recall_score(y_true, yp, pos_label=pos, zero_division=0)
        rn = recall_score(y_true, yp, pos_label=neg, zero_division=0)
        try:    auc = roc_auc_score((y_true == pos).astype(int), p)
        except Exception: auc = np.nan
        return dict(Acc=accuracy_score(y_true, yp), Recall_1=rp,
                    Prec_1=precision_score(y_true, yp, pos_label=pos, zero_division=0),
                    F1_1=f1_score(y_true, yp, pos_label=pos, zero_division=0),
                    Gmean=float(np.sqrt(rp * rn)), AUC=auc,
                    BalAcc=balanced_accuracy_score(y_true, yp))
    yp = classes[np.asarray(prob).argmax(1)]
    rec = recall_score(y_true, yp, average=None, labels=classes, zero_division=0)
    try:    auc = roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
    except Exception: auc = np.nan
    return dict(Acc=accuracy_score(y_true, yp),
                Recall_1=recall_score(y_true, yp, average="macro", zero_division=0),
                Prec_1=precision_score(y_true, yp, average="macro", zero_division=0),
                F1_1=f1_score(y_true, yp, average="macro", zero_division=0),
                Gmean=float(np.prod(np.clip(rec, 1e-12, None)) ** (1.0 / len(rec))),
                AUC=auc, BalAcc=balanced_accuracy_score(y_true, yp))


# ============================================================
# run_dataset()
# ============================================================
# ============================================================


# ============================================================
def n_splits_for(y, want=10):
    _, cnt = np.unique(y, return_counts=True)
    return int(max(3, min(want, cnt.min())))


def run_dataset(key, seeds=(11, 23, 37), want_folds=10, do_descriptors=True):
    R = load_dataset(key, verbose=False); Xr, yy, meta = R["X"], R["y"], R["meta"]
    MC = meta["multiclass"]; classes = np.unique(yy)
    ns = n_splits_for(yy, want_folds)
    print(f"\n=== {meta['name']} === n={meta['n']} features={meta['n_feat']} "
          f"classes={meta['n_class']} IR={meta['IR']:.2f} | {len(seeds)}x{ns}-fold"
          f"{' | MULTICLASS (macro metrics)' if MC else ''}")

    rows, drows, evr_all = [], [], []
    t0 = time.time()
    for rep, seed in enumerate(seeds, 1):
        skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xr, yy), 1):
            Xtr, Xte, ytr, yte, evr = make_fold_data(Xr, yy, tr, te)
            evr_all.append(float(np.sum(evr)))
            pos_f, neg_f = pos_neg_of(ytr)
            rs = 5000 * rep + fold
            Ptr, Pte = quantum_statevectors(Xtr), quantum_statevectors(Xte)

            def enc(Xy):
                return quantum_statevectors(Xy[0]), Xy[1]

            def _safe(sm):
                try:    return sm.fit_resample(Xtr, ytr)
                except Exception: return Xtr, ytr

            arms = {}
            arms["QKNN + No Resampling"] = (Ptr, ytr)
            arms["QKNN + SMOTE"]            = enc(_safe(SMOTE(random_state=rs)))
            arms["QKNN + SMOTE-ENN"]        = enc(_safe(SMOTEENN(random_state=rs)))
            arms["QKNN + Borderline-SMOTE"] = enc(_safe(BorderlineSMOTE(random_state=rs)))
            arms["QKNN + SMOTE-Tomek"]      = enc(_safe(SMOTETomek(random_state=rs)))

            if MC:


                arms["QKNN + Quantum-SMOTE"] = qs_enn_mc(
                    Ptr, ytr, use_borderline=False, use_qenn=False,
                    rng=np.random.RandomState(rs))[:2]
                arms["QKNN + Quantum-SMOTEV2"] = qs_enn_mc(
                    Ptr, ytr, use_borderline=False, use_qenn=False,
                    synth_mode="neighbor", rng=np.random.RandomState(rs))[:2]
                full = qs_enn_mc(Ptr, ytr, rng=np.random.RandomState(rs))
                arms["QKNN + QS-ENN"] = full[:2]; st_r = full[2]
                arms["ablation: without borderline weighting"] = qs_enn_mc(
                    Ptr, ytr, use_borderline=False, rng=np.random.RandomState(rs))[:2]
                arms["ablation: P2 = neighbor target"] = qs_enn_mc(
                    Ptr, ytr, synth_mode="neighbor", rng=np.random.RandomState(rs))[:2]
                arms["ablation: without Q-ENN"] = qs_enn_mc(
                    Ptr, ytr, use_qenn=False, rng=np.random.RandomState(rs))[:2]
            else:
                arms["QKNN + Quantum-SMOTE"]   = quantum_smote(Ptr, ytr, Xtr, rng=np.random.RandomState(rs))[:2]
                arms["QKNN + Quantum-SMOTEV2"] = quantum_smotev2(Ptr, ytr, rng=np.random.RandomState(rs))[:2]
                full = qs_enn_variant(Ptr, ytr, rng=np.random.RandomState(rs))
                arms["QKNN + QS-ENN"] = full[:2]; st_r = full[2]
                arms["ablation: without borderline weighting"] = qs_enn_variant(Ptr, ytr, use_borderline=False, rng=np.random.RandomState(rs))[:2]
                arms["ablation: P2 = neighbor target"]    = qs_enn_variant(Ptr, ytr, synth_mode="neighbor", rng=np.random.RandomState(rs))[:2]
                arms["ablation: without Q-ENN"]      = qs_enn_variant(Ptr, ytr, use_qenn=False, rng=np.random.RandomState(rs))[:2]

            for nm, (Pres, yres) in arms.items():
                prob = knn_proba_mc(swap_fidelity_distmat(Pte, Pres), yres,
                                    K_NEIGHBORS, classes)
                row = dict(dataset=meta["name"], key=key, rep=rep, fold=fold,
                           seed=seed, arm=nm, multiclass=MC,
                           **score_any(yte, prob, classes, pos_label=pos_f))
                if nm == "QKNN + QS-ENN":
                    row.update(st_r); row["cum_var"] = float(np.sum(evr))
                rows.append(row)


            Xb, yb = _safe(SMOTEENN(random_state=rs))
            knn = KNeighborsClassifier(K_NEIGHBORS).fit(Xb, yb)
            pr = knn.predict_proba(Xte)
            if pr.shape[1] != len(classes):
                full_pr = np.zeros((len(Xte), len(classes)))
                for j, c in enumerate(knn.classes_):
                    full_pr[:, list(classes).index(c)] = pr[:, j]
                pr = full_pr
            rows.append(dict(dataset=meta["name"], key=key, rep=rep, fold=fold,
                             seed=seed, arm="KNN + SMOTE-ENN", multiclass=MC,
                             **score_any(yte, pr, classes, pos_label=pos_f)))

            if do_descriptors and not MC:
                try:
                    d = fold_descriptors(Ptr, ytr, rng=np.random.RandomState(rs))
                    d.update(dataset=meta["name"], key=key, rep=rep, fold=fold)
                    drows.append(d)
                except Exception:
                    pass
        print(f"   repeat {rep}/{len(seeds)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(outp(f"folds_{key}.csv"), index=False, float_format="%.15g")
    if drows:
        pd.DataFrame(drows).to_csv(outp(f"descriptors_{key}.csv"), index=False, float_format="%.15g")


    ab = []
    for metric in ["Recall_1", "F1_1", "Gmean"]:
        pv = df.pivot_table(index=["rep", "fold"], columns="arm", values=metric)
        raw, tmp = [], []
        for lab, arm in [("Without Pillar 1 (borderline weighting)", "ablation: without borderline weighting"),
                         ("Pillar 2 -> neighbor target",           "ablation: P2 = neighbor target"),
                         ("Without Pillar 3 (protective editing)",   "ablation: without Q-ENN")]:
            if arm not in pv.columns:
                continue
            r_ = wilcoxon_full(pv[arm].values, pv["QKNN + QS-ENN"].values)
            tmp.append(dict(dataset=meta["name"], key=key, metric=metric, variant=lab,
                            delta=r_["delta"], N=r_["N"], z=r_["z"], r=r_["r"]))
            raw.append(r_["p"] if np.isfinite(r_["p"]) else 1.0)
        for row, pa in zip(tmp, holm(np.array(raw))):
            row["p_holm"] = float(pa); row["significant"] = bool(pa < 0.05); ab.append(row)
    pd.DataFrame(ab).to_csv(outp(f"ablation_{key}.csv"), index=False, float_format="%.15g")

    print(f"   retained variance: {np.mean(evr_all)*100:.1f}% ± {np.std(evr_all, ddof=1)*100:.1f}%")
    top = df.groupby("arm")["F1_1"].mean().sort_values(ascending=False)
    print(f"   F1 QS-ENN = {top.get('QKNN + QS-ENN', np.nan):.4f} | best: "
          f"{top.index[0]} ({top.iloc[0]:.4f})")
    return df


# ============================================================

# ============================================================
# ============================================================

# ============================================================
import glob

def load_all_folds(pattern=None):
    pattern = outp("folds_*.csv") if pattern is None else pattern
    fs = [f for f in sorted(glob.glob(pattern))
          if not os.path.basename(f).startswith("folds_ALL")]
    if not fs:
        raise FileNotFoundError(f"No folds_*.csv files were found in {OUT_DIR}. "
                                f"Run the dataset experiments first.")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def make_tables(df, baselines=("QKNN + SMOTE-ENN", "QKNN + Quantum-SMOTE",
                               "QKNN + Quantum-SMOTEV2")):
    prop = "QKNN + QS-ENN"

    t2 = (df[df["arm"] == prop]
          .groupby("dataset")[["Recall_1", "Prec_1", "F1_1", "Gmean", "AUC", "Acc"]].mean())


    t3 = df.pivot_table(index="dataset", columns="arm", values="F1_1", aggfunc="mean")


    rows = []
    for ds, g in df.groupby("dataset"):
        prop_s = g[g["arm"] == prop].sort_values(["rep", "fold"])["F1_1"].to_numpy(float)
        rec = dict(dataset=ds, QS_ENN_F1=float(prop_s.mean()))
        raw_p = []
        for b in baselines:
            base_s = g[g["arm"] == b].sort_values(["rep", "fold"])["F1_1"].to_numpy(float)
            if len(base_s) != len(prop_s) or len(prop_s) == 0:
                rec[f"d_{b}"] = np.nan; raw_p.append(1.0); continue
            rec[f"d_{b}"] = float(prop_s.mean() - base_s.mean())
            raw_p.append(wilcoxon_full(prop_s, base_s)["p"])
        padj = holm(np.array(raw_p))
        for b, pa in zip(baselines, padj):
            rec[f"p_{b}"] = float(pa)
        rows.append(rec)
    t5 = pd.DataFrame(rows).set_index("dataset")


    checks = []
    for ds in t5.index:
        for b in baselines:
            if b in t3.columns and not np.isnan(t5.loc[ds, f"d_{b}"]):
                from_t3 = t3.loc[ds, prop] - t3.loc[ds, b]
                checks.append(dict(dataset=ds, baseline=b,
                                   delta_t5=t5.loc[ds, f"d_{b}"], delta_from_t3=from_t3,
                                   difference=abs(t5.loc[ds, f"d_{b}"] - from_t3)))
    chk = pd.DataFrame(checks)
    return t2, t3, t5, chk


# ============================================================
# LEAVE-ONE-DATASET-OUT TEST
# ============================================================
# ============================================================



# ============================================================
import glob, os

def build_applicability_table():
    d_files = sorted(glob.glob(outp("descriptors_*.csv")))
    a_files = sorted(glob.glob(outp("ablation_*.csv")))
    if len(d_files) < 3:
        print(f"Only {len(d_files)} datasets are available. Run the remaining datasets first "
              f"(10 are recommended for a meaningful LODO analysis).")
        return None
    D = pd.concat([pd.read_csv(f) for f in d_files], ignore_index=True)
    A = pd.concat([pd.read_csv(f) for f in a_files], ignore_index=True)
    Dm = D.groupby("dataset").mean(numeric_only=True).drop(columns=["rep", "fold"], errors="ignore")
    rows = []
    for ds, g in A[A['metric'] == "F1_1"].groupby("dataset") if "dataset" in A.columns else []:
        pass
    piv = A[A['metric'] == "F1_1"].pivot_table(index="dataset", columns='variant', values="delta")
    return Dm.join(piv, how="inner")


def lodo_sign_test(tab, descriptor, target_col):
    """Fit a one-variable threshold on n-1 datasets and predict the sign of the held-out delta."""
    from itertools import product
    ok, n = 0, 0
    for held in tab.index:
        tr = tab.drop(index=held)
        xs = tr[descriptor].to_numpy(float); ys = np.sign(tr[target_col].to_numpy(float))
        cands = np.unique(xs)
        best_t, best_acc, best_dir = None, -1, 1
        for t in cands:
            for dirn in (1, -1):
                pred = np.where(xs > t, dirn, -dirn)
                acc = np.mean(pred == ys)
                if acc > best_acc:
                    best_t, best_acc, best_dir = t, acc, dirn
        xh = float(tab.loc[held, descriptor]); yh = np.sign(float(tab.loc[held, target_col]))
        pred_h = best_dir if xh > best_t else -best_dir
        ok += int(pred_h == yh); n += 1
    return ok, n


# ============================================================
# REPRESENTATION SENSITIVITY (K x encoding)
# ============================================================
# ============================================================

# ============================================================
def make_encoder(depth=2, ent="ring", n_q=None):
    """Create a vectorized feature-map encoder with configurable depth and entanglement while keeping the qubit budget fixed."""


    n_q = n_qubits if n_q is None else n_q
    prm = {"s": np.ones(n_q),
           "beta":  np.linspace(0.5, 1.0, depth),
           "gamma": np.linspace(0.3, 0.7, depth)}
    def encode(X):
        return fast_statevectors(X, depth=depth, ent=ent, params=prm, n=n_q)
    return encode


# ============================================================

# ============================================================
# ============================================================

# ============================================================
from sklearn.model_selection import StratifiedKFold as _SKF
from sklearn.neighbors import KNeighborsClassifier

INNER_SPLITS = BUDGET["inner_splits"]
GRID_SIZE    = 9

def _grid(a_name, a_vals, b_name, b_vals):
    return [{a_name: a, b_name: b} for a in a_vals for b in b_vals]

PARAM_GRIDS = {
    "SMOTE":             _grid("k_neighbors", [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "Borderline-SMOTE":  _grid("k_neighbors", [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "SMOTE-ENN":         _grid("smote_k",     [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "SMOTE-Tomek":       _grid("smote_k",     [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "Quantum-SMOTE":     _grid("n_clusters",  [2, 3, 5], "angle_frac",        [0.10, 0.15, 0.25]),
    "Quantum-SMOTEV2":   _grid("split_factor",[5, 10, 20], "num_bins",        [3, 5, 8]),
    "QS-ENN":            _grid("k_sl",        [3, 5, 7], "target_ratio",      [0.5, 0.75, 1.0]),
}


def build_arm(method, Xtr, ytr, Ptr, params, rng):
    """Return resampled statevectors and labels for one method/configuration using a common resampler interface."""


    p = dict(params)
    ss = p.pop("sampling_strategy", 1.0)
    if method == "SMOTE":
        Xr, yr = SMOTE(k_neighbors=p["k_neighbors"], sampling_strategy=ss,
                       random_state=rng).fit_resample(Xtr, ytr)
        return quantum_statevectors(Xr), yr
    if method == "Borderline-SMOTE":
        Xr, yr = BorderlineSMOTE(k_neighbors=p["k_neighbors"], sampling_strategy=ss,
                                 random_state=rng).fit_resample(Xtr, ytr)
        return quantum_statevectors(Xr), yr
    if method == "SMOTE-ENN":
        Xr, yr = SMOTEENN(smote=SMOTE(k_neighbors=p["smote_k"], sampling_strategy=ss,
                                      random_state=rng),
                          random_state=rng).fit_resample(Xtr, ytr)
        return quantum_statevectors(Xr), yr
    if method == "SMOTE-Tomek":
        Xr, yr = SMOTETomek(smote=SMOTE(k_neighbors=p["smote_k"], sampling_strategy=ss,
                                        random_state=rng),
                            random_state=rng).fit_resample(Xtr, ytr)
        return quantum_statevectors(Xr), yr
    if method == "Quantum-SMOTE":
        Pq, yq, _ = quantum_smote(Ptr, ytr, Xtr, n_clusters=p["n_clusters"],
                                  angle_frac=p["angle_frac"], target_ratio=ss,
                                  rng=np.random.RandomState(rng))
        return Pq, yq
    if method == "Quantum-SMOTEV2":
        Pv, yv, _ = quantum_smotev2(Ptr, ytr, split_factor=p["split_factor"],
                                    num_bins=p["num_bins"], target_ratio=ss,
                                    rng=np.random.RandomState(rng))
        return Pv, yv
    if method == "QS-ENN":
        Pr, yr, _ = qs_enn_variant(Ptr, ytr, k_sl=p["k_sl"], target_ratio=p["target_ratio"],
                                   rng=np.random.RandomState(rng))
        return Pr, yr
    raise ValueError(method)


def _f1_on(Pte, yte, Pres, yres, pos_f, neg_f):


    p = knn_proba1_from_distmat(swap_fidelity_distmat(Pte, Pres), yres,
                                K_NEIGHBORS, pos_label=pos_f)
    yp = np.where(np.asarray(p) >= 0.5, pos_f, neg_f)
    return f1_score(yte, yp, pos_label=pos_f, zero_division=0), p


def select_params(method, Xtr, ytr, Ptr, seed, inner_splits=None):
    """Select hyperparameters using inner cross-validation on the outer training fold only."""
    inner_splits = BUDGET["inner_splits"] if inner_splits is None else inner_splits
    inner = _SKF(n_splits=inner_splits, shuffle=True, random_state=seed)
    best, best_s = None, -np.inf
    for cfg in PARAM_GRIDS[method]:
        scores = []
        for j, (itr, ite) in enumerate(inner.split(Xtr, ytr)):
            Xi, Xj = Xtr[itr], Xtr[ite]
            yi, yj = ytr[itr], ytr[ite]
            Pi, Pj = Ptr[itr], Ptr[ite]
            pos_i, neg_i = pos_neg_of(yi)
            try:
                Pres, yres = build_arm(method, Xi, yi, Pi, cfg, seed * 100 + j)
                s, _ = _f1_on(Pj, yj, Pres, yres, pos_i, neg_i)
            except Exception:
                s = 0.0
            scores.append(s)
        m = float(np.mean(scores))
        if m > best_s:
            best, best_s = cfg, m
    return best, best_s

# ============================================================



# ============================================================
GRID_SIZE = 9

def _grid(a, av, b, bv):
    return [{a: x, b: y} for x in av for y in bv]

PARAM_GRIDS = {
    "SMOTE":            _grid("k_neighbors", [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "Borderline-SMOTE": _grid("k_neighbors", [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "SMOTE-ENN":        _grid("smote_k",     [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "SMOTE-Tomek":      _grid("smote_k",     [3, 5, 7], "sampling_strategy", [0.5, 0.75, 1.0]),
    "Quantum-SMOTE":    _grid("n_clusters",  [2, 3, 5], "angle_frac",        [0.10, 0.15, 0.25]),
    "Quantum-SMOTEV2":  _grid("split_factor", [5, 10, 20], "num_bins",       [3, 5, 8]),
    "QS-ENN":           _grid("k_sl",        [3, 5, 7], "target_ratio",      [0.5, 0.75, 1.0]),
}
for _m, _g in PARAM_GRIDS.items():
    assert len(_g) == GRID_SIZE, f"grid {_m} = {len(_g)}, expected {GRID_SIZE}"


def run_nested_cv(key, seeds=None, folds=None, inner=None):
    """Run equal-budget nested cross-validation for the supported binary datasets."""
    seeds = BUDGET["nested_seeds"] if seeds is None else seeds
    folds = BUDGET["nested_folds"] if folds is None else folds
    inner = BUDGET["inner_splits"] if inner is None else inner
    R = load_dataset(key, verbose=False); Xr, yy, meta = R["X"], R["y"], R["meta"]
    if meta["multiclass"]:
        raise NotImplementedError("Nested CV is currently implemented for binary datasets only.")
    rows, chosen, t0 = [], [], time.time()
    for rep, seed in enumerate(seeds, 1):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xr, yy), 1):
            Xtr, Xte, ytr, yte, _ = make_fold_data(Xr, yy, tr, te)
            pos_f, neg_f = pos_neg_of(ytr)
            Ptr, Pte = quantum_statevectors(Xtr), quantum_statevectors(Xte)
            rs = 5000 * rep + fold
            f1b, _ = _f1_on(Pte, yte, Ptr, ytr, pos_f, neg_f)
            rows.append(dict(dataset=meta["name"], key=key, rep=rep, fold=fold,
                             seed=seed, arm="QKNN + No Resampling", F1_1=f1b))
            for method in PARAM_GRIDS:
                cfg, inner_s = select_params(method, Xtr, ytr, Ptr, rs, inner)
                Pres, yres = build_arm(method, Xtr, ytr, Ptr, cfg, rs)
                f1o, _ = _f1_on(Pte, yte, Pres, yres, pos_f, neg_f)
                rows.append(dict(dataset=meta["name"], key=key, rep=rep, fold=fold,
                                 seed=seed, arm=f"QKNN + {method}", F1_1=f1o))
                chosen.append(dict(dataset=meta["name"], key=key, rep=rep, fold=fold,
                                   arm=method, inner_F1=inner_s, **cfg))
        print(f"   nested repeat {rep}/{len(seeds)} ({time.time()-t0:.0f}s)")
    nd, cd = pd.DataFrame(rows), pd.DataFrame(chosen)
    nd.to_csv(outp(f"nested_{key}.csv"), index=False, float_format="%.15g")
    cd.to_csv(outp(f"nested_chosen_{key}.csv"), index=False, float_format="%.15g")
    for m in PARAM_GRIDS:
        sub = cd[cd["arm"] == m].drop(columns=["dataset", "key", "rep", "fold",
                                               "arm", "inner_F1"], errors="ignore")
        sub = sub.dropna(axis=1, how="all")
        if not len(sub) or not len(sub.columns):
            continue
        modus = {c: (sub[c].mode().iloc[0] if len(sub[c].mode()) else "—")
                 for c in sub.columns}
        stab = {c: f"{(sub[c] == modus[c]).mean():.0%}" for c in sub.columns}
        print(f"  {m:18} modus={modus} | stabilitas={stab}")
    return nd, cd


ENCODINGS = {
    "D2-ring (default)": dict(depth=2, ent="ring"),
    "D1-ring":        dict(depth=1, ent="ring"),
    "D3-ring":        dict(depth=3, ent="ring"),
    "D2-linear":      dict(depth=2, ent="linear"),
    "D2-full":        dict(depth=2, ent="full"),
}
K_VALUES = [1, 3, 5, 7, 9]


def run_representation(key, seeds=None, folds=None):
    """Evaluate sensitivity to neighborhood size and quantum feature-map configuration."""
    seeds = BUDGET["rep_seeds"] if seeds is None else seeds
    folds = BUDGET["rep_folds"] if folds is None else folds
    R = load_dataset(key, verbose=False); Xr, yy, meta = R["X"], R["y"], R["meta"]
    rows, t0 = [], time.time()
    for enc_name, kw in ENCODINGS.items():
        encode = make_encoder(**kw)
        for rep, seed in enumerate(seeds, 1):
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(skf.split(Xr, yy), 1):
                Xtr, Xte, ytr, yte, _ = make_fold_data(Xr, yy, tr, te)
                pos_f, neg_f = pos_neg_of(ytr)
                Ptr, Pte = encode(Xtr), encode(Xte)
                rs = 5000 * rep + fold
                mk = qs_enn_mc if meta["multiclass"] else qs_enn_variant
                res = {"No Resampling": (Ptr, ytr),
                       "Quantum-SMOTE": (qs_enn_mc(Ptr, ytr, use_borderline=False,
                                                   use_qenn=False,
                                                   rng=np.random.RandomState(rs))[:2]
                                         if meta["multiclass"] else
                                         quantum_smote(Ptr, ytr, Xtr,
                                                       rng=np.random.RandomState(rs))[:2]),
                       "QS-ENN": mk(Ptr, ytr, rng=np.random.RandomState(rs))[:2]}
                for arm, (Pres, yres) in res.items():
                    D = swap_fidelity_distmat(Pte, Pres)
                    for K in K_VALUES:
                        prob = knn_proba_mc(D, yres, K, np.unique(yy))
                        rows.append(dict(dataset=meta["name"], key=key,
                                         encoding=enc_name, K=K, arm=arm,
                                         rep=rep, fold=fold,
                                         **score_any(yte, prob, np.unique(yy),
                                                     pos_label=pos_f)))
        print(f"   encoding {enc_name} ({time.time()-t0:.0f}s)")
    rd = pd.DataFrame(rows)
    rd.to_csv(outp(f"representation_{key}.csv"), index=False, float_format="%.15g")
    piv = rd.pivot_table(index=["encoding", "K"], columns="arm", values="F1_1")
    piv["delta_QSENN_vs_QSMOTE"] = piv["QS-ENN"] - piv["Quantum-SMOTE"]
    sgn = np.sign(piv["delta_QSENN_vs_QSMOTE"].to_numpy())
    sgn = sgn[~np.isnan(sgn)]
    print(f"   tanda delta di {len(sgn)} kombinasi: positif {int((sgn>0).sum())} "
          f"| negatif {int((sgn<0).sum())} | nol {int((sgn==0).sum())}")
    return rd, piv


def applicability_lodo():
    """Evaluate simple geometry-based applicability rules with leave-one-dataset-out sign prediction."""


    tab = build_applicability_table()
    if tab is None:
        return None
    PAIRS = [("fraction_toward_majority_centroid", "Pillar 2 -> neighbor target"),
             ("centroid_concentration",   "Pillar 2 -> neighbor target"),
             ("minority_purity",       "Without Pillar 3 (protective editing)"),
             ("editing_asymmetry",   "Without Pillar 3 (protective editing)"),
             ("class_overlap",          "Without Pillar 3 (protective editing)"),
             ("synthetic_contamination",   "Without Pillar 1 (borderline weighting)")]
    out = []
    for dsc, tgt in PAIRS:
        if dsc not in tab.columns or tgt not in tab.columns:
            continue
        sub = tab[[dsc, tgt]].dropna()
        if len(sub) < 4:
            continue
        rho, p = spearmanr(sub[dsc], sub[tgt])
        ok, n = lodo_sign_test(sub, dsc, tgt)
        out.append(dict(descriptor=dsc, target=tgt, n_datasets=len(sub),
                        spearman_rho=rho, p=p, LODO_correct=f"{ok}/{n}",
                        LODO_accuracy=ok / n))
    if not out:
        n_bin = len(tab)
        print(f"Hanya {n_bin} datasets have complete descriptors and ablation results. "
              f"Uji LODO memerlukan minimal 4 (idealnya 9 dataset biner). "
              f"Run the remaining datasets first.")
        return None
    res = pd.DataFrame(out).sort_values("LODO_accuracy", ascending=False)
    res.to_csv(outp("applicability_lodo.csv"), index=False, float_format="%.15g")
    return res


SHOT_LEVELS = [128, 512, 2048, 8192, None]


def run_finite_shot(key, seeds=None, folds=None, n_real=None, levels=None):
    """Evaluate the binary pipeline with finite-shot fidelity estimates."""
    seeds  = BUDGET["shot_seeds"] if seeds is None else seeds
    folds  = BUDGET["shot_folds"] if folds is None else folds
    n_real = BUDGET["shot_realisasi"] if n_real is None else n_real
    levels = SHOT_LEVELS if levels is None else levels
    R = load_dataset(key, verbose=False); Xr, yy, meta = R["X"], R["y"], R["meta"]
    if meta["multiclass"]:
        raise NotImplementedError("Finite-shot evaluation is currently implemented for binary datasets only.")
    rows, t0 = [], time.time()
    for rep, seed in enumerate(seeds, 1):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xr, yy), 1):
            Xtr, Xte, ytr, yte, _ = make_fold_data(Xr, yy, tr, te)
            pos_f, neg_f = pos_neg_of(ytr)
            Ptr, Pte = quantum_statevectors(Xtr), quantum_statevectors(Xte)
            rs = 5000 * rep + fold
            D_exact = _FID_EXACT(Pte, Ptr)
            r_exact = quantum_safe_levels(Ptr, ytr, pos_f, QSENN_K_SAFELEVEL)
            _, y_ex, _ = qs_enn_variant(Ptr, ytr, rng=np.random.RandomState(rs))
            n_keep_exact = len(y_ex)
            for shots in levels:
                for real in range(1 if shots is None else n_real):
                    rng_s = np.random.RandomState(90000 + 7 * real + fold)
                    with finite_shot_mode(shots, rng_s):
                        Pres, yres, _ = qs_enn_variant(Ptr, ytr,
                                                       rng=np.random.RandomState(rs))
                        p = knn_proba1_from_distmat(
                            swap_fidelity_distmat(Pte, Pres), yres, K_NEIGHBORS,
                            pos_label=pos_f)
                        r_noisy = quantum_safe_levels(Ptr, ytr, pos_f, QSENN_K_SAFELEVEL)
                        D_noisy = fidelity_distmat(Pte, Ptr)
                    yp = np.where(np.asarray(p) >= 0.5, pos_f, neg_f)
                    taus = [kendalltau(D_exact[i], D_noisy[i]).correlation
                            for i in range(0, len(Pte), max(1, len(Pte) // 20))]
                    rows.append(dict(
                        dataset=meta["name"], key=key, rep=rep, fold=fold,
                        realisasi=real, shots=(shots if shots is not None else -1),
                        F1_1=f1_score(yte, yp, pos_label=pos_f, zero_division=0),
                        Recall_1=recall_score(yte, yp, pos_label=pos_f, zero_division=0),
                        kendall_tau=float(np.nanmean(taus)),
                        shift_safelevel=float(np.mean(np.abs(r_noisy - r_exact))),
                        delta_keputusan_enn=abs(len(yres) - n_keep_exact) / max(n_keep_exact, 1)))
        print(f"   finite-shot repeat {rep}/{len(seeds)} ({time.time()-t0:.0f}s)")
    sd = pd.DataFrame(rows)
    sd.to_csv(outp(f"finiteshot_{key}.csv"), index=False, float_format="%.15g")
    return sd


# ============================================================
# FAST ENCODER ACTIVATION
# ============================================================
def _activate_fast_encoder(verbose=True):
    """Use the vectorized NumPy encoder after validating it against the PennyLane qnode when PennyLane is available."""


    global quantum_statevectors
    if qml is None:
        quantum_statevectors = fast_statevectors
        if verbose:
            print("PennyLane is not available -> using the NumPy encoder directly.")
        return True
    rng = np.random.RandomState(0)
    Xc = rng.uniform(-0.15, 1.15, (25, n_qubits))
    err = float(np.abs(np.array([np.asarray(_statevector(x)) for x in Xc])
                       - fast_statevectors(Xc)).max())
    if err < 1e-9:
        quantum_statevectors = fast_statevectors
        if verbose:
            print(f"Encoder vektorisasi AKTIF (selisih vs qnode {err:.2e}).")
        return True
    print(f"WARNING: difference {err:.2e} is too large; using the PennyLane qnode.")
    return False


_activate_fast_encoder(verbose=False)


# ============================================================
# BATCH
# ============================================================
def run_all(keys=None, force=False, **kw):
    """Run multiple datasets, skipping completed fold-level result files unless force=True."""
    keys = list(REGISTRY) if keys is None else list(keys)
    results, failures, skipped = {}, {}, []
    t0 = time.time()
    for k in keys:
        existing = outp(f"folds_{k}.csv")
        if os.path.exists(existing) and not force:
            results[k] = pd.read_csv(existing)
            skipped.append(k)
            print(f"[{k:10}] skipped ({len(results[k])} rows) — use force=True to rerun")
            continue
        try:
            t1 = time.time()
            results[k] = run_dataset(k, **kw)
            print(f"   completed in {time.time()-t1:.0f}s -> {existing}")
        except Exception as e:
            failures[k] = f"{type(e).__name__}: {e}"
            print(f"   !! {k} FAILED -> {failures[k]}")
    print(f"\\n{'='*60}\\n{time.time()-t0:.0f}s | completed {len(results)} "
          f"| skipped {len(skipped)} | failed {len(failures)}")
    for k, e in failures.items():
        print(f"   {k}: {e}")
    return results, failures
def tabel_satu(key):
    """Return a compact summary table for one dataset."""
    df = pd.read_csv(outp(f"folds_{key}.csv"))
    return (df.groupby("arm")[["Recall_1", "Prec_1", "F1_1", "Gmean", "AUC", "Acc"]]
              .mean().round(4).sort_values("F1_1", ascending=False))


def tabel_satu_lengkap():
    """Regenerate the dataset-summary table directly from the configured raw inputs and fold-level outputs."""
    rows = []
    for k in REGISTRY:
        try:
            m = load_dataset(k, verbose=False)["meta"]
        except Exception:
            continue
        f = outp(f"folds_{k}.csv")
        cv = pd.read_csv(f)["cum_var"].dropna() if os.path.exists(f) else pd.Series(dtype=float)
        rows.append({"Dataset": m["name"], "n": m["n"], "Features": m["n_feat"],
                     "Classes": m["n_class"], "IR": round(m["IR"], 2),
                     "Var (%)": (f"{cv.mean()*100:.1f} ± {cv.std(ddof=1)*100:.1f}"
                                 if len(cv) > 1 else "—")})
    return pd.DataFrame(rows)


# English public aliases
def table_one(key):
    """Return the mean performance summary for one dataset."""
    return tabel_satu(key)

def table_one_full():
    """Return the cross-dataset dataset/preprocessing summary."""
    return tabel_satu_lengkap()

__all__ = ["init", "outp", "REGISTRY", "load_dataset", "preflight",
           "make_fold_data", "quantum_statevectors", "fast_statevectors",
           "qs_enn_variant", "qs_enn_mc", "quantum_smote", "quantum_smotev2",
           "run_dataset", "run_all", "tabel_satu", "tabel_satu_lengkap", "table_one", "table_one_full",
           "make_tables", "load_all_folds", "wilcoxon_full", "holm", "stat_table",
           "fold_descriptors", "build_applicability_table", "lodo_sign_test",
           "applicability_lodo", "make_encoder", "finite_shot_mode",
           "fidelity_finite_shot", "run_nested_cv", "run_representation",
           "run_finite_shot", "VAL_SEEDS", "DESIGN_SEEDS", "BUDGET"]



# ============================================================
# REGISTRY CHECK
# ============================================================
def cek_registri(keys=None):
    """Validate all dataset-registry entries, targets, files, and potential single-feature leakage indicators."""


    keys = list(REGISTRY) if keys is None else keys
    berkas = list_csv()
    print(f"CSV files in {DATA_DIR!r} ({len(berkas)}):")
    for f in berkas:
        print(f"   {f}")
    print()

    baris = []
    for k in keys:
        cfg = REGISTRY[k]
        try:
            path = _pick_file(cfg["file"], key=k, verbose=False)
            if path is None:
                baris.append(dict(key=k, berkas="MISSING", target="—",
                                  n="—", classes="—", catatan="check filename"))
                continue
            R = load_dataset(k, verbose=False)
            m = R["meta"]
            bocor = ""
            if m["n_class"] == 2:
                for j, nm in enumerate(R["features"]):
                    col = R["X"][:, j]; ok = ~np.isnan(col)
                    if ok.sum() < 20 or len(np.unique(col[ok])) < 2:
                        continue
                    try:
                        a = max(roc_auc_score(R["y"][ok], col[ok]),
                                1 - roc_auc_score(R["y"][ok], col[ok]))
                    except Exception:
                        continue
                    if a > 0.90:
                        bocor += f"{nm}({a:.2f}) "
            baris.append(dict(key=k, berkas=os.path.basename(path),
                              target=m["target"], n=m["n"], classes=m["n_class"],
                              catatan=(f"POSSIBLE LEAKAGE: {bocor}" if bocor else "OK")))
        except Exception as e:
            baris.append(dict(key=k, berkas="—", target="—", n="—", classes="—",
                              catatan=f"{type(e).__name__}: {str(e)[:70]}"))
    t = pd.DataFrame(baris)
    display(t)
    ok = (t["notes"] == "OK").sum()
    print(f"\n{ok}/{len(t)} clean. Rows with notes other than OK should be checked.")
    print("Rows marked POSSIBLE LEAKAGE contain features that almost")
    print("perfectly predict the target on their own -- often diagnostic tests rather than")
    print("risk factors. Add them to REGISTRY[key]['drop'] when appropriate.")
    return t


__all__ = __all__ + ["cek_registri", "list_csv"]
