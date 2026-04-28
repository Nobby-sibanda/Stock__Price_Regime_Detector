"""
Regime detection: HMM, KMeans (auto-k), Ensemble, Walk-forward.
All detectors output integer labels in {0=TRENDING, 1=MEAN-REV, 2=VOLATILE}.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM

from .config import REGIME_NAMES
from .features import VOL_IDX, SLOPE_IDX

_DEFAULT_K = 3


# ── Semantic label helpers ────────────────────────────────────────────────────

def _compute_mapping(raw: np.ndarray, feat_vals: np.ndarray, n: int) -> dict:
    """
    Build raw_cluster → semantic_label mapping from cluster centroids.
    Rule: highest-vol cluster → VOLATILE (2);
          highest |slope| among the rest → TRENDING (0);
          everything else → MEAN-REV (1).
    Works for any n (≥ 2).
    """
    centroids = np.array([
        feat_vals[raw == i].mean(axis=0) if (raw == i).any()
        else np.zeros(feat_vals.shape[1])
        for i in range(n)
    ])
    vol_order = np.argsort(centroids[:, VOL_IDX])[::-1]
    mapping   = {int(vol_order[0]): 2}
    remaining = [i for i in range(n) if i not in mapping]
    slp_win   = max(remaining, key=lambda i: abs(centroids[i, SLOPE_IDX]))
    mapping[slp_win] = 0
    for i in range(n):
        if i not in mapping:
            mapping[i] = 1
    return mapping


def _apply_mapping(raw: np.ndarray, mapping: dict) -> np.ndarray:
    return np.array([mapping.get(int(l), 1) for l in raw])


# ── k / component selection ───────────────────────────────────────────────────

def auto_select_k(X: np.ndarray, k_range=range(2, 7)) -> int:
    """Pick k ∈ k_range that maximises KMeans silhouette score."""
    best_k, best_score = _DEFAULT_K, -1.0
    for k in k_range:
        km  = KMeans(n_clusters=k, n_init=20, random_state=42)
        lbl = km.fit_predict(X)
        if len(np.unique(lbl)) < 2:
            continue
        score = silhouette_score(X, lbl)
        if score > best_score:
            best_score, best_k = score, k
    print(f"    Auto-selected k={best_k}  (silhouette={best_score:.4f})")
    return best_k


# ── Core fitters ─────────────────────────────────────────────────────────────

def _fit_hmm(X: np.ndarray, n: int) -> GaussianHMM:
    m = GaussianHMM(n_components=n, covariance_type="full",
                    n_iter=300, random_state=42)
    m.fit(X)
    return m


def _fit_kmeans(X: np.ndarray, n: int) -> KMeans:
    return KMeans(n_clusters=n, n_init=30, random_state=42).fit(X)


# ── Public detectors ──────────────────────────────────────────────────────────

def detect_hmm(features: pd.DataFrame, n_regimes: int | None = None) -> np.ndarray:
    X   = StandardScaler().fit_transform(features)
    n   = n_regimes or _DEFAULT_K
    m   = _fit_hmm(X, n)
    raw = m.predict(X)
    mapping = _compute_mapping(raw, features.values, n)
    return _apply_mapping(raw, mapping)


def detect_kmeans(features: pd.DataFrame, n_regimes: int | None = None) -> np.ndarray:
    X   = StandardScaler().fit_transform(features)
    n   = n_regimes or auto_select_k(X)
    km  = _fit_kmeans(X, n)
    raw = km.labels_
    sil = silhouette_score(X, raw)
    print(f"    KMeans silhouette score: {sil:.4f}")
    mapping = _compute_mapping(raw, features.values, n)
    return _apply_mapping(raw, mapping)


def detect_ensemble(features: pd.DataFrame, n_regimes: int | None = None) -> np.ndarray:
    """
    Majority-vote ensemble of HMM + KMeans.
    Both models share the same k (auto-selected once).
    Where they agree the label is kept; ties go to HMM (probabilistic model).
    """
    X = StandardScaler().fit_transform(features)
    n = n_regimes or auto_select_k(X)

    hmm_raw = _fit_hmm(X, n).predict(X)
    km_raw  = _fit_kmeans(X, n).labels_

    hmm_labels = _apply_mapping(hmm_raw, _compute_mapping(hmm_raw, features.values, n))
    km_labels  = _apply_mapping(km_raw,  _compute_mapping(km_raw,  features.values, n))

    ensemble = np.where(hmm_labels == km_labels, hmm_labels, hmm_labels)
    sil = silhouette_score(X, ensemble)
    print(f"    Ensemble silhouette score: {sil:.4f}")
    return ensemble


def walk_forward_detect(
    features: pd.DataFrame,
    n_regimes: int | None = None,
    train_days: int = 252,
    step_days:  int = 63,
) -> np.ndarray:
    """
    Rolling-window HMM re-fit: train on the last train_days, predict the
    next step_days, then slide forward.  Avoids look-ahead bias and keeps
    the model calibrated to recent market structure.
    """
    X_scaled  = StandardScaler().fit_transform(features)
    feat_vals = features.values
    n         = len(features)
    n_reg     = n_regimes or _DEFAULT_K
    regimes   = np.full(n, -1, dtype=int)

    # Seed: fit on the initial training window
    init_end = min(train_days, n)
    m0   = _fit_hmm(X_scaled[:init_end], n_reg)
    raw0 = m0.predict(X_scaled[:init_end])
    mapping0 = _compute_mapping(raw0, feat_vals[:init_end], n_reg)
    regimes[:init_end] = _apply_mapping(raw0, mapping0)

    for end in range(train_days, n, step_days):
        t_start = max(0, end - train_days)
        X_train = X_scaled[t_start:end]
        f_train = feat_vals[t_start:end]
        try:
            m       = _fit_hmm(X_train, n_reg)
            raw_tr  = m.predict(X_train)
            mapping = _compute_mapping(raw_tr, f_train, n_reg)

            pred_end = min(end + step_days, n)
            X_pred   = X_scaled[end:pred_end]
            if len(X_pred) == 0:
                continue
            raw_pred = m.predict(X_pred)
            regimes[end:pred_end] = _apply_mapping(raw_pred, mapping)
        except Exception:
            # Carry forward previous label on rare fit failures
            regimes[end:min(end + step_days, n)] = regimes[end - 1]

    return regimes


# ── Regime smoothing ──────────────────────────────────────────────────────────

def smooth_regimes(regimes: np.ndarray, min_days: int = 3) -> np.ndarray:
    """
    Replace runs shorter than min_days with the preceding regime.
    Reduces whipsaw signal changes caused by single-day label flips.
    """
    if min_days <= 1:
        return regimes
    smoothed = regimes.copy()
    n = len(smoothed)
    i = 0
    while i < n:
        r = smoothed[i]
        j = i + 1
        while j < n and smoothed[j] == r:
            j += 1
        if (j - i) < min_days and i > 0:
            smoothed[i:j] = smoothed[i - 1]
        i = j
    return smoothed


# ── Transition matrix ─────────────────────────────────────────────────────────

def transition_matrix(regimes: np.ndarray, n_regimes: int = 3) -> pd.DataFrame:
    mat = np.zeros((n_regimes, n_regimes))
    for a, b in zip(regimes[:-1], regimes[1:]):
        if 0 <= a < n_regimes and 0 <= b < n_regimes:
            mat[a, b] += 1
    mat /= mat.sum(axis=1, keepdims=True) + 1e-9
    names = [REGIME_NAMES[i] for i in range(n_regimes)]
    return pd.DataFrame(mat, index=names, columns=names)
