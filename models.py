"""
Regime detection: HMM, KMeans (auto-k), Ensemble, Walk-forward.
All public detectors return (labels, regime_prob) where:
  labels      : int array in {0=TRENDING, 1=MEAN-REV, 2=VOLATILE}
  regime_prob : float array, confidence of the assigned label (0-1)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM

from .config import REGIME_NAMES
from .features import VOL_IDX, SLOPE_IDX

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):
        return it

log = logging.getLogger(__name__)

_DEFAULT_K = 3


# ── Semantic label helpers ────────────────────────────────────────────────────

def _compute_mapping(raw: np.ndarray, feat_vals: np.ndarray, n: int) -> dict:
    """
    Build raw_cluster -> semantic_label mapping from cluster centroids.
    Highest-vol cluster -> VOLATILE (2); highest |slope| among the rest ->
    TRENDING (0); everything else -> MEAN-REV (1).
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
    """Pick k in k_range that maximises KMeans silhouette score."""
    best_k, best_score = _DEFAULT_K, -1.0
    for k in k_range:
        km  = KMeans(n_clusters=k, n_init=20, random_state=42)
        lbl = km.fit_predict(X)
        if len(np.unique(lbl)) < 2:
            continue
        score = silhouette_score(X, lbl)
        if score > best_score:
            best_score, best_k = score, k
    log.info("KMeans auto-selected k=%d  (silhouette=%.4f)", best_k, best_score)
    return best_k


def auto_select_k_hmm(X: np.ndarray, k_range=range(2, 7)) -> int:
    """Pick k in k_range that minimises HMM BIC -- model-consistent component selection."""
    best_k, best_bic = _DEFAULT_K, np.inf
    for k in k_range:
        try:
            m = GaussianHMM(n_components=k, covariance_type="full",
                            n_iter=200, random_state=42)
            m.fit(X)
            n_params = k * k + k * X.shape[1] + k * X.shape[1] ** 2
            bic = -2 * m.score(X) * len(X) + n_params * np.log(len(X))
            if bic < best_bic:
                best_bic, best_k = bic, k
        except Exception:
            continue
    log.info("HMM auto-selected k=%d  (BIC=%.2f)", best_k, best_bic)
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

def detect_hmm(
    features: pd.DataFrame, n_regimes: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, regime_prob) -- prob is max HMM posterior per observation."""
    X       = StandardScaler().fit_transform(features)
    n       = n_regimes or auto_select_k_hmm(X)
    m       = _fit_hmm(X, n)
    raw     = m.predict(X)
    proba   = m.predict_proba(X).max(axis=1)
    mapping = _compute_mapping(raw, features.values, n)
    return _apply_mapping(raw, mapping), proba


def detect_kmeans(
    features: pd.DataFrame, n_regimes: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, regime_prob) -- prob is distance-based confidence (0-1)."""
    X       = StandardScaler().fit_transform(features)
    n       = n_regimes or auto_select_k(X)
    km      = _fit_kmeans(X, n)
    raw     = km.labels_
    sil     = silhouette_score(X, raw)
    log.info("KMeans silhouette score: %.4f", sil)
    mapping = _compute_mapping(raw, features.values, n)
    dists   = km.transform(X)
    min_d   = dists.min(axis=1)
    max_d   = dists.max(axis=1)
    proba   = 1.0 - min_d / (max_d + 1e-9)
    return _apply_mapping(raw, mapping), proba


def detect_ensemble(
    features: pd.DataFrame, n_regimes: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Majority-vote ensemble of HMM + KMeans.
    Both models share the same k (auto-selected via HMM BIC).
    Where they agree the label is kept; where they disagree KMeans acts as
    the second opinion. Confidence is the mean of both models' regime probabilities.
    """
    X   = StandardScaler().fit_transform(features)
    n   = n_regimes or auto_select_k_hmm(X)

    hmm_m = _fit_hmm(X, n)
    km_m  = _fit_kmeans(X, n)

    hmm_raw   = hmm_m.predict(X)
    km_raw    = km_m.labels_

    hmm_proba = hmm_m.predict_proba(X).max(axis=1)
    km_dists  = km_m.transform(X)
    km_proba  = 1.0 - km_dists.min(axis=1) / (km_dists.max(axis=1) + 1e-9)

    hmm_labels = _apply_mapping(hmm_raw, _compute_mapping(hmm_raw, features.values, n))
    km_labels  = _apply_mapping(km_raw,  _compute_mapping(km_raw,  features.values, n))

    ensemble    = np.where(hmm_labels == km_labels, hmm_labels, km_labels)
    regime_prob = (hmm_proba + km_proba) / 2.0

    sil = silhouette_score(X, ensemble)
    log.info("Ensemble silhouette score: %.4f", sil)
    return ensemble, regime_prob


def walk_forward_detect(
    features: pd.DataFrame,
    n_regimes: int | None = None,
    train_days: int = 252,
    step_days:  int = 63,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling-window HMM re-fit: train on the last train_days, predict the next
    step_days, then slide forward. Avoids look-ahead bias and keeps the model
    calibrated to recent market structure.
    Returns (labels, regime_prob).
    """
    X_scaled  = StandardScaler().fit_transform(features)
    feat_vals = features.values
    n         = len(features)
    n_reg     = n_regimes or _DEFAULT_K
    regimes   = np.full(n, -1, dtype=int)
    probs     = np.full(n, 0.5)

    init_end = min(train_days, n)
    m0       = _fit_hmm(X_scaled[:init_end], n_reg)
    raw0     = m0.predict(X_scaled[:init_end])
    proba0   = m0.predict_proba(X_scaled[:init_end]).max(axis=1)
    mapping0 = _compute_mapping(raw0, feat_vals[:init_end], n_reg)
    regimes[:init_end] = _apply_mapping(raw0, mapping0)
    probs[:init_end]   = proba0

    fold_ends = range(init_end, n, step_days)
    for end in _tqdm(fold_ends, desc="Walk-forward", unit="fold", leave=False):
        t_start = max(0, end - train_days)
        X_train = X_scaled[t_start:end]
        f_train = feat_vals[t_start:end]
        try:
            m        = _fit_hmm(X_train, n_reg)
            raw_tr   = m.predict(X_train)
            mapping  = _compute_mapping(raw_tr, f_train, n_reg)
            pred_end = min(end + step_days, n)
            X_pred   = X_scaled[end:pred_end]
            if len(X_pred) == 0:
                continue
            raw_pred  = m.predict(X_pred)
            prob_pred = m.predict_proba(X_pred).max(axis=1)
            regimes[end:pred_end] = _apply_mapping(raw_pred, mapping)
            probs[end:pred_end]   = prob_pred
            log.debug("Walk-forward fold end=%d window=[%d, %d)", end, t_start, end)
        except Exception:
            regimes[end:min(end + step_days, n)] = regimes[end - 1]
            probs[end:min(end + step_days, n)]   = probs[end - 1]

    return regimes, probs


# ── Regime smoothing ──────────────────────────────────────────────────────────

def smooth_regimes(regimes: np.ndarray, min_days: int = 3) -> np.ndarray:
    """Replace runs shorter than min_days with the preceding regime."""
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
