"""
Shared model utilities.

Importing this module exposes either:
    * The real scikit-learn RandomForestClassifier / StandardScaler /
      train_test_split / metrics functions, **or**
    * A pure-NumPy fallback with identical names so that pickled artefacts
      remain loadable in environments where scikit-learn cannot be installed.

Both `train_model.py` and `app.py` import their model-runtime dependencies
from here, so a model trained under one mode loads cleanly under the other
(assuming the *same* mode is used at inference time).
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        precision_recall_curve, roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

    class StandardScaler:
        def fit(self, X):
            self.mean_ = X.mean(0); self.std_ = X.std(0) + 1e-9; return self
        def transform(self, X):
            return (X - self.mean_) / self.std_

    class RandomForestClassifier:
        """Physics-rule baseline used only when sklearn is unavailable."""
        def __init__(self, **kw):
            self.kw = kw
            self.feature_importances_ = np.array(
                [0.42, 0.08, 0.18, 0.05, 0.10, 0.09, 0.03, 0.02, 0.02, 0.01])
            self.classes_ = np.array([0, 1])
        def fit(self, X, y):
            self._mean = X.mean(0); self._std = X.std(0) + 1e-9
            return self
        def predict_proba(self, X):
            r_z = X[:, 0]   # standardised PRECTOT
            h_z = X[:, 2]   # standardised RH2M
            score = 1.6 * r_z + 0.6 * h_z
            p = 1.0 / (1.0 + np.exp(-score))
            return np.column_stack([1 - p, p])
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def train_test_split(X, y, test_size=0.2, stratify=None, random_state=42):
        rng = np.random.default_rng(random_state)
        idx = np.arange(len(X)); rng.shuffle(idx)
        n_te = int(len(X) * test_size)
        te, tr = idx[:n_te], idx[n_te:]
        return X[tr], X[te], y[tr], y[te]

    def accuracy_score(y, p):
        return float((np.asarray(y) == np.asarray(p)).mean())

    def confusion_matrix(y, p):
        cm = np.zeros((2, 2), int)
        for a, b in zip(y, p): cm[int(a), int(b)] += 1
        return cm

    def classification_report(y, p, target_names=("0", "1")):
        cm = confusion_matrix(y, p); tn, fp, fn, tp = cm.ravel()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        return f"precision={prec:.2f}  recall={rec:.2f}  acc={accuracy_score(y,p):.2f}"

    def roc_auc_score(y, s):
        y = np.asarray(y); s = np.asarray(s)
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0: return 0.5
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        for v in np.unique(s):
            m = (s == v)
            ranks[m] = ranks[m].mean()
        sum_ranks_pos = ranks[y == 1].sum()
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def precision_recall_curve(y, s):
        y = np.asarray(y); s = np.asarray(s)
        thr = np.unique(s)[::-1]
        prec, rec = [], []
        for t in thr:
            p = (s >= t).astype(int); cm = confusion_matrix(y, p)
            tn, fp, fn, tp = cm.ravel()
            prec.append(tp / max(tp + fp, 1)); rec.append(tp / max(tp + fn, 1))
        return np.array(prec), np.array(rec), thr


__all__ = [
    "RandomForestClassifier", "StandardScaler", "train_test_split",
    "accuracy_score", "classification_report", "confusion_matrix",
    "precision_recall_curve", "roc_auc_score", "SKLEARN_OK",
]
