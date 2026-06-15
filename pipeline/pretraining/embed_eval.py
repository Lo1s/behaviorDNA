"""
pipeline/pretraining/embed_eval.py
==================================
Phase 8.2 — **contrastive-native** evaluation of a frozen sequence encoder.

Phase 8 / 8.1 scored transfer by *reconstruction-error* AUC after fine-tuning the
whole autoencoder. That is the wrong probe for a contrastive encoder (it has no
trained decoder) and it inherits the "near-separable at random init" caveat. Here
we instead **freeze** the encoder, embed chunks through its 16-D bottleneck, and
score the *representation directly* with classical embedding-space detectors:

  * unsupervised one-class scorers fit on legit-only embeddings (the real
    deployment shape) — Mahalanobis (Ledoit-Wolf covariance), One-Class SVM, and
    kNN distance — each yielding a cheat-vs-legit ROC AUC;
  * a supervised linear probe (cross-validated logistic regression) as a
    representation-quality cross-check (is cheat *linearly* separable in the
    embedding?).

The decisive comparison is each pretrained encoder against a **random-init**
encoder under the *same* probe: anything above random means the objective learned
transferable structure. All scorers return "higher = more anomalous".
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pipeline.models.lstm_ae import LSTMAutoencoder, _select_device


@torch.no_grad()
def embed_chunks(
    model: LSTMAutoencoder,
    chunks: np.ndarray | torch.Tensor,
    *,
    device: str = "auto",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode ``(N, L, 8)`` chunks → ``(N, bottleneck_dim)`` frozen embeddings."""
    torch_device = _select_device(device)
    model = model.to(torch_device)
    model.eval()
    if isinstance(chunks, np.ndarray):
        chunks = torch.from_numpy(chunks).float()
    if chunks.numel() == 0:
        return np.zeros((0, model.bottleneck_dim), dtype=np.float32)
    out: list[np.ndarray] = []
    for start in range(0, chunks.size(0), batch_size):
        batch = chunks[start : start + batch_size].to(torch_device, non_blocking=True)
        out.append(model.encode(batch).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Unsupervised one-class scorers (fit on legit embeddings; higher = anomalous)
# ---------------------------------------------------------------------------
def mahalanobis_scores(train_legit: np.ndarray, eval_emb: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance to the legit embedding distribution."""
    cov = LedoitWolf().fit(train_legit)
    return cov.mahalanobis(eval_emb).astype(np.float64)


def ocsvm_scores(
    train_legit: np.ndarray, eval_emb: np.ndarray, *, nu: float = 0.1
) -> np.ndarray:
    """Negative One-Class-SVM decision function (RBF, standardised features)."""
    scaler = StandardScaler().fit(train_legit)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(
        scaler.transform(train_legit)
    )
    return -clf.decision_function(scaler.transform(eval_emb)).astype(np.float64)


def knn_scores(
    train_legit: np.ndarray, eval_emb: np.ndarray, *, k: int = 5
) -> np.ndarray:
    """Mean distance to the ``k`` nearest legit embeddings (anomaly = far)."""
    k = max(1, min(k, len(train_legit)))
    nn = NearestNeighbors(n_neighbors=k).fit(train_legit)
    dist, _ = nn.kneighbors(eval_emb)
    return dist.mean(axis=1).astype(np.float64)


SCORERS = {
    "mahalanobis": mahalanobis_scores,
    "ocsvm": ocsvm_scores,
    "knn": knn_scores,
}


def oneclass_auc(
    train_legit: np.ndarray,
    eval_legit: np.ndarray,
    eval_cheat: np.ndarray,
    scorer,
) -> float:
    """ROC AUC of a one-class ``scorer`` (fit on ``train_legit``) separating
    ``eval_cheat`` (label 1) from ``eval_legit`` (label 0)."""
    if len(train_legit) == 0 or len(eval_legit) == 0 or len(eval_cheat) == 0:
        return float("nan")
    s_legit = scorer(train_legit, eval_legit)
    s_cheat = scorer(train_legit, eval_cheat)
    y = np.r_[np.zeros(len(s_legit)), np.ones(len(s_cheat))]
    return float(roc_auc_score(y, np.r_[s_legit, s_cheat]))


# ---------------------------------------------------------------------------
# Supervised representation-quality probe
# ---------------------------------------------------------------------------
def linear_probe_auc(
    emb: np.ndarray, labels: np.ndarray, *, seed: int = 0, n_splits: int = 5
) -> float:
    """Cross-validated logistic-regression ROC AUC on the frozen embedding.

    Out-of-fold ``predict_proba`` over stratified folds → an honest measure of how
    *linearly separable* cheat is in the representation (supervised; complements
    the unsupervised one-class AUCs). Returns NaN if a class is too small to split.
    """
    labels = np.asarray(labels).astype(int)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return float("nan")
    splits = int(min(n_splits, counts.min()))
    if splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000)
    proba = cross_val_predict(clf, emb, labels, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, proba))
