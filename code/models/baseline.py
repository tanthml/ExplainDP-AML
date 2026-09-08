"""Centralized MLP and XGBoost baselines (non-federated, non-DP)."""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HIDDEN_DIMS, DROPOUT, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED
from models.mlp import MLP


def make_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    device_str: str = "cpu",
    lr: float = None,
) -> tuple:
    device = torch.device(device_str)
    input_dim = X_train.shape[1]

    model = MLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    class_weights = make_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr if lr is not None else LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_val_f1, best_weights = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
            probs_val = torch.softmax(logits, dim=1).cpu().numpy()
        from sklearn.metrics import precision_recall_curve
        scores_val = probs_val[:, 1]
        prec_v, rec_v, thresh_v = precision_recall_curve(y_val, scores_val)
        f1v = np.where(prec_v + rec_v > 0, 2 * prec_v * rec_v / (prec_v + rec_v), 0.0)
        best_t = float(thresh_v[np.argmax(f1v)]) if len(thresh_v) else 0.5
        preds = (scores_val >= best_t).astype(int)
        val_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.get_weights()

    model.set_weights(best_weights)
    return model, best_val_f1


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, device_str: str = "cpu") -> dict:
    import numpy as np_inner
    device = torch.device(device_str)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    from sklearn.metrics import precision_recall_curve

    scores = probs[:, 1]
    auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else 0.0

    # Threshold-optimized F1 via PR curve — O(n log n), avoids O(200n) scan loop.
    prec, rec, thresh = precision_recall_curve(y, scores)
    f1_cls1 = np_inner.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    best_idx = int(np_inner.argmax(f1_cls1))
    best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5

    preds = (scores >= best_thresh).astype(int)
    best_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    report = classification_report(y, preds, zero_division=0, output_dict=True)
    return {"f1_macro": best_f1, "auc": auc, "report": report, "probs": probs, "preds": preds,
            "threshold": best_thresh}


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> object:
    from xgboost import XGBClassifier

    counts = np.bincount(y_train)
    scale_pos = counts[0] / max(counts[1], 1)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_xgboost(clf, X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_curve
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs) if len(np.unique(y)) == 2 else 0.0
    prec, rec, thresh = precision_recall_curve(y, probs)
    f1_cls1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    best_idx = int(np.argmax(f1_cls1))
    best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5
    preds = (probs >= best_thresh).astype(int)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    report = classification_report(y, preds, zero_division=0, output_dict=True)
    return {"f1_macro": f1, "auc": auc, "report": report, "probs": probs, "preds": preds,
            "threshold": best_thresh}
