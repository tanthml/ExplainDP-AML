"""FedAvg aggregation server (pure simulation — no network)."""

import torch
import numpy as np


def fedavg_aggregate(weights_list: list, num_samples_list: list) -> list:
    """Weighted average of model parameters proportional to dataset size."""
    total = sum(num_samples_list)
    aggregated = []
    for layer_tensors in zip(*weights_list):
        weighted = sum(
            (n / total) * t for n, t in zip(num_samples_list, layer_tensors)
        )
        aggregated.append(weighted)
    return aggregated


def evaluate_global(model, X: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    from sklearn.metrics import precision_recall_curve

    scores = probs[:, 1]
    auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else 0.0

    # Threshold-optimized F1 using PR curve — O(n log n), much faster than scanning.
    # Standard for highly imbalanced data (0.1% illicit) where argmax(0.5) always
    # predicts the majority class.
    prec, rec, thresh = precision_recall_curve(y, scores)
    f1_cls1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    best_idx = int(np.argmax(f1_cls1))
    best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5

    preds = (scores >= best_thresh).astype(int)
    best_f1 = f1_score(y, preds, average="macro", zero_division=0)
    return {"f1_macro": float(best_f1), "auc": float(auc), "preds": preds, "probs": probs,
            "threshold": float(best_thresh)}
