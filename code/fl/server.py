"""FedAvg aggregation server (pure simulation — no network)."""

import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.utils import select_f1_threshold


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


def evaluate_global(model, X: np.ndarray, y: np.ndarray, device: torch.device,
                     threshold: float = None) -> dict:
    """Evaluate at a fixed threshold, or select one via the PR curve if none is given.

    Standard for highly imbalanced data (0.1% illicit) where argmax(0.5) always
    predicts the majority class. Pass `threshold=None` on a validation split to
    select it, then pass the returned value back in as `threshold` on the test
    split -- selecting on the test split itself leaks test labels into the
    reported number.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    # Batched forward pass: the test set is 1M+ rows, and an unbatched forward
    # through this MLP's 512-dim first hidden layer briefly needs several GB
    # of activation memory (observed to OOM a 12GB GPU when another process
    # was sharing it) -- batching keeps peak memory flat regardless of split
    # size or how many other jobs share the GPU.
    eval_batch = 200_000
    X_t = torch.tensor(X, dtype=torch.float32)
    probs_chunks = []
    with torch.no_grad():
        for i in range(0, len(X_t), eval_batch):
            logits = model(X_t[i:i + eval_batch].to(device))
            probs_chunks.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(probs_chunks).numpy()

    scores = probs[:, 1]
    auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else 0.0

    if threshold is None:
        threshold = select_f1_threshold(y, scores)

    preds = (scores >= threshold).astype(int)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return {"f1_macro": float(f1), "auc": float(auc), "preds": preds, "probs": probs,
            "threshold": float(threshold)}
