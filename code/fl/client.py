"""Non-DP federated learning client (plain FedAvg)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS
from models.mlp import MLP
from fl.dp_client import focal_loss


class FLClient:
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_kwargs: dict,
        device: torch.device,
        sampler: str = "positive_preserving",
    ):
        self.client_id = client_id
        self.X = torch.tensor(X_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.long)
        self.model_kwargs = model_kwargs
        self.device = device
        self.n_samples = len(y_train)
        self.sampler = sampler  # "positive_preserving" (default) or "uniform"

        # Class weights for imbalanced AML data (minlength=2 guards single-class clients)
        counts = np.bincount(y_train, minlength=2)
        w = 1.0 / (counts + 1e-9)
        w = w / w.sum() * len(counts)
        self.class_weights = torch.tensor(w, dtype=torch.float32).to(device)

    def _subsample(self, max_samples):
        """Mirrors DPFLClient._subsample (fl/dp_client.py): same two sampler
        modes and the same inclusion-probability bookkeeping needed for
        inverse-probability-weighted loss, kept here so the non-private
        matched baselines (reviewer point 6) use an identical sampling
        procedure to their DP-FL counterparts."""
        n_total = len(self.X)
        if not max_samples or n_total <= max_samples:
            return self.X, self.y, torch.ones(n_total)

        if self.sampler == "uniform":
            perm = torch.randperm(n_total)[:max_samples]
            incl_prob = torch.full((len(perm),), max_samples / n_total)
            return self.X[perm], self.y[perm], incl_prob

        pos_idx = (self.y == 1).nonzero(as_tuple=True)[0]
        neg_idx = (self.y == 0).nonzero(as_tuple=True)[0]
        n_pos = len(pos_idx)
        n_neg = min(len(neg_idx), max(0, max_samples - n_pos))
        if n_neg == 0:
            perm = torch.randperm(n_pos)[:max_samples]
            idx = pos_idx[perm]
            return self.X[idx], self.y[idx], torch.ones(len(idx))
        neg_perm = torch.randperm(len(neg_idx))[:n_neg]
        idx = torch.cat([pos_idx, neg_idx[neg_perm]])
        incl_prob = torch.cat([torch.ones(n_pos), torch.full((n_neg,), n_neg / len(neg_idx))])
        perm2 = torch.randperm(len(idx))
        return self.X[idx[perm2]], self.y[idx[perm2]], incl_prob[perm2]

    def train(
        self,
        global_weights: list,
        epochs: int = LOCAL_EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        max_samples: int = None,
        mu: float = 0.0,
        loss_type: str = "weighted",
    ) -> tuple:
        """`mu > 0` applies FedProx: adds (mu/2)*||theta_local - theta_global||^2
        to the loss every step, penalising local drift from the round's
        starting global model. `mu=0` (default) is plain FedAvg. `loss_type`
        in {"weighted" (default), "ipw", "focal"} mirrors DPFLClient.train."""
        X_train, y_train, incl_prob = self._subsample(max_samples)

        model = MLP(**self.model_kwargs).to(self.device)
        model.set_weights(global_weights)
        global_params = [w.clone().to(self.device) for w in global_weights] if mu > 0 else None

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X_train, y_train, incl_prob), batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            for xb, yb, pb in loader:
                xb, yb, pb = xb.to(self.device), yb.to(self.device), pb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                if loss_type == "focal":
                    loss = focal_loss(logits, yb, self.class_weights)
                elif loss_type == "ipw":
                    ce = nn.functional.cross_entropy(logits, yb, reduction="none")
                    ipw = 1.0 / pb.clamp(min=1e-6)
                    loss = (ce * ipw).sum() / ipw.sum()
                else:
                    loss = nn.functional.cross_entropy(logits, yb, weight=self.class_weights)
                if global_params is not None:
                    prox = sum((p - g).pow(2).sum() for p, g in zip(model.parameters(), global_params))
                    loss = loss + (mu / 2) * prox
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        return model.get_weights(), self.n_samples, total_loss / max(steps, 1)
