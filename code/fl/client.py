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


class FLClient:
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_kwargs: dict,
        device: torch.device,
    ):
        self.client_id = client_id
        self.X = torch.tensor(X_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.long)
        self.model_kwargs = model_kwargs
        self.device = device
        self.n_samples = len(y_train)

        # Class weights for imbalanced AML data (minlength=2 guards single-class clients)
        counts = np.bincount(y_train, minlength=2)
        w = 1.0 / (counts + 1e-9)
        w = w / w.sum() * len(counts)
        self.class_weights = torch.tensor(w, dtype=torch.float32).to(device)

    def train(
        self,
        global_weights: list,
        epochs: int = LOCAL_EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        max_samples: int = None,
    ) -> tuple:
        X_train, y_train = self.X, self.y
        if max_samples and len(X_train) > max_samples:
            # Stratified subsample: keep all positive + sample negative
            pos_idx = (y_train == 1).nonzero(as_tuple=True)[0]
            neg_idx = (y_train == 0).nonzero(as_tuple=True)[0]
            n_pos = len(pos_idx)
            n_neg = min(len(neg_idx), max(0, max_samples - n_pos))
            neg_perm = torch.randperm(len(neg_idx))[:n_neg]
            idx = torch.cat([pos_idx, neg_idx[neg_perm]])
            idx = idx[torch.randperm(len(idx))]
            X_train, y_train = X_train[idx], y_train[idx]

        model = MLP(**self.model_kwargs).to(self.device)
        model.set_weights(global_weights)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        return model.get_weights(), self.n_samples, total_loss / max(steps, 1)
