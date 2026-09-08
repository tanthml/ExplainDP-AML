"""DP-SGD federated learning client using Opacus."""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("[warn] opacus not installed — DP client unavailable")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS, MAX_GRAD_NORM, DELTA
from models.mlp import MLP

# Maximum samples per client per round for DP training.
# This caps step count for CPU tractability (~69,000 -> ~195 steps/round on
# the largest client), not noise: verified via Opacus's own accountant
# (get_noise_multiplier, target_epsilon=5, delta=1e-5, E=5, B=256), the
# uncapped 3.5M-row client actually needs LESS noise (sigma~0.43) than this
# 10K cap (sigma~0.78), since capping raises the per-step sampling rate.
# Keeping all positives + sampling negatives also falls outside Opacus's
# accounting entirely (label-dependent subsampling happens before
# make_private_with_epsilon runs) -- see paper Sec. "Privacy Unit and
# Accounting Scope" for the resulting scoping caveats.
DP_MAX_SAMPLES = 10_000


class DPFLClient:
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_kwargs: dict,
        device: torch.device,
        max_samples: int = DP_MAX_SAMPLES,
    ):
        if not OPACUS_AVAILABLE:
            raise RuntimeError("opacus is required for DPFLClient")

        self.client_id = client_id
        self.X = torch.tensor(X_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.long)
        self.model_kwargs = model_kwargs
        self.device = device
        self.n_samples = len(y_train)
        self.max_samples = max_samples

        # Class weights from the ORIGINAL (full) distribution so that
        # the model is calibrated to the true class prior even when we
        # subsample during training.
        counts = np.bincount(y_train, minlength=2)
        w = 1.0 / (counts + 1e-9)
        w = w / w.sum() * len(counts)
        self.class_weights = torch.tensor(w, dtype=torch.float32).to(device)

    def _subsample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep all positive samples; randomly down-sample negatives to max_samples total."""
        if self.max_samples is None or len(self.X) <= self.max_samples:
            return self.X, self.y

        pos_idx = torch.where(self.y == 1)[0]
        neg_idx = torch.where(self.y == 0)[0]
        n_pos = len(pos_idx)
        n_neg = min(len(neg_idx), max(0, self.max_samples - n_pos))

        if n_neg == 0:
            perm = torch.randperm(n_pos)[:self.max_samples]
            return self.X[pos_idx[perm]], self.y[pos_idx[perm]]

        neg_perm = torch.randperm(len(neg_idx))[:n_neg]
        idx = torch.cat([pos_idx, neg_idx[neg_perm]])
        idx = idx[torch.randperm(len(idx))]
        return self.X[idx], self.y[idx]

    def train(
        self,
        global_weights: list,
        epochs: int = LOCAL_EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        target_epsilon: float = 5.0,
        delta: float = DELTA,
        max_grad_norm: float = MAX_GRAD_NORM,
    ) -> tuple:
        X_sub, y_sub = self._subsample()
        model = MLP(**self.model_kwargs).to(self.device)
        model.set_weights(global_weights)

        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # drop_last=True is required by Opacus
        loader = DataLoader(
            TensorDataset(X_sub, y_sub),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        if len(loader) == 0:
            return global_weights, self.n_samples, float("inf"), 0.0

        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )

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

        epsilon_spent = privacy_engine.get_epsilon(delta)

        raw_model = model._module if hasattr(model, "_module") else model
        return raw_model.get_weights(), self.n_samples, epsilon_spent, total_loss / max(steps, 1)
