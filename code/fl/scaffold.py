"""SCAFFOLD (Karimireddy et al. 2020) for the non-private FedAvg baseline.

Non-DP only: combining SCAFFOLD's control-variate correction with DP-SGD
would require its own privacy analysis (the correction term, like FedProx's
proximal term, is a deterministic function of already-revealed weights, but
unlike FedProx it also depends on each client's locally-accumulated gradient
trajectory in a way that is not a simple post-hoc gradient injection -- we
did not have time in the camera-ready window to work this out carefully, so
it is left to future work, documented in the paper's Limitations).

Uses vanilla SGD, not Adam, matching Karimireddy et al.'s "option II" local
control-variate update, which assumes a plain SGD local step; mixing it with
Adam would invalidate the (1/(K*eta)) correction formula.
"""

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


class SCAFFOLDClient:
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
        self.sampler = sampler
        self.c_i = None  # local control variate, lazily zero-initialised on first round

        counts = np.bincount(y_train, minlength=2)
        w = 1.0 / (counts + 1e-9)
        w = w / w.sum() * len(counts)
        self.class_weights = torch.tensor(w, dtype=torch.float32).to(device)

    def _subsample(self, max_samples):
        """Identical to FLClient._subsample / DPFLClient._subsample."""
        n_total = len(self.X)
        if not max_samples or n_total <= max_samples:
            return self.X, self.y
        if self.sampler == "uniform":
            perm = torch.randperm(n_total)[:max_samples]
            return self.X[perm], self.y[perm]
        pos_idx = (self.y == 1).nonzero(as_tuple=True)[0]
        neg_idx = (self.y == 0).nonzero(as_tuple=True)[0]
        n_pos = len(pos_idx)
        n_neg = min(len(neg_idx), max(0, max_samples - n_pos))
        if n_neg == 0:
            perm = torch.randperm(n_pos)[:max_samples]
            return self.X[pos_idx[perm]], self.y[pos_idx[perm]]
        neg_perm = torch.randperm(len(neg_idx))[:n_neg]
        idx = torch.cat([pos_idx, neg_idx[neg_perm]])
        idx = idx[torch.randperm(len(idx))]
        return self.X[idx], self.y[idx]

    def train(
        self,
        global_weights: list,
        c_global: list,
        epochs: int = LOCAL_EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        max_samples: int = None,
        loss_type: str = "weighted",
    ):
        """Returns (delta_y, delta_c, n_samples, avg_loss), both deltas as
        CPU tensors, matching this codebase's get_weights()/aggregate
        convention elsewhere (fl/server.py)."""
        X_train, y_train = self._subsample(max_samples)

        model = MLP(**self.model_kwargs).to(self.device)
        model.set_weights(global_weights)
        x0 = [w.clone().to(self.device) for w in global_weights]
        c_global_dev = [c.to(self.device) for c in c_global]
        if self.c_i is None:
            self.c_i = [torch.zeros_like(p) for p in x0]

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                if loss_type == "focal":
                    loss = focal_loss(logits, yb, self.class_weights)
                else:
                    loss = nn.functional.cross_entropy(logits, yb, weight=self.class_weights)
                loss.backward()
                with torch.no_grad():
                    for p, ci, c in zip(model.parameters(), self.c_i, c_global_dev):
                        if p.grad is not None:
                            p.grad.add_(c - ci)  # SCAFFOLD correction: g_i - c_i + c
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        K = max(steps, 1)
        with torch.no_grad():
            new_c_i, delta_y, delta_c = [], [], []
            for p, x0_p, ci, c in zip(model.parameters(), x0, self.c_i, c_global_dev):
                y_final = p.detach()
                new_ci = ci - c + (x0_p - y_final) / (K * lr)
                new_c_i.append(new_ci)
                delta_y.append((y_final - x0_p).cpu())
                delta_c.append((new_ci - ci).cpu())
            self.c_i = new_c_i

        return delta_y, delta_c, self.n_samples, total_loss / K


def run_scaffold(clients, model_kwargs, X_val, y_val, num_rounds, device, max_samples=None,
                  lr=LEARNING_RATE, loss_type="weighted"):
    """Server loop. Unlike this codebase's sample-weighted FedAvg elsewhere,
    SCAFFOLD's own convergence analysis (Karimireddy et al. 2020) uses a
    simple per-client average for both the model and control-variate update
    (not weighted by client dataset size); kept faithful to that here rather
    than mixed with this codebase's usual weighting convention."""
    from fl.server import evaluate_global

    global_model = MLP(**model_kwargs).to(device)
    global_weights = global_model.get_weights()
    c_global = [torch.zeros_like(w) for w in global_weights]
    N = len(clients)
    history = []
    best_f1, best_weights, best_round, best_threshold = -1.0, None, None, None

    for r in range(1, num_rounds + 1):
        all_dy, all_dc, all_loss = [], [], []
        for c in clients:
            dy, dc, n, loss = c.train(global_weights, c_global, max_samples=max_samples, lr=lr, loss_type=loss_type)
            all_dy.append(dy)
            all_dc.append(dc)
            all_loss.append(loss)

        global_weights = [
            gw + sum(dy[i] for dy in all_dy) / N for i, gw in enumerate(global_weights)
        ]
        c_global = [
            c + sum(dc[i] for dc in all_dc) / N for i, c in enumerate(c_global)
        ]
        global_model.set_weights(global_weights)
        metrics = evaluate_global(global_model, X_val, y_val, device)
        avg_loss = float(np.mean(all_loss))
        history.append({"round": r, "val_f1_macro": metrics["f1_macro"], "val_auc": metrics["auc"], "loss": avg_loss})
        print(f"  Round {r:02d}  val F1={metrics['f1_macro']:.4f}  val AUC={metrics['auc']:.4f}  loss={avg_loss:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_weights = [w.clone() for w in global_weights]
            best_round = r
            best_threshold = metrics["threshold"]

    global_model.set_weights(best_weights)
    return global_model, history, best_round, best_threshold
