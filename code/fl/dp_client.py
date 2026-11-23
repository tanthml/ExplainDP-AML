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


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor,
               gamma: float = 2.0) -> torch.Tensor:
    """Per-sample focal loss (Lin et al. 2017), combined with the same
    class-prior weights used elsewhere in this codebase: the focusing term
    (1-p_t)^gamma down-weights already-easy (high-confidence-correct)
    examples, applied on top of (not instead of) class weighting, which
    serves the role of the standard alpha_t term. Returns a weighted-mean
    scalar, with the same sum-of-weights normalisation `nn.CrossEntropyLoss`
    itself uses, so it is directly comparable in scale to the weighted CE
    loss used elsewhere."""
    ce = nn.functional.cross_entropy(logits, targets, reduction="none")  # unweighted, for a clean p_t
    pt = torch.exp(-ce)
    w = class_weights[targets]
    per_sample = w * (1 - pt).pow(gamma) * ce
    return per_sample.sum() / w.sum()


class DPFLClient:
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_kwargs: dict,
        device: torch.device,
        max_samples: int = DP_MAX_SAMPLES,
        sampler: str = "positive_preserving",
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
        self.sampler = sampler  # "positive_preserving" (default) or "uniform"

        # Class weights from the ORIGINAL (full) distribution so that
        # the model is calibrated to the true class prior even when we
        # subsample during training.
        counts = np.bincount(y_train, minlength=2)
        w = 1.0 / (counts + 1e-9)
        w = w / w.sum() * len(counts)
        self.class_weights = torch.tensor(w, dtype=torch.float32).to(device)

    def _subsample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (X, y, inclusion_probability) for whichever sampler this
        client was configured with. `inclusion_probability` is this round's
        per-row P(row selected | its class), needed for inverse-probability
        weighting (reviewer point 6); it is 1.0 everywhere for rows that are
        always kept (e.g. every positive under "positive_preserving")."""
        n_total = len(self.X)
        if self.max_samples is None or n_total <= self.max_samples:
            return self.X, self.y, torch.ones(n_total)

        if self.sampler == "uniform":
            perm = torch.randperm(n_total)[: self.max_samples]
            incl_prob = torch.full((len(perm),), self.max_samples / n_total)
            return self.X[perm], self.y[perm], incl_prob

        # "positive_preserving" (default, the paper's main strategy):
        # keep all positives, uniformly subsample negatives to fill the rest.
        pos_idx = torch.where(self.y == 1)[0]
        neg_idx = torch.where(self.y == 0)[0]
        n_pos = len(pos_idx)
        n_neg = min(len(neg_idx), max(0, self.max_samples - n_pos))

        if n_neg == 0:
            perm = torch.randperm(n_pos)[:self.max_samples]
            idx = pos_idx[perm]
            return self.X[idx], self.y[idx], torch.ones(len(idx))

        neg_perm = torch.randperm(len(neg_idx))[:n_neg]
        idx = torch.cat([pos_idx, neg_idx[neg_perm]])
        incl_prob = torch.cat([
            torch.ones(n_pos),
            torch.full((n_neg,), n_neg / len(neg_idx)),
        ])
        perm2 = torch.randperm(len(idx))
        idx, incl_prob = idx[perm2], incl_prob[perm2]
        return self.X[idx], self.y[idx], incl_prob

    def train(
        self,
        global_weights: list,
        epochs: int = LOCAL_EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        target_epsilon: float = 5.0,
        delta: float = DELTA,
        max_grad_norm: float = MAX_GRAD_NORM,
        mu: float = 0.0,
        loss_type: str = "weighted",
    ) -> tuple:
        """`mu > 0` applies FedProx under DP-SGD. The (mu)*(p - p_global) proximal
        gradient is injected directly into each parameter's `.grad` AFTER Opacus's
        own clip-and-noise step (`optimizer.pre_step()`), not added to the loss
        before `.backward()`: Opacus's per-sample-gradient hooks capture gradients
        at the layer level (from each layer's per-sample input activations and
        output gradients), so a term like (p - p_global)^2 that depends on a leaf
        parameter directly, with no per-sample data dependence, never reaches
        Opacus's `grad_sample` mechanism -- adding it to the loss changes the loss
        value but silently has zero effect on the trained weights (verified
        empirically before this fix). Since the proximal term is a deterministic
        function of already-public global and local weights, not of any raw data
        sample, injecting its (un-noised, un-clipped) gradient after the DP step
        adds no additional privacy cost and needs none of Opacus's per-sample
        machinery."""
        X_sub, y_sub, incl_prob = self._subsample()
        model = MLP(**self.model_kwargs).to(self.device)
        model.set_weights(global_weights)
        global_params = [w.clone().to(self.device) for w in global_weights] if mu > 0 else None

        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # drop_last=True is required by Opacus. incl_prob travels alongside
        # (X, y) so inverse-probability weighting (loss_type="ipw") can use
        # each row's actual per-round inclusion probability inside the loop.
        loader = DataLoader(
            TensorDataset(X_sub, y_sub, incl_prob),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        if len(loader) == 0:
            return global_weights, self.n_samples, float("inf"), 0.0, None

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
            for xb, yb, pb in loader:
                xb, yb, pb = xb.to(self.device), yb.to(self.device), pb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                if loss_type == "focal":
                    loss = focal_loss(logits, yb, self.class_weights)
                elif loss_type == "ipw":
                    # Inverse-probability weighting: reweight each sample by
                    # 1/inclusion_probability so the subsampled batch estimates
                    # the loss of the ORIGINAL (pre-subsampling) distribution,
                    # instead of (or in addition to) the fixed class-prior
                    # weights used by loss_type="weighted".
                    ce = nn.functional.cross_entropy(logits, yb, reduction="none")
                    ipw = 1.0 / pb.clamp(min=1e-6)
                    loss = (ce * ipw).sum() / ipw.sum()
                else:  # "weighted" (default): fixed class-prior weights
                    loss = nn.functional.cross_entropy(logits, yb, weight=self.class_weights)
                loss.backward()
                if global_params is not None:
                    if optimizer.pre_step():  # Opacus: clip per-sample grads, add noise, scale
                        for p, g in zip(model.parameters(), global_params):
                            if p.grad is not None:
                                p.grad.add_(mu * (p.detach() - g))
                        optimizer.original_optimizer.step()
                    # else: Opacus skipped this step (e.g. an empty Poisson-sampled
                    # batch); nothing to do, matching plain optimizer.step()'s own
                    # behaviour of silently no-op-ing in that case.
                else:
                    optimizer.step()
                total_loss += loss.item()
                steps += 1

        epsilon_spent = privacy_engine.get_epsilon(delta)

        # Accounting transparency: the raw parameters Opacus's own accountant
        # used, reported as-is (not re-derived) so the paper can cite exactly
        # what the library computed, per reviewer request for noise
        # multiplier / sampling probability / step count / per-client size.
        accounting_info = {
            "client_id": self.client_id,
            "n_raw": self.n_samples,
            "n_subsampled": len(X_sub),
            "sample_rate": getattr(loader, "sample_rate", batch_size / max(len(X_sub), 1)),
            "noise_multiplier": getattr(optimizer, "noise_multiplier", None),
            "steps": steps,
            "max_grad_norm": max_grad_norm,
            "target_epsilon": target_epsilon,
            "delta": delta,
            "epsilon_spent": epsilon_spent,
        }

        raw_model = model._module if hasattr(model, "_module") else model
        return raw_model.get_weights(), self.n_samples, epsilon_spent, total_loss / max(steps, 1), accounting_info
