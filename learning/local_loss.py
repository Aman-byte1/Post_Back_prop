"""
Post-Backprop 4B — Local Loss
===============================
Per-layer local prediction heads and error signal computation.

Each transformer layer gets a lightweight linear prediction head
that maps hidden states → vocabulary logits.  The cross-entropy
error between predicted and true next tokens produces a local
error vector e_l that drives PEPITA weight updates.

The prediction heads themselves are trained via a manually-computed
gradient (no autograd) since they are single linear layers with
an analytically known gradient.
"""

import torch
import torch.nn.functional as F


class LocalPredictionHead:
    """A single linear layer:  h → h @ W_head^T → logits.

    This is a per-layer auxiliary predictor.  Its weight is updated
    via the manual softmax gradient (no autograd).
    """

    def __init__(self, d_model: int, vocab_size: int,
                 init_std: float = 0.01, dtype: torch.dtype = torch.float16):
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Shape: (vocab_size, d_model) — same layout as nn.Linear
        self.weight = torch.randn(
            vocab_size, d_model, dtype=dtype
        ) * init_std

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, D) hidden states.
        Returns:
            logits: (B, T, V) unnormalised log-probabilities.
        """
        return h @ self.weight.T  # (B, T, V)

    def compute_error_and_update(
        self, h: torch.Tensor, target_ids: torch.Tensor, lr: float
    ) -> torch.Tensor:
        """Compute local cross-entropy error vector and update head weights.

        Returns the error vector e = softmax(logits) - one_hot(target),
        which is the gradient of cross-entropy loss w.r.t. logits.
        This error vector is used by PEPITA to perturb the input for
        the second forward pass.

        The head weight update is:
            dW = -(1/N) · error^T @ h
        which is the exact gradient of CE w.r.t. W_head, computed
        manually without autograd.

        Args:
            h:          (B, T, D) hidden states from this layer.
            target_ids: (B, T)    ground-truth next-token labels.
            lr:         learning rate for the head.
        Returns:
            error:      (B, T, V) the softmax-minus-onehot error signal.
        """
        B, T, D = h.shape
        V = self.vocab_size

        # Forward through head
        logits = self.forward(h)                              # (B, T, V)

        # Softmax probabilities (in fp32 for stability)
        probs = F.softmax(logits.float(), dim=-1)             # (B, T, V) fp32

        # One-hot targets
        target_oh = F.one_hot(target_ids, V).float()          # (B, T, V) fp32

        # Error signal: gradient of CE w.r.t. logits
        error = probs - target_oh                              # (B, T, V) fp32

        # --- Manual gradient for head weight update ---
        # dL/dW = (1/N) · error^T @ h
        # where N = B * T (total tokens in batch)
        N = B * T
        error_flat = error.reshape(N, V)                       # (N, V)
        h_flat = h.float().reshape(N, D)                       # (N, D)
        grad_W = error_flat.T @ h_flat / N                     # (V, D)

        # Update head weight (gradient descent, no autograd)
        self.weight.data.add_(-lr * grad_W.to(self.weight.dtype))

        return error.to(h.dtype)

    def to(self, device):
        self.weight = self.weight.to(device)
        return self

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, d):
        self.weight = d["weight"]


def compute_local_cross_entropy(logits: torch.Tensor,
                                target_ids: torch.Tensor) -> float:
    """Compute cross-entropy loss for monitoring (no gradient needed).

    Args:
        logits:     (B, T, V)
        target_ids: (B, T)
    Returns:
        scalar loss value.
    """
    B, T, V = logits.shape
    logits_flat = logits.float().reshape(-1, V)
    targets_flat = target_ids.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss.item()
