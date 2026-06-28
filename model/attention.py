"""
Post-Backprop 4B — Linear Attention
=====================================
Linear attention replaces the softmax(QK^T/√d)V computation with
    φ(Q) · (φ(K)^T · V)
where φ(x) = elu(x) + 1.

This makes every operation a pure matrix multiplication, which:
  1. Preserves CHL-compatible transpose symmetry.
  2. Reduces attention from O(T²d) to O(Td²) — faster for long contexts.
  3. Avoids the non-differentiable-in-the-CHL-sense softmax.

All operations are manual (no autograd graph is constructed).
"""

import torch
import torch.nn.functional as F


def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Feature map φ(x) = elu(x) + 1.  Always positive."""
    return F.elu(x) + 1.0


class LinearAttention:
    """Multi-head linear attention with explicit weight matrices.

    Weights:
        W_Q, W_K, W_V: (d_model, d_model)
        W_O:            (d_model, d_model)

    All stored as plain tensors (requires_grad=False).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 init_std: float = 0.02, dtype: torch.dtype = torch.float16):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        # Projection matrices — plain tensors, no autograd
        self.W_Q = torch.randn(d_model, d_model, dtype=dtype) * init_std
        self.W_K = torch.randn(d_model, d_model, dtype=dtype) * init_std
        self.W_V = torch.randn(d_model, d_model, dtype=dtype) * init_std
        self.W_O = torch.randn(d_model, d_model, dtype=dtype) * init_std

    # ------------------------------------------------------------------ #
    #  Forward pass — returns output AND all intermediates needed for the
    #  PEPITA weight update (pre/post-synaptic activations per matrix).
    # ------------------------------------------------------------------ #
    def forward(self, h: torch.Tensor, causal: bool = True):
        """
        Args:
            h:      (B, T, D)  input hidden states.
            causal: if True, apply causal (left-to-right) linear attention.
        Returns:
            out:    (B, T, D)  attention output.
            cache:  dict of intermediate activations for PEPITA updates.
        """
        B, T, D = h.shape
        H, K = self.n_heads, self.d_head

        # --- Projections (linear layers) ---
        Q = h @ self.W_Q                       # (B, T, D)
        Kmat = h @ self.W_K                    # (B, T, D)
        V = h @ self.W_V                       # (B, T, D)

        # --- Reshape to multi-head ---
        Q = Q.view(B, T, H, K).transpose(1, 2)      # (B, H, T, K)
        Kmat = Kmat.view(B, T, H, K).transpose(1, 2)  # (B, H, T, K)
        V = V.view(B, T, H, K).transpose(1, 2)      # (B, H, T, K)

        # --- Feature maps ---
        Q_feat = _elu_plus_one(Q)    # (B, H, T, K), all positive
        K_feat = _elu_plus_one(Kmat) # (B, H, T, K)

        if causal:
            # Causal linear attention via cumulative sum trick
            # Avoids materializing T×T matrix
            attn_out = _causal_linear_attention(Q_feat, K_feat, V)
        else:
            # Non-causal: standard linear attention
            KV = K_feat.transpose(-2, -1) @ V        # (B, H, K, K)
            Z = K_feat.transpose(-2, -1).sum(dim=-1, keepdim=True)  # (B, H, K, 1)
            attn_out = Q_feat @ KV                    # (B, H, T, K)
            # Normalize by sum of kernel values for numerical stability
            denom = Q_feat @ Z                        # (B, H, T, 1)
            attn_out = attn_out / denom.clamp(min=1e-6)

        # --- Reshape back ---
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # --- Output projection ---
        out = attn_out @ self.W_O    # (B, T, D)

        # Cache intermediates for PEPITA
        cache = {
            "h_in": h,             # pre-synaptic for Q, K, V
            "Q": Q.view(B, T, D),  # post-synaptic for W_Q (reshaped flat)
            "K": Kmat.view(B, T, D),
            "V": V.view(B, T, D),
            "attn_ctx": attn_out,  # pre-synaptic for W_O
            "out": out,            # post-synaptic for W_O
        }
        return out, cache

    def get_weight_pairs(self):
        """Return (name, weight, pre_key, post_key) for PEPITA updates."""
        return [
            ("W_Q", self.W_Q, "h_in", "Q"),
            ("W_K", self.W_K, "h_in", "K"),
            ("W_V", self.W_V, "h_in", "V"),
            ("W_O", self.W_O, "attn_ctx", "out"),
        ]

    def to(self, device):
        self.W_Q = self.W_Q.to(device)
        self.W_K = self.W_K.to(device)
        self.W_V = self.W_V.to(device)
        self.W_O = self.W_O.to(device)
        return self

    def state_dict(self):
        return {
            "W_Q": self.W_Q, "W_K": self.W_K,
            "W_V": self.W_V, "W_O": self.W_O,
        }

    def load_state_dict(self, d):
        self.W_Q = d["W_Q"]
        self.W_K = d["W_K"]
        self.W_V = d["W_V"]
        self.W_O = d["W_O"]


def _causal_linear_attention(Q: torch.Tensor, K: torch.Tensor,
                             V: torch.Tensor) -> torch.Tensor:
    """Causal linear attention via the cumulative-sum trick.

    Instead of materialising a T×T attention matrix, we maintain a
    running (K, K)-shaped state matrix S_t = Σ_{i≤t} k_i v_i^T and
    compute output_t = Q_t · S_t.

    Complexity: O(B·H·T·K²)  vs  O(B·H·T²·K) for standard attention.

    Args:
        Q, K, V: (B, H, T, K) with positive feature-mapped Q and K.
    Returns:
        out: (B, H, T, K)
    """
    B, H, T, K = Q.shape
    device = Q.device
    dtype = Q.dtype

    # Accumulate KV outer products cumulatively
    # K: (B, H, T, K), V: (B, H, T, K)
    # We want S_t = sum_{i=1}^{t} K_i^T V_i  →  shape (B, H, K, K)
    # Then out_t = Q_t @ S_t

    # Efficient vectorised implementation using cumsum on the
    # outer-product tensor KV_{t} = k_t ⊗ v_t
    KV = torch.einsum("bhti,bhtj->bhtij", K, V)  # (B, H, T, K, K)
    S = KV.cumsum(dim=2)                           # (B, H, T, K, K)

    # out_t = Q_t @ S_t → einsum over the K dimension
    out = torch.einsum("bhti,bhtij->bhtj", Q, S)  # (B, H, T, K)

    # Normalise by cumulative key sums for stability
    Z = K.cumsum(dim=2)                            # (B, H, T, K)
    denom = (Q * Z).sum(dim=-1, keepdim=True)      # (B, H, T, 1)
    out = out / denom.clamp(min=1e-6)

    return out
