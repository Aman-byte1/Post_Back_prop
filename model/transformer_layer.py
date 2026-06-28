"""
Post-Backprop 4B — Transformer Layer
======================================
A single transformer block:   RMSNorm → LinearAttention → residual
                             → RMSNorm → FFN            → residual

All operations are explicit tensor arithmetic — no nn.Module, no
autograd graph.  Every forward pass returns a cache of pre/post-
synaptic activations for every weight matrix so that PEPITA updates
can be computed locally.
"""

import torch
import torch.nn.functional as F
from model.attention import LinearAttention


# ------------------------------------------------------------------ #
#  RMSNorm (parameter-free variant + learnable gain)                  #
# ------------------------------------------------------------------ #
class RMSNorm:
    """Root-Mean-Square layer normalisation with a learnable gain vector.

    norm(x) = x / rms(x) * gain
    rms(x)  = sqrt(mean(x²) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-6,
                 dtype: torch.dtype = torch.float16):
        self.gain = torch.ones(d_model, dtype=dtype)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, then cast back
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * rms).to(x.dtype) * self.gain

    def to(self, device):
        self.gain = self.gain.to(device)
        return self

    def state_dict(self):
        return {"gain": self.gain}

    def load_state_dict(self, d):
        self.gain = d["gain"]


# ------------------------------------------------------------------ #
#  Feed-Forward Network (SwiGLU-free: plain up → GELU → down)        #
# ------------------------------------------------------------------ #
class FFN:
    """Two-layer feed-forward network with GELU activation.

    h → W_up → GELU → W_down → output

    Weights are plain tensors.  The forward pass returns intermediate
    activations for PEPITA updates.
    """

    def __init__(self, d_model: int, d_ffn: int,
                 init_std: float = 0.02, dtype: torch.dtype = torch.float16):
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.W_up = torch.randn(d_model, d_ffn, dtype=dtype) * init_std
        self.W_down = torch.randn(d_ffn, d_model, dtype=dtype) * init_std

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: (B, T, D)
        Returns:
            out:   (B, T, D)
            cache: dict with pre/post-synaptic activations for W_up and W_down
        """
        up = h @ self.W_up                     # (B, T, d_ffn)
        act = F.gelu(up.float()).to(h.dtype)   # GELU in fp32 then back
        out = act @ self.W_down                # (B, T, D)

        cache = {
            "ffn_in": h,      # pre-synaptic for W_up
            "ffn_up": up,     # post-synaptic for W_up  (pre-activation)
            "ffn_act": act,   # pre-synaptic for W_down (post-activation)
            "ffn_out": out,   # post-synaptic for W_down
        }
        return out, cache

    def get_weight_pairs(self):
        """Return (name, weight, pre_key, post_key) for PEPITA updates."""
        return [
            ("W_up",   self.W_up,   "ffn_in",  "ffn_up"),
            ("W_down", self.W_down, "ffn_act", "ffn_out"),
        ]

    def to(self, device):
        self.W_up = self.W_up.to(device)
        self.W_down = self.W_down.to(device)
        return self

    def state_dict(self):
        return {"W_up": self.W_up, "W_down": self.W_down}

    def load_state_dict(self, d):
        self.W_up = d["W_up"]
        self.W_down = d["W_down"]


# ------------------------------------------------------------------ #
#  Full Transformer Block                                             #
# ------------------------------------------------------------------ #
class TransformerLayer:
    """One transformer block.

    Architecture (pre-norm):
        h → norm1 → LinearAttention → + h (residual)
          → norm2 → FFN             → + (residual)

    Returns the block output plus a merged cache of every weight
    matrix's pre/post-synaptic activations.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 d_ffn: int, norm_eps: float = 1e-6,
                 init_std: float = 0.02, dtype: torch.dtype = torch.float16):
        self.norm1 = RMSNorm(d_model, norm_eps, dtype)
        self.attn = LinearAttention(d_model, n_heads, d_head, init_std, dtype)
        self.norm2 = RMSNorm(d_model, norm_eps, dtype)
        self.ffn = FFN(d_model, d_ffn, init_std, dtype)
        self.d_model = d_model

    def forward(self, h: torch.Tensor, causal: bool = True):
        """
        Args:
            h:      (B, T, D) input from previous layer.
            causal: whether attention is causal.
        Returns:
            h_out:  (B, T, D) output hidden states.
            cache:  dict of all intermediates for PEPITA.
        """
        # ----- Self-attention sub-block -----
        h_normed1 = self.norm1.forward(h)
        attn_out, attn_cache = self.attn.forward(h_normed1, causal=causal)
        h_mid = h + attn_out  # residual connection

        # ----- FFN sub-block -----
        h_normed2 = self.norm2.forward(h_mid)
        ffn_out, ffn_cache = self.ffn.forward(h_normed2)
        h_out = h_mid + ffn_out  # residual connection

        # Merge caches — prefix keys to avoid collision
        cache = {}
        for k, v in attn_cache.items():
            cache[f"attn.{k}"] = v
        for k, v in ffn_cache.items():
            cache[f"ffn.{k}"] = v
        cache["block_in"] = h
        cache["block_out"] = h_out

        return h_out, cache

    def get_all_weight_pairs(self):
        """All (name, weight_tensor, pre_cache_key, post_cache_key) tuples."""
        pairs = []
        for name, W, pre, post in self.attn.get_weight_pairs():
            pairs.append((f"attn.{name}", W, f"attn.{pre}", f"attn.{post}"))
        for name, W, pre, post in self.ffn.get_weight_pairs():
            pairs.append((f"ffn.{name}", W, f"ffn.{pre}", f"ffn.{post}"))
        return pairs

    def get_all_weights(self):
        """Flat list of all trainable weight tensors (for optimizer state)."""
        return [
            self.attn.W_Q, self.attn.W_K, self.attn.W_V, self.attn.W_O,
            self.ffn.W_up, self.ffn.W_down,
            self.norm1.gain, self.norm2.gain,
        ]

    def get_all_weight_names(self):
        """Corresponding names for get_all_weights()."""
        return [
            "attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O",
            "ffn.W_up", "ffn.W_down",
            "norm1.gain", "norm2.gain",
        ]

    def to(self, device):
        self.norm1.to(device)
        self.attn.to(device)
        self.norm2.to(device)
        self.ffn.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def state_dict(self):
        return {
            "norm1": self.norm1.state_dict(),
            "attn": self.attn.state_dict(),
            "norm2": self.norm2.state_dict(),
            "ffn": self.ffn.state_dict(),
        }

    def load_state_dict(self, d):
        self.norm1.load_state_dict(d["norm1"])
        self.attn.load_state_dict(d["attn"])
        self.norm2.load_state_dict(d["norm2"])
        self.ffn.load_state_dict(d["ffn"])
