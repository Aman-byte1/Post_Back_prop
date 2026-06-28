"""
Post-Backprop 4B — Embeddings
==============================
Token and positional embeddings, randomly initialized and trained
from scratch via local learning rules (no autograd).

All tensors are created with requires_grad=False.
"""

import math
import torch
import torch.nn.functional as F


class TokenEmbedding:
    """Lookup table mapping token IDs → d_model vectors.

    Randomly initialized from N(0, init_std).  The same weight matrix
    is reused (transposed) as the LM head for next-token prediction,
    following weight tying.
    """

    def __init__(self, vocab_size: int, d_model: int, init_std: float = 0.02,
                 dtype: torch.dtype = torch.float16):
        self.weight = torch.randn(
            vocab_size, d_model, dtype=dtype
        ) * init_std  # (V, D)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq) of int64 token indices.
        Returns:
            (batch, seq, d_model) embeddings.
        """
        return self.weight[token_ids]

    def to(self, device):
        self.weight = self.weight.to(device)
        return self

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, d):
        self.weight = d["weight"]


class PositionalEmbedding:
    """Learnable absolute positional embeddings.

    Randomly initialized from N(0, init_std).  Trained via the same
    local PEPITA update applied to the embedding lookup.
    """

    def __init__(self, max_seq_len: int, d_model: int,
                 init_std: float = 0.02, dtype: torch.dtype = torch.float16):
        self.weight = torch.randn(
            max_seq_len, d_model, dtype=dtype
        ) * init_std  # (T, D)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: number of positions to return.
        Returns:
            (seq_len, d_model) positional embeddings.
        """
        return self.weight[:seq_len]

    def to(self, device):
        self.weight = self.weight.to(device)
        return self

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, d):
        self.weight = d["weight"]


class Embeddings:
    """Combined token + positional embeddings.

    Produces the input to layer 0 of the transformer:
        h_0 = TokenEmbed(ids) + PosEmbed(positions)
    """

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int,
                 init_std: float = 0.02, dtype: torch.dtype = torch.float16):
        self.token_embed = TokenEmbedding(vocab_size, d_model, init_std, dtype)
        self.pos_embed = PositionalEmbedding(max_seq_len, d_model, init_std, dtype)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq) int64.
        Returns:
            (batch, seq, d_model) float tensor.
        """
        B, T = token_ids.shape
        tok = self.token_embed.forward(token_ids)        # (B, T, D)
        pos = self.pos_embed.forward(T).unsqueeze(0)     # (1, T, D)
        return tok + pos

    def to(self, device):
        self.token_embed.to(device)
        self.pos_embed.to(device)
        return self

    def state_dict(self):
        return {
            "token_embed": self.token_embed.state_dict(),
            "pos_embed": self.pos_embed.state_dict(),
        }

    def load_state_dict(self, d):
        self.token_embed.load_state_dict(d["token_embed"])
        self.pos_embed.load_state_dict(d["pos_embed"])
