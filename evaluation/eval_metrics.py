"""
Post-Backprop 4B — Evaluation Metrics
=======================================
Evaluation utilities for WikiText-2 perplexity and text generation.

The LM head uses weight-tying with the token embedding table,
so no separate output head needs training (the token embeddings
are trained during the sequential pass).

For benchmarks like HellaSwag and PIQA, we provide a log-likelihood
scoring function that scores each candidate completion and picks
the most probable.
"""

import os
import math
import time
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Optional

from config import Config, ModelConfig
from model.embeddings import Embeddings
from model.transformer_layer import TransformerLayer, RMSNorm


class InferenceModel:
    """Assembles the full model from saved layer checkpoints for inference.

    Runs the full forward pass through all layers sequentially,
    keeping only one layer in GPU memory at a time.
    """

    def __init__(self, cfg: Config, checkpoint_dir: str,
                 device: str = "cuda"):
        self.cfg = cfg
        self.mc = cfg.model
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.checkpoint_dir = checkpoint_dir

        # Embeddings (load from checkpoint or reconstruct)
        self.embeddings = Embeddings(
            self.mc.vocab_size, self.mc.max_seq_len, self.mc.d_model,
            self.mc.init_std, self.dtype,
        )
        embed_path = os.path.join(checkpoint_dir, "embeddings.pt")
        if os.path.exists(embed_path):
            data = torch.load(embed_path, map_location="cpu", weights_only=True)
            self.embeddings.load_state_dict(data)

        # Final norm
        self.final_norm = RMSNorm(self.mc.d_model, self.mc.norm_eps, self.dtype)
        norm_path = os.path.join(checkpoint_dir, "final_norm.pt")
        if os.path.exists(norm_path):
            data = torch.load(norm_path, map_location="cpu", weights_only=True)
            self.final_norm.load_state_dict(data)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass: embeddings → all layers → norm → logits.

        Processes layers sequentially, loading each from disk.

        Args:
            token_ids: (B, T) int64 tensor of token IDs.
        Returns:
            logits: (B, T, V) float tensor.
        """
        mc = self.mc
        device = self.device

        # Embeddings
        self.embeddings.to(device)
        h = self.embeddings.forward(token_ids.to(device))

        # Process each layer
        for layer_idx in range(mc.n_layers):
            layer = TransformerLayer(
                mc.d_model, mc.n_heads, mc.d_head, mc.d_ffn,
                mc.norm_eps, mc.init_std, self.dtype,
            )
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"layer_{layer_idx:03d}.pt"
            )
            if os.path.exists(ckpt_path):
                data = torch.load(ckpt_path, map_location="cpu",
                                  weights_only=True)
                layer.load_state_dict(data["layer"])
            layer.to(device)

            h, _ = layer.forward(h, causal=True)

            layer.cpu()
            del layer
            torch.cuda.empty_cache()

        # Final norm
        self.final_norm.to(device)
        h = self.final_norm.forward(h)

        # LM head: weight-tied with token embeddings
        # logits = h @ embed_table.T
        logits = h @ self.embeddings.token_embed.weight.T  # (B, T, V)

        return logits

    def compute_perplexity(self, token_ids: torch.Tensor) -> float:
        """Compute perplexity over a batch of token sequences.

        Args:
            token_ids: (B, T) where T includes both input and target positions.
                       We use tokens[:-1] as input and tokens[1:] as targets.
        Returns:
            perplexity: exp(mean cross-entropy).
        """
        B, T = token_ids.shape
        x = token_ids[:, :-1]     # (B, T-1)
        y = token_ids[:, 1:]      # (B, T-1)

        logits = self.forward(x)  # (B, T-1, V)

        # Cross-entropy
        logits_flat = logits.float().reshape(-1, self.mc.vocab_size)
        targets_flat = y.reshape(-1).to(logits_flat.device)
        ce = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
        return math.exp(ce.item())


def evaluate_wikitext(cfg: Config, checkpoint_dir: str,
                      device: str = "cuda") -> dict:
    """Evaluate on WikiText-2 test set.

    Returns:
        dict with 'perplexity' and 'cross_entropy'.
    """
    import tiktoken
    from datasets import load_dataset

    print("[eval] Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tiktoken.get_encoding("gpt2")

    # Tokenise all text
    all_tokens = []
    for example in ds:
        text = example.get("text", "")
        if text.strip():
            all_tokens.extend(enc.encode_ordinary(text))
    print(f"[eval] WikiText-2 test: {len(all_tokens):,} tokens")

    # Create sequences
    seq_len = cfg.model.max_seq_len
    n_seqs = len(all_tokens) // (seq_len + 1)
    tokens_arr = np.array(all_tokens[: n_seqs * (seq_len + 1)], dtype=np.int64)
    tokens_arr = tokens_arr.reshape(n_seqs, seq_len + 1)

    # Build inference model
    model = InferenceModel(cfg, checkpoint_dir, device)

    # Evaluate in batches
    eval_bs = cfg.eval.eval_batch_size
    total_ce = 0.0
    total_tokens = 0

    n_batches = (n_seqs + eval_bs - 1) // eval_bs
    if cfg.eval.max_eval_batches is not None:
        n_batches = min(n_batches, cfg.eval.max_eval_batches)

    print(f"[eval] Evaluating {n_batches} batches...")
    for i in range(n_batches):
        start = i * eval_bs
        end = min(start + eval_bs, n_seqs)
        batch = torch.from_numpy(tokens_arr[start:end])

        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)

        logits = model.forward(x)
        logits_flat = logits.float().reshape(-1, cfg.model.vocab_size)
        targets_flat = y.reshape(-1)
        ce = F.cross_entropy(logits_flat, targets_flat, reduction="sum")

        n_toks = targets_flat.numel()
        total_ce += ce.item()
        total_tokens += n_toks

        if (i + 1) % 10 == 0:
            running_ppl = math.exp(total_ce / total_tokens)
            print(f"  batch {i+1}/{n_batches} | running PPL = {running_ppl:.1f}")

    avg_ce = total_ce / total_tokens
    ppl = math.exp(avg_ce)
    print(f"[eval] WikiText-2 Perplexity: {ppl:.2f}")
    print(f"[eval] WikiText-2 Cross-Entropy: {avg_ce:.4f}")

    return {"perplexity": ppl, "cross_entropy": avg_ce, "total_tokens": total_tokens}


def generate_text(cfg: Config, checkpoint_dir: str,
                  prompt: str, max_new_tokens: int = 100,
                  temperature: float = 0.8, top_k: int = 50,
                  device: str = "cuda") -> str:
    """Generate text autoregressively from a prompt.

    Args:
        prompt:         input text string.
        max_new_tokens: how many tokens to generate.
        temperature:    sampling temperature (1.0 = neutral).
        top_k:          top-k filtering.
    Returns:
        generated text string (prompt + continuation).
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    model = InferenceModel(cfg, checkpoint_dir, device)

    for _ in range(max_new_tokens):
        # Truncate to max context
        context = tokens[-cfg.model.max_seq_len:]
        x = torch.tensor([context], dtype=torch.int64, device=device)

        logits = model.forward(x)          # (1, T, V)
        next_logits = logits[0, -1, :]     # (V,)

        # Temperature scaling
        next_logits = next_logits.float() / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, top_k)
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits.scatter_(0, topk_idx, topk_vals)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)

        # Stop on end-of-text
        if next_token == enc.eot_token:
            break

    return enc.decode(tokens)
