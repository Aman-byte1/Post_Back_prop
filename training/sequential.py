"""
Post-Backprop 4B — Sequential Trainer
=======================================
Layer-by-layer training loop implementing PEPITA local difference
learning for a 34-layer linear-attention transformer.

Training proceeds sequentially:
  for layer_idx in 0 .. 33:
      1. Load layer weights to GPU
      2. Read cached activations from layer_idx - 1
      3. For each batch:
           a. Free pass: forward through the layer
           b. Compute local error via prediction head
           c. Perturbed pass: forward with perturbed input
           d. PEPITA weight update: cross-correlation of input with
              output difference
      4. Cache this layer's output activations to disk
      5. Save checkpoint, offload layer to CPU

All weight updates are strictly local (no cross-layer gradient flow)
and computed without autograd.
"""

import os
import sys
import time
import math
import torch
import numpy as np
from typing import Optional

from config import Config, ModelConfig, TrainingConfig
from model.embeddings import Embeddings
from model.transformer_layer import TransformerLayer, RMSNorm
from learning.local_loss import LocalPredictionHead, compute_local_cross_entropy
from learning.pepita_update import PEPITAUpdater, compute_norm_updates
from training.memory_swap import ActivationCache, save_checkpoint
from data.dataset import TokenDataset, DataLoaderSimple


class SequentialTrainer:
    """Orchestrates layer-sequential PEPITA training."""

    def __init__(self, cfg: Config, token_file: str, device: str = "cuda"):
        self.cfg = cfg
        self.mc = cfg.model
        self.tc = cfg.training
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.tc.use_fp16 else torch.float32

        # Dataset
        self.dataset = TokenDataset(token_file, self.tc.seq_len)
        max_batches_per_layer = self.tc.max_tokens_per_layer // (
            self.tc.batch_size * self.tc.seq_len
        )
        self.loader = DataLoaderSimple(
            self.dataset, self.tc.batch_size,
            shuffle=True, seed=self.tc.seed,
            max_batches=max_batches_per_layer,
        )

        # Embeddings (kept on GPU throughout — small relative to layers)
        self.embeddings = Embeddings(
            self.mc.vocab_size, self.mc.max_seq_len, self.mc.d_model,
            self.mc.init_std, self.dtype,
        ).to(self.device)

        # Final RMSNorm (applied after last layer)
        self.final_norm = RMSNorm(self.mc.d_model, self.mc.norm_eps, self.dtype)
        self.final_norm.to(self.device)

        # Activation cache
        n_samples = len(self.dataset)
        self.act_cache = ActivationCache(
            self.tc.activation_cache_dir, n_samples,
            self.tc.seq_len, self.mc.d_model, self.dtype,
        )

        # Timer
        self.start_time = None
        self.max_seconds = self.tc.max_train_minutes * 60

    def _time_remaining(self) -> float:
        """Seconds remaining in the training budget."""
        if self.start_time is None:
            return float("inf")
        return self.max_seconds - (time.time() - self.start_time)

    def _log(self, msg: str):
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = self._time_remaining()
        print(f"[{elapsed/60:6.1f}m | {remaining/60:5.1f}m left] {msg}",
              flush=True)

    # ---------------------------------------------------------------- #
    #  Phase 0: Cache the embedding outputs as "layer -1" activations   #
    # ---------------------------------------------------------------- #
    def cache_embeddings(self):
        """Run all tokens through the embedding layer and cache to disk."""
        self._log("Caching embedding outputs...")
        writer = self.act_cache.create_writer(-1)  # layer -1
        n_samples = len(self.dataset)

        # Process in chunks to avoid OOM
        chunk = 256
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            # Gather input tokens
            xs = []
            for i in range(start, end):
                x, _ = self.dataset[i]
                xs.append(x)
            x_batch = torch.stack(xs).to(self.device)
            h = self.embeddings.forward(x_batch)
            self.act_cache.write_batch(writer, np.arange(start, end), h)

        writer.flush()
        self._log(f"  Embeddings cached: {n_samples:,} sequences")

    # ---------------------------------------------------------------- #
    #  Train one layer                                                  #
    # ---------------------------------------------------------------- #
    def train_layer(self, layer_idx: int) -> dict:
        """Train a single transformer layer using PEPITA.

        Returns a dict of training statistics.
        """
        mc, tc = self.mc, self.tc

        # --- Create layer, head, updater ---
        layer = TransformerLayer(
            mc.d_model, mc.n_heads, mc.d_head, mc.d_ffn,
            mc.norm_eps, mc.init_std, self.dtype,
        ).to(self.device)

        head = LocalPredictionHead(
            mc.d_model, mc.vocab_size,
            init_std=0.01, dtype=self.dtype,
        ).to(self.device)

        updater = PEPITAUpdater(
            mc.d_model, mc.vocab_size,
            perturbation_scale=tc.perturbation_scale,
            dtype=self.dtype,
            seed=layer_idx * 1000 + tc.seed,
        ).to(self.device)

        # --- Prepare activation writer for this layer's outputs ---
        writer = self.act_cache.create_writer(layer_idx)

        # --- Training loop ---
        losses = []
        update_norms_accum = {}
        batches_done = 0
        layer_start = time.time()

        # We iterate over the full dataset, applying PEPITA at each batch
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        np.random.seed(tc.seed + layer_idx)
        np.random.shuffle(indices)

        # Compute max batches for this layer
        time_per_layer = self.max_seconds / mc.n_layers
        max_batches = tc.max_tokens_per_layer // (tc.batch_size * tc.seq_len)

        b = 0
        while b < max_batches:
            # Time check
            if self._time_remaining() < 30:
                self._log("  ⚠ Time budget nearly exhausted, stopping layer")
                break
            layer_elapsed = time.time() - layer_start
            if layer_elapsed > time_per_layer * 1.1:
                self._log("  ⚠ Layer time budget exceeded, moving to next")
                break

            # Get batch indices
            batch_start = (b * tc.batch_size) % n_samples
            batch_indices = indices[batch_start: batch_start + tc.batch_size]
            if len(batch_indices) < tc.batch_size:
                # Wrap around
                np.random.shuffle(indices)
                batch_indices = indices[:tc.batch_size]

            # Read previous layer's cached activations
            h_prev = self.act_cache.read_batch(
                layer_idx - 1, batch_indices, self.device
            )

            # Read target tokens
            ys = []
            for i in batch_indices:
                _, y = self.dataset[i]
                ys.append(y)
            target_ids = torch.stack(ys).to(self.device)

            # ========== PEPITA TWO-PASS UPDATE ========== #

            # --- Pass 1: Free forward ---
            h_free, free_cache = layer.forward(h_prev, causal=True)

            # --- Compute local error via prediction head ---
            error = head.compute_error_and_update(
                h_free.detach(), target_ids, lr=tc.head_lr
            )

            # Log loss periodically
            if b % 100 == 0:
                logits = head.forward(h_free)
                loss_val = compute_local_cross_entropy(logits, target_ids)
                losses.append(loss_val)

            # --- Perturb input ---
            h_prev_perturbed = updater.perturb_input(h_prev, error)

            # --- Pass 2: Perturbed forward ---
            h_perturbed, perturbed_cache = layer.forward(
                h_prev_perturbed, causal=True
            )

            # --- Weight updates ---
            # Learning rate with warmup
            if b < tc.warmup_batches:
                lr_scale = (b + 1) / tc.warmup_batches
            elif tc.lr_schedule == "cosine":
                progress = (b - tc.warmup_batches) / max(
                    1, max_batches - tc.warmup_batches
                )
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            else:
                lr_scale = 1.0
            current_lr = tc.lr * lr_scale

            weight_pairs = layer.get_all_weight_pairs()
            norms = updater.compute_weight_updates(
                free_cache, perturbed_cache, weight_pairs, current_lr
            )
            for k, v in norms.items():
                update_norms_accum.setdefault(k, []).append(v)

            # --- Cache this layer's free-pass output for the next layer ---
            self.act_cache.write_batch(writer, batch_indices, h_free.detach())

            b += 1
            batches_done = b

            # Progress logging
            if b % 500 == 0:
                avg_loss = np.mean(losses[-10:]) if losses else float("nan")
                self._log(
                    f"  Layer {layer_idx:2d} | batch {b:5d}/{max_batches} | "
                    f"loss={avg_loss:.3f} | lr={current_lr:.2e}"
                )

        # Flush activation cache
        writer.flush()

        # Save checkpoint
        save_checkpoint(layer_idx, layer, head, updater, tc.checkpoint_dir)

        # Offload
        layer.cpu()
        head.to("cpu")
        updater.to("cpu")
        del layer, head, updater
        torch.cuda.empty_cache()

        # Stats
        layer_time = time.time() - layer_start
        avg_norms = {k: np.mean(v) for k, v in update_norms_accum.items()}
        stats = {
            "layer_idx": layer_idx,
            "batches": batches_done,
            "tokens": batches_done * tc.batch_size * tc.seq_len,
            "time_seconds": layer_time,
            "final_loss": losses[-1] if losses else float("nan"),
            "mean_loss": np.mean(losses) if losses else float("nan"),
            "update_norms": avg_norms,
        }
        self._log(
            f"  Layer {layer_idx:2d} done | {batches_done} batches | "
            f"{stats['tokens']:,} tokens | {layer_time:.1f}s | "
            f"loss={stats['final_loss']:.3f}"
        )
        return stats

    # ---------------------------------------------------------------- #
    #  Full training run                                                #
    # ---------------------------------------------------------------- #
    def train(self) -> list:
        """Run the full layer-sequential training pipeline.

        Returns a list of per-layer statistics dicts.
        """
        self.start_time = time.time()
        self._log("=" * 60)
        self._log("Post-Backprop 4B — Training Started")
        self._log(self.cfg.summary())
        self._log("=" * 60)

        # Phase 0: cache embeddings
        self.cache_embeddings()

        # Phase 1: train layers sequentially
        all_stats = []
        for layer_idx in range(self.mc.n_layers):
            if self._time_remaining() < 60:
                self._log(f"⚠ Stopping at layer {layer_idx} — time exhausted")
                break
            stats = self.train_layer(layer_idx)
            all_stats.append(stats)

        total_time = time.time() - self.start_time
        self._log("=" * 60)
        self._log(f"Training complete in {total_time / 60:.1f} minutes")
        total_tokens = sum(s["tokens"] for s in all_stats)
        self._log(f"Total tokens processed: {total_tokens:,}")
        self._log("=" * 60)

        return all_stats
