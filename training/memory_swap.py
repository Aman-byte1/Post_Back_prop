"""
Post-Backprop 4B — Memory Swap
================================
Layer offloading and activation caching for layer-sequential training
on a single T4 GPU (16 GB VRAM).

Only ONE transformer layer + its PEPITA updater + its local prediction
head are in GPU memory at any time.  Intermediate activations (the
output of each layer) are cached to disk so the next layer can read
them as its input.
"""

import os
import torch
import numpy as np
from typing import Optional


class ActivationCache:
    """Disk-backed activation cache using numpy memory-mapped files.

    For each layer, after processing the full dataset, we store the
    layer's output activations as a memory-mapped float16 tensor on
    disk.  The next layer reads from this cache.
    """

    def __init__(self, cache_dir: str, n_samples: int, seq_len: int,
                 d_model: int, dtype: torch.dtype = torch.float16):
        self.cache_dir = cache_dir
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.d_model = d_model
        self.np_dtype = np.float16 if dtype == torch.float16 else np.float32
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, layer_idx: int) -> str:
        return os.path.join(self.cache_dir, f"layer_{layer_idx:03d}.dat")

    def create_writer(self, layer_idx: int) -> np.memmap:
        """Create a writable memmap for layer_idx's activations."""
        path = self._path(layer_idx)
        shape = (self.n_samples, self.seq_len, self.d_model)
        mm = np.memmap(path, dtype=self.np_dtype, mode="w+", shape=shape)
        return mm

    def open_reader(self, layer_idx: int) -> np.memmap:
        """Open a read-only memmap for layer_idx's activations."""
        path = self._path(layer_idx)
        shape = (self.n_samples, self.seq_len, self.d_model)
        mm = np.memmap(path, dtype=self.np_dtype, mode="r", shape=shape)
        return mm

    def read_batch(self, layer_idx: int, batch_indices,
                   device: torch.device) -> torch.Tensor:
        """Read a batch of activations into a GPU tensor.

        Args:
            layer_idx:     which layer's outputs to read.
            batch_indices: array of sample indices.
            device:        target torch device.
        Returns:
            (len(batch_indices), seq_len, d_model) tensor.
        """
        mm = self.open_reader(layer_idx)
        batch_np = mm[batch_indices]  # reads from disk
        return torch.from_numpy(batch_np.copy()).to(device)

    def write_batch(self, writer: np.memmap, batch_indices,
                    activations: torch.Tensor):
        """Write a batch of activations to the memmap.

        Args:
            writer:        writable memmap from create_writer().
            batch_indices: array of sample indices.
            activations:   (batch, seq, d_model) tensor.
        """
        writer[batch_indices] = activations.cpu().numpy()

    def layer_exists(self, layer_idx: int) -> bool:
        return os.path.exists(self._path(layer_idx))

    def cleanup(self, keep_layers: Optional[set] = None):
        """Remove cached activations (optionally keeping some layers)."""
        for f in os.listdir(self.cache_dir):
            if f.startswith("layer_") and f.endswith(".dat"):
                idx = int(f.split("_")[1].split(".")[0])
                if keep_layers is None or idx not in keep_layers:
                    os.remove(os.path.join(self.cache_dir, f))


def save_checkpoint(layer_idx: int, layer, head, updater,
                    checkpoint_dir: str):
    """Save a trained layer's weights to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"layer_{layer_idx:03d}.pt")
    torch.save({
        "layer": layer.state_dict(),
        "head": head.state_dict(),
    }, path)


def load_checkpoint(layer_idx: int, layer, head,
                    checkpoint_dir: str) -> bool:
    """Load a layer checkpoint if it exists.  Returns True if loaded."""
    path = os.path.join(checkpoint_dir, f"layer_{layer_idx:03d}.pt")
    if not os.path.exists(path):
        return False
    data = torch.load(path, map_location="cpu", weights_only=True)
    layer.load_state_dict(data["layer"])
    head.load_state_dict(data["head"])
    return True
