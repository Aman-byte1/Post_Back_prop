"""
Post-Backprop 4B — Dataset
============================
Data loading and tokenisation for layer-sequential training.

We use the GPT-2 tokenizer (via tiktoken for speed) and stream
a curated subset of OpenWebText, pre-tokenised into fixed-length
chunks of seq_len tokens.
"""

import os
import math
import torch
import numpy as np
from typing import Optional


def get_tokenizer():
    """Return a GPT-2 byte-pair-encoding tokenizer via tiktoken."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


class TokenDataset:
    """Memory-mapped dataset of pre-tokenised sequences.

    Stores all tokens as a flat int32 numpy memmap, then yields
    contiguous (seq_len+1)-token windows (input + next-token target).
    """

    def __init__(self, token_file: str, seq_len: int):
        """
        Args:
            token_file: path to a flat .npy or .bin file of int32 tokens.
            seq_len:    context window size.
        """
        if token_file.endswith(".npy"):
            self.tokens = np.load(token_file, mmap_mode="r")
        else:
            self.tokens = np.memmap(token_file, dtype=np.int32, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.tokens)
        # Number of full (seq_len+1)-windows we can extract
        self.n_samples = (self.n_tokens - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        x = chunk[:-1]   # input tokens
        y = chunk[1:]     # target tokens (next-token labels)
        return x, y


class DataLoaderSimple:
    """Minimal dataloader that yields batches of (input, target) pairs.

    Avoids torch.utils.data.DataLoader overhead for maximum throughput
    on a single-GPU sequential training loop.
    """

    def __init__(self, dataset: TokenDataset, batch_size: int,
                 shuffle: bool = True, seed: int = 42,
                 max_batches: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_batches = max_batches
        self.rng = np.random.RandomState(seed)
        self.n_batches = len(dataset) // batch_size
        if max_batches is not None:
            self.n_batches = min(self.n_batches, max_batches)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        for b in range(self.n_batches):
            batch_idx = indices[b * self.batch_size: (b + 1) * self.batch_size]
            xs, ys = [], []
            for i in batch_idx:
                x, y = self.dataset[i]
                xs.append(x)
                ys.append(y)
            yield torch.stack(xs), torch.stack(ys)


def prepare_dataset(
    output_path: str,
    max_tokens: int = 15_000_000,
    dataset_name: str = "openwebtext",
    seq_len: int = 512,
):
    """Download, tokenise, and save a curated token file.

    This function is intended to be run BEFORE the training timer
    starts (data preparation is excluded from the 180-min budget).

    Args:
        output_path: where to save the flat token file (.npy).
        max_tokens:  cap on total tokens to retain.
        dataset_name: HuggingFace dataset identifier.
        seq_len:     context length (for logging only).
    """
    import tiktoken
    from datasets import load_dataset

    print(f"[data] Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    all_tokens = []
    n = 0
    for example in ds:
        text = example.get("text", "")
        if not text.strip():
            continue
        tokens = enc.encode_ordinary(text) + [eot]
        all_tokens.extend(tokens)
        n += len(tokens)
        if n >= max_tokens:
            break
        if n % 1_000_000 < 1000:
            print(f"  ... {n:,} tokens collected")

    all_tokens = all_tokens[:max_tokens]
    arr = np.array(all_tokens, dtype=np.int32)
    np.save(output_path, arr)
    n_seqs = (len(arr) - 1) // seq_len
    print(f"[data] Saved {len(arr):,} tokens ({n_seqs:,} sequences of {seq_len}) "
          f"to {output_path}")
    return output_path
