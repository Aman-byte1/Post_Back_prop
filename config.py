"""
Post-Backprop 4B — Configuration
=================================
Central hyperparameters for a 4.005-billion-parameter transformer
trained entirely with zero-gradient, local learning algorithms.

Architecture: 34-layer linear-attention transformer
Learning:     PEPITA-style local difference learning
Hardware:     Single NVIDIA T4 (16 GB VRAM)
Budget:       ≤ 180 minutes wall-clock training
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters."""

    # --- Vocabulary & Embeddings ---
    vocab_size: int = 50_257          # GPT-2 tokenizer vocabulary
    max_seq_len: int = 512            # Context window (reduced for compute)

    # --- Transformer Dimensions ---
    d_model: int = 3072               # Hidden dimension
    n_heads: int = 24                 # Number of attention heads
    d_head: int = 128                 # Per-head dimension (d_model / n_heads)
    d_ffn: int = 12_288               # FFN intermediate dimension (4 × d_model)
    n_layers: int = 34                # Depth → ~4.005B total parameters

    # --- Attention ---
    attention_type: str = "linear"    # "linear" (elu+1 kernel)

    # --- Normalization ---
    norm_eps: float = 1e-6            # RMSNorm epsilon

    # --- Initialization ---
    init_std: float = 0.02            # Std-dev for weight initialization

    @property
    def params_per_layer(self) -> int:
        """Trainable parameters in one transformer layer."""
        attn = 4 * self.d_model * self.d_model          # Q, K, V, O
        ffn = 2 * self.d_model * self.d_ffn              # up + down
        norms = 2 * self.d_model                         # 2 × RMSNorm
        return attn + ffn + norms

    @property
    def total_params(self) -> int:
        """Total model parameters (embeddings + transformer + LM head)."""
        embed = self.vocab_size * self.d_model            # token embeddings
        pos = self.max_seq_len * self.d_model             # position embeddings
        transformer = self.n_layers * self.params_per_layer
        final_norm = self.d_model                         # final RMSNorm
        # LM head is weight-tied with token embeddings → 0 extra
        return embed + pos + transformer + final_norm


@dataclass
class TrainingConfig:
    """Training-loop and PEPITA hyperparameters."""

    # --- Compute budget ---
    max_train_minutes: float = 150.0    # Wall-clock cap (30 min reserved for eval)
    batch_size: int = 2                 # Micro-batch on T4
    seq_len: int = 512                  # Matches ModelConfig.max_seq_len

    # --- PEPITA learning ---
    lr: float = 1e-3                    # Base learning rate
    lr_schedule: str = "cosine"         # "cosine" or "constant"
    warmup_batches: int = 200           # Linear warmup steps per layer
    perturbation_scale: float = 0.1     # Scale of the error injection into h_prev

    # --- Local prediction heads ---
    head_lr: float = 5e-3               # LR for per-layer prediction heads
    head_hidden: int = 0                # 0 = linear head; >0 = one hidden layer

    # --- Data ---
    dataset_name: str = "openwebtext"   # HuggingFace dataset identifier
    dataset_subset: Optional[str] = None
    max_tokens_per_layer: int = 15_000_000  # ~15 M tokens budget per layer
    num_workers: int = 2                # DataLoader workers

    # --- Layer swapping ---
    activation_cache_dir: str = "cache/activations"
    checkpoint_dir: str = "checkpoints"
    use_fp16: bool = True               # fp16 forward, fp32 optimizer accumulators

    # --- Reproducibility ---
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation hyperparameters."""

    eval_batch_size: int = 4
    wikitext_split: str = "test"
    max_eval_batches: Optional[int] = None  # None = full eval set


@dataclass
class Config:
    """Top-level configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        assert self.model.d_model == self.model.n_heads * self.model.d_head, (
            f"d_model ({self.model.d_model}) must equal "
            f"n_heads × d_head ({self.model.n_heads} × {self.model.d_head})"
        )
        assert self.training.seq_len == self.model.max_seq_len

    def summary(self) -> str:
        """Human-readable summary of the configuration."""
        m = self.model
        t = self.training
        total = m.total_params
        lines = [
            "=" * 60,
            "Post-Backprop 4B -- Configuration Summary",
            "=" * 60,
            f"  Layers:          {m.n_layers}",
            f"  d_model:         {m.d_model}",
            f"  Heads:           {m.n_heads} x {m.d_head}",
            f"  FFN dim:         {m.d_ffn}",
            f"  Attention:       {m.attention_type}",
            f"  Vocab:           {m.vocab_size:,}",
            f"  Context:         {m.max_seq_len}",
            f"  Params/layer:    {m.params_per_layer:,}",
            f"  Total params:    {total:,}  ({total / 1e9:.3f} B)",
            "-" * 60,
            f"  Batch size:      {t.batch_size}",
            f"  LR:              {t.lr}",
            f"  Perturb scale:   {t.perturbation_scale}",
            f"  Tokens/layer:    {t.max_tokens_per_layer:,}",
            f"  Time budget:     {t.max_train_minutes} min",
            "=" * 60,
        ]
        return "\n".join(lines)


# Quick sanity check when module is imported
_cfg = Config()
assert _cfg.model.total_params > 3_900_000_000, (
    f"Total params {_cfg.model.total_params:,} is below 3.9B target"
)
