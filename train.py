"""
Post-Backprop 4B — Main Training Script
=========================================
Entry point for training a 4-billion-parameter transformer from
scratch using PEPITA-style local difference learning — zero autograd,
zero backpropagation.

Usage:
    # Step 1: Prepare data (excluded from timer)
    python train.py --prepare-data

    # Step 2: Train (timed — 180 min budget)
    python train.py --train

    # Step 3: Evaluate
    python train.py --evaluate

    # Step 4: Generate text
    python train.py --generate "Once upon a time"

    # Verify zero autograd
    python train.py --verify
"""

import os
import sys
import time
import argparse
import json
import torch

from config import Config


def cmd_prepare_data(cfg: Config, args):
    """Download and tokenise training data (excluded from timer)."""
    from data.dataset import prepare_dataset

    os.makedirs("data_cache", exist_ok=True)
    output_path = os.path.join("data_cache", "tokens.npy")

    if os.path.exists(output_path) and not args.force:
        print(f"[data] Token file already exists: {output_path}")
        print(f"       Use --force to regenerate.")
        return output_path

    prepare_dataset(
        output_path=output_path,
        max_tokens=cfg.training.max_tokens_per_layer,
        dataset_name=cfg.training.dataset_name,
        seq_len=cfg.training.seq_len,
    )
    return output_path


def cmd_train(cfg: Config, args):
    """Run the full layer-sequential PEPITA training loop."""
    from training.sequential import SequentialTrainer

    token_file = os.path.join("data_cache", "tokens.npy")
    if not os.path.exists(token_file):
        print("[train] Token file not found. Run --prepare-data first.")
        sys.exit(1)

    # Set seeds
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[train] WARNING: No GPU detected. Training will be extremely slow.")

    trainer = SequentialTrainer(cfg, token_file, device=device)

    # Print VRAM before training
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e6
        print(f"[train] GPU memory before training: {mem:.1f} MB")

    stats = trainer.train()

    # Save embeddings and final norm
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    torch.save(
        trainer.embeddings.state_dict(),
        os.path.join(cfg.training.checkpoint_dir, "embeddings.pt"),
    )
    torch.save(
        trainer.final_norm.state_dict(),
        os.path.join(cfg.training.checkpoint_dir, "final_norm.pt"),
    )

    # Save training stats
    stats_path = os.path.join(cfg.training.checkpoint_dir, "training_stats.json")
    # Convert numpy types to Python types for JSON
    def _to_serialisable(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serialisable(v) for v in obj]
        return obj

    with open(stats_path, "w") as f:
        json.dump(_to_serialisable(stats), f, indent=2)
    print(f"[train] Stats saved to {stats_path}")

    return stats


def cmd_evaluate(cfg: Config, args):
    """Evaluate on WikiText-2."""
    from evaluation.eval_metrics import evaluate_wikitext

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_wikitext(cfg, cfg.training.checkpoint_dir, device=device)

    results_path = os.path.join(cfg.training.checkpoint_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Results saved to {results_path}")
    return results


def cmd_generate(cfg: Config, args):
    """Generate text from a prompt."""
    from evaluation.eval_metrics import generate_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = generate_text(
        cfg, cfg.training.checkpoint_dir,
        prompt=args.generate,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print("\n" + "=" * 60)
    print("Generated Text:")
    print("=" * 60)
    print(text)
    print("=" * 60)
    return text


def cmd_verify(cfg: Config, args):
    """Run AST verification for zero autograd."""
    from utils.verify_ast import scan_directory

    source_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("Post-Backprop 4B — Autograd Verification")
    print("=" * 60)
    print(f"Scanning: {source_dir}")

    n_files, violations = scan_directory(source_dir)

    print(f"\nFiles scanned: {n_files}")
    print(f"Violations found: {len(violations)}")

    if violations:
        print("\nVIOLATIONS:")
        for v in violations:
            print(v)
        print("\nRESULT: [FAIL] autograd primitives detected")
        sys.exit(1)
    else:
        print("\nRESULT: [PASS] zero autograd usage confirmed")


def cmd_info(cfg: Config, args):
    """Print model configuration and parameter count."""
    print(cfg.summary())

    # Breakdown
    mc = cfg.model
    embed = mc.vocab_size * mc.d_model
    pos = mc.max_seq_len * mc.d_model
    per_layer = mc.params_per_layer
    total_transformer = mc.n_layers * per_layer
    total = mc.total_params

    print("\nParameter Breakdown:")
    print(f"  Token embeddings:  {embed:>15,}  ({embed/1e9:.3f} B)")
    print(f"  Position embeddings: {pos:>13,}  ({pos/1e6:.1f} M)")
    print(f"  Per layer:         {per_layer:>15,}  ({per_layer/1e6:.1f} M)")
    print(f"  All {mc.n_layers} layers:     {total_transformer:>15,}  ({total_transformer/1e9:.3f} B)")
    print(f"  Final norm:        {mc.d_model:>15,}")
    print(f"  {'-' * 40}")
    print(f"  TOTAL:             {total:>15,}  ({total/1e9:.3f} B)")

    # Memory estimates
    weight_mb = total * 2 / 1e6  # fp16
    per_layer_mb = per_layer * 2 / 1e6
    optimizer_mb = per_layer * 4 * 2 / 1e6  # 2 fp32 states per weight
    print(f"\nMemory Estimates (per layer in GPU):")
    print(f"  Weights (fp16):      {per_layer_mb:>8.1f} MB")
    print(f"  Optimizer (fp32 x2): {optimizer_mb:>8.1f} MB")
    print(f"  Total per layer:     {per_layer_mb + optimizer_mb:>8.1f} MB")
    print(f"\nFull model on disk (fp16): {weight_mb:>8.1f} MB ({weight_mb/1024:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Post-Backprop 4B — Train a 4B-param LLM with local learning"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-data", action="store_true",
                       help="Download and tokenise training data")
    group.add_argument("--train", action="store_true",
                       help="Run PEPITA layer-sequential training")
    group.add_argument("--evaluate", action="store_true",
                       help="Evaluate on WikiText-2")
    group.add_argument("--generate", type=str, metavar="PROMPT",
                       help="Generate text from a prompt")
    group.add_argument("--verify", action="store_true",
                       help="Verify zero autograd usage via AST")
    group.add_argument("--info", action="store_true",
                       help="Print model info and parameter count")

    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of data")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Generation temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k for generation")

    args = parser.parse_args()
    cfg = Config()

    if args.prepare_data:
        cmd_prepare_data(cfg, args)
    elif args.train:
        cmd_train(cfg, args)
    elif args.evaluate:
        cmd_evaluate(cfg, args)
    elif args.generate is not None:
        cmd_generate(cfg, args)
    elif args.verify:
        cmd_verify(cfg, args)
    elif args.info:
        cmd_info(cfg, args)


if __name__ == "__main__":
    main()
