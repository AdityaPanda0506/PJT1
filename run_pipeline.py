"""
NSSG-DimNet Master Pipeline Runner
==================================
Executes end-to-end:
1. Phase 1: Data Ingestion, Multi-Aspect Unpacking, Stratified Continuous 70/15/15 Split, NRC-VAD Lexicon & Dual-Graph Generation.
2. Phases 2-5: NSSG-DimNet Training with Switch-Gating, Dual-Branch (Biaffine + H-NSG RGAT), Cross-Attention Fusion & Parallel Regression Heads.
3. Phase 6: Publication-Grade Test Evaluation and Metric Logging.
"""

import os
import argparse
from data_pipeline import build_data_loaders
from train import run_training
from evaluate import evaluate_test_set


def main():
    parser = argparse.ArgumentParser(description="Run NSSG-DimNet End-to-End Pipeline")
    parser.add_argument("--dataset_path", type=str, default="DimABSA_Final_Dataset_600.csv", help="Path to DimABSA dataset")
    parser.add_argument("--hindi_nrc_path", type=str, default="Hindi-NRC-VAD-Lexicon.txt", help="Path to Hindi NRC VAD Lexicon")
    parser.add_argument("--all_txt_path", type=str, default="all.txt", help="Path to all.txt language tag dataset")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Transformer backbone identifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for GNN and regression heads")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze transformer backbone for ultra-fast training")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on existing checkpoint")

    args = parser.parse_args()

    print("\n" + "="*80, flush=True)
    print("      NSSG-DimNet: Neuro-Symbolic Switch-Gated Dual-Graph Network", flush=True)
    print("      Dimensional Aspect-Based Sentiment Analysis on Code-Mixed Hinglish", flush=True)
    print("="*80, flush=True)

    if not args.eval_only:
        print("\n>>> STEP 1 & 2: Executing Data Processing & Model Training Pipeline...", flush=True)
        best_ckpt = run_training(
            dataset_path=args.dataset_path,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_name=args.model_name,
            freeze_backbone=args.freeze_backbone
        )
    else:
        best_ckpt = os.path.join(args.checkpoint_dir, "best_nssg_dimnet.pt")

    print("\n>>> STEP 3: Executing Final Test Set Evaluation & Metrics Calculation...", flush=True)
    evaluate_test_set(
        checkpoint_path=best_ckpt,
        dataset_path=args.dataset_path,
        output_csv="data/test_predictions.csv",
        batch_size=args.batch_size
    )
    print("\n>>> Full NSSG-DimNet Pipeline Execution Succeeded!", flush=True)


if __name__ == "__main__":
    main()
