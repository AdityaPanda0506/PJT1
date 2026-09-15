"""
NSSG-DimNet Evaluation & Inference Module
=========================================
Evaluates the best trained checkpoint on the test split,
computes publication-grade metric tables, saves test predictions,
and bundles full model artifacts into a complete pickle (.pkl) deployment file.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional

from data_pipeline import build_data_loaders
from model import NSSGDimNet
from metrics import compute_metrics


def summarize_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Computes total, trainable, and frozen parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params
    }


@torch.no_grad()
def evaluate_test_set(
    checkpoint_path: str = "checkpoints/best_nssg_dimnet.pt",
    dataset_path: str = "DimABSA_Final_Dataset_600.csv",
    output_csv: str = "data/test_predictions.csv",
    output_pkl: str = "checkpoints/best_nssg_dimnet.pkl",
    batch_size: int = 16,
    device_str: Optional[str] = None
) -> Dict[str, float]:
    """
    Loads best checkpoint and computes publication metrics on Test set,
    and updates the .pkl artifact with final test metrics and parameters.
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n[Evaluation Engine] Loading checkpoint: {checkpoint_path} on {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "xlm-roberta-base")

    # Load test split
    _, _, test_loader, _, _, test_dataset = build_data_loaders(
        csv_path=dataset_path,
        batch_size=batch_size,
        tokenizer_name=model_name
    )

    # Initialize model
    model = NSSGDimNet(model_name=model_name, hidden_dim=768).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    param_stats = summarize_model_parameters(model)

    v_preds, v_trues = [], []
    a_preds, a_trues = [], []
    records = []

    for batch in test_loader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch_gpu)

        v_pred_b = outputs["valence_pred"].cpu().tolist()
        a_pred_b = outputs["arousal_pred"].cpu().tolist()
        v_true_b = batch_gpu["valence_target"].cpu().tolist()
        a_true_b = batch_gpu["arousal_target"].cpu().tolist()

        v_preds.extend(v_pred_b)
        v_trues.extend(v_true_b)
        a_preds.extend(a_pred_b)
        a_trues.extend(a_true_b)

        sentences = batch["sentence"]
        aspects = batch["aspect"]
        opinions = batch["opinion"]

        for i in range(len(v_pred_b)):
            records.append({
                "sentence": sentences[i],
                "aspect": aspects[i],
                "opinion": opinions[i],
                "valence_true": round(v_true_b[i], 4),
                "valence_pred": round(v_pred_b[i], 4),
                "valence_error": round(abs(v_true_b[i] - v_pred_b[i]), 4),
                "arousal_true": round(a_true_b[i], 4),
                "arousal_pred": round(a_pred_b[i], 4),
                "arousal_error": round(abs(a_true_b[i] - a_pred_b[i]), 4)
            })

    # Compute comprehensive evaluation metrics
    metrics = compute_metrics(v_trues, v_preds, a_trues, a_preds)

    # Save detailed predictions
    df_preds = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_preds.to_csv(output_csv, index=False)

    # Update or save full pickle package
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    final_payload = {
        "model_name": model_name,
        "model_state_dict": state_dict_cpu,
        "architecture_config": {
            "hidden_dim": model.hidden_dim,
            "max_switch_dist": 16,
            "num_heads": 8,
            "num_rgat_layers": 2,
            "dropout": 0.1
        },
        "lexicon_priors": test_dataset.lexicon_engine.priors if hasattr(test_dataset, "lexicon_engine") else {},
        "global_lang_map": test_dataset.global_lang_map if hasattr(test_dataset, "global_lang_map") else {},
        "parameter_summary": param_stats,
        "val_metrics": checkpoint.get("val_metrics", {}),
        "test_metrics": metrics,
        "sample_test_predictions": records[:10],
        "training_epoch": checkpoint.get("epoch", 1)
    }

    with open(output_pkl, "wb") as f_pkl:
        pickle.dump(final_payload, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)

    # Publication-Grade Formatted Output
    print("\n" + "="*85)
    print("                     NSSG-DimNet MODEL & EVALUATION SUMMARY")
    print("="*85)
    print(f" Backbone Architecture          : {model_name}")
    print(f" Total Model Parameters         : {param_stats['total_params']:,}")
    print(f" Trainable Parameters           : {param_stats['trainable_params']:,}")
    print(f" Total Test Instances Evaluated : {len(v_trues)}")
    print("-" * 85)
    print(f" {'REGRESSION METRIC':<35} | {'VALENCE (V)':<20} | {'AROUSAL (A)':<20}")
    print("-" * 85)
    print(f" {'Root Mean Squared Error (RMSE)':<35} | {metrics['rmse_v']:<20.4f} | {metrics['rmse_a']:<20.4f}")
    print(f" {'Mean Absolute Error (MAE)':<35} | {metrics['mae_v']:<20.4f} | {metrics['mae_a']:<20.4f}")
    print(f" {'Concordance Correlation (CCC)':<35} | {metrics['ccc_v']:<20.4f} | {metrics['ccc_a']:<20.4f}")
    print(f" {'Pearson Correlation (r)':<35} | {metrics['pearson_v']:<20.4f} | {metrics['pearson_a']:<20.4f}")
    print(f" {'Coefficient of Determ. (R^2)':<35} | {metrics['r2_v']:<20.4f} | {metrics['r2_a']:<20.4f}")
    print("-" * 85)
    print(f" {'OVERALL COMPOSITE SCORES':<35} | {'VALUE':<20}")
    print("-" * 85)
    print(f" {'Overall Average RMSE':<35} | {metrics['rmse_avg']:<20.4f}")
    print(f" {'Overall Average MAE':<35} | {metrics['mae_avg']:<20.4f}")
    print(f" {'Overall Average CCC':<35} | {metrics['ccc_avg']:<20.4f}")
    print(f" {'Overall Average Pearson (r)':<35} | {metrics['pearson_avg']:<20.4f}")
    print("-" * 85)
    print(f" {'ACCURACY & TOLERANCE METRICS':<35} | {'VALUE (%)':<20}")
    print("-" * 85)
    print(f" {'Polarity Classification Accuracy':<35} | {metrics['acc_polarity']:<20.2f}%")
    print(f" {'Arousal State Accuracy':<35} | {metrics['acc_arousal_state']:<20.2f}%")
    print(f" {'4-Quadrant Circumplex Accuracy':<35} | {metrics['acc_quadrant']:<20.2f}%")
    print(f" {'Valence within ±0.10 Tolerance':<35} | {metrics['acc_v_tol_10']:<20.2f}%")
    print(f" {'Arousal within ±0.10 Tolerance':<35} | {metrics['acc_a_tol_10']:<20.2f}%")
    print(f" {'Joint V&A within ±0.10 Tolerance':<35} | {metrics['acc_tol_10']:<20.2f}%")
    print(f" {'Joint V&A within ±0.15 Tolerance':<35} | {metrics['acc_tol_15']:<20.2f}%")
    print("="*85, flush=True)
    print(f" [OK] Predictions exported to : {output_csv}", flush=True)
    print(f" [OK] Model PKL saved to      : {output_pkl}\n", flush=True)

    return metrics


if __name__ == "__main__":
    if os.path.exists("checkpoints/best_nssg_dimnet.pt"):
        evaluate_test_set()
    else:
        print("[Notice] No checkpoint found. Run 'python train.py' or 'python run_pipeline.py' first.")
