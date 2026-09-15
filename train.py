"""
NSSG-DimNet Training & Validation Pipeline
==========================================
Implements:
- End-to-End optimization using AdamW with linear warmup and cosine decay.
- Gradient clipping (norm 1.0).
- Early stopping based on validation Average RMSE / CCC.
- Model checkpoint saving as both PyTorch format (.pt) and standard Pickle format (.pkl).
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import math
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Tuple, Optional

from data_pipeline import build_data_loaders
from model import NSSGDimNet
from loss import HybridCCCSmoothL1Loss
from metrics import compute_metrics


def get_linear_warmup_cosine_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Creates a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device: torch.device,
    epoch: int = 1,
    clip_grad_norm: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    v_preds, v_trues, a_preds, a_trues = [], [], [], []
    total_batches = len(train_loader)

    for step, batch in enumerate(train_loader, 1):
        # Move tensors to device
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(batch_gpu)

        v_pred = outputs["valence_pred"]
        a_pred = outputs["arousal_pred"]
        v_true = batch_gpu["valence_target"]
        a_true = batch_gpu["arousal_target"]

        loss = criterion(v_pred, v_true, a_pred, a_true)
        loss.backward()

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        v_preds.extend(v_pred.detach().cpu().tolist())
        v_trues.extend(v_true.detach().cpu().tolist())
        a_preds.extend(a_pred.detach().cpu().tolist())
        a_trues.extend(a_true.detach().cpu().tolist())

        if step % 10 == 0 or step == total_batches:
            print(f"  [Epoch {epoch:02d}] Step {step:02d}/{total_batches:02d} | Batch Loss: {loss_val:.4f} | Running Avg Loss: {total_loss/step:.4f}", flush=True)

    avg_loss = total_loss / len(train_loader)
    train_metrics = compute_metrics(v_trues, v_preds, a_trues, a_preds)
    return avg_loss, train_metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader,
    criterion,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    v_preds, v_trues, a_preds, a_trues = [], [], [], []

    for batch in data_loader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch_gpu)

        v_pred = outputs["valence_pred"]
        a_pred = outputs["arousal_pred"]
        v_true = batch_gpu["valence_target"]
        a_true = batch_gpu["arousal_target"]

        loss = criterion(v_pred, v_true, a_pred, a_true)
        total_loss += loss.item()

        v_preds.extend(v_pred.cpu().tolist())
        v_trues.extend(v_true.cpu().tolist())
        a_preds.extend(a_pred.cpu().tolist())
        a_trues.extend(a_true.cpu().tolist())

    avg_loss = total_loss / max(1, len(data_loader))
    val_metrics = compute_metrics(v_trues, v_preds, a_trues, a_preds)
    return avg_loss, val_metrics


def run_training(
    dataset_path: str = "DimABSA_Final_Dataset_600.csv",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    patience: int = 5,
    model_name: str = "xlm-roberta-base",
    freeze_backbone: bool = True,
    unfreeze_layers: int = 2
):
    """Executes full training with early stopping and checkpointing (.pt and .pkl)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training Engine] Device: {device} | Model Backbone: {model_name} | Epochs: {epochs} | Batch Size: {batch_size} | Unfreeze Layers: {unfreeze_layers}", flush=True)

    # Build data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = build_data_loaders(
        csv_path=dataset_path,
        batch_size=batch_size,
        tokenizer_name=model_name
    )

    # Instantiate Model, Loss & Optimizer
    model = NSSGDimNet(
        model_name=model_name,
        hidden_dim=768,
        freeze_backbone=freeze_backbone,
        unfreeze_layers=unfreeze_layers,
        dropout=0.1
    ).to(device)

    # Hybrid CCC + MSE Loss for direct RMSE minimization
    criterion = HybridCCCSmoothL1Loss(lambda_ccc=1.0, lambda_mse=2.5, lambda_l1=0.5)

    # Differential learning rate
    backbone_params = []
    custom_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            backbone_params.append(param)
        else:
            custom_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": 2e-5, "weight_decay": weight_decay})
    if custom_params:
        param_groups.append({"params": custom_params, "lr": lr, "weight_decay": weight_decay})

    optimizer = AdamW(param_groups)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_pt_path = os.path.join(checkpoint_dir, "best_nssg_dimnet.pt")
    checkpoint_pkl_path = os.path.join(checkpoint_dir, "best_nssg_dimnet.pkl")

    print("\n" + "="*90, flush=True)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train RMSE':<10} | {'Val Loss':<10} | {'Val RMSE_V':<10} | {'Val RMSE_A':<10} | {'Val RMSE_Avg':<12} | {'Val CCC_Avg':<11}", flush=True)
    print("="*90, flush=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch=epoch
        )
        val_loss, val_metrics = evaluate_epoch(
            model, val_loader, criterion, device
        )

        val_rmse_avg = val_metrics["rmse_avg"]
        val_ccc_avg = val_metrics["ccc_avg"]

        print(
            f"{epoch:<6d} | {train_loss:<10.4f} | {train_metrics['rmse_avg']:<10.4f} | {val_loss:<10.4f} | "
            f"{val_metrics['rmse_v']:<10.4f} | {val_metrics['rmse_a']:<10.4f} | {val_rmse_avg:<12.4f} | {val_ccc_avg:<11.4f}",
            flush=True
        )

        # Early stopping & checkpointing
        if val_rmse_avg < best_val_rmse:
            best_val_rmse = val_rmse_avg
            best_epoch = epoch
            patience_counter = 0
            
            # Prepare state payload for Streamlit and PyTorch deployment
            state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            save_payload = {
                "epoch": epoch,
                "model_state_dict": state_dict_cpu,
                "val_metrics": val_metrics,
                "model_name": model_name,
                "architecture_config": {
                    "hidden_dim": model.hidden_dim,
                    "max_switch_dist": 16,
                    "num_heads": 8,
                    "num_rgat_layers": 2,
                    "dropout": 0.1,
                    "unfreeze_layers": unfreeze_layers
                },
                "lexicon_priors": train_dataset.lexicon_engine.priors,
                "global_lang_map": train_dataset.global_lang_map
            }

            # 1. Save PyTorch checkpoint (.pt)
            torch.save(save_payload, checkpoint_pt_path)

            # 2. Save Streamlit-Ready Pickle checkpoint (.pkl)
            with open(checkpoint_pkl_path, "wb") as f_pkl:
                pickle.dump(save_payload, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [OK Checkpoint Saved] Epoch {epoch:02d} -> {checkpoint_pkl_path} (Val RMSE: {val_rmse_avg:.4f})", flush=True)
                
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at epoch {epoch}. Best Val RMSE: {best_val_rmse:.4f} (Epoch {best_epoch})", flush=True)
                break

    # Evaluate best model on test set
    print("\n[Evaluation] Loading best model checkpoint to compute test set metrics...", flush=True)
    with open(checkpoint_pkl_path, "rb") as f:
        best_saved = pickle.load(f)
    model.load_state_dict(best_saved["model_state_dict"])
    _, test_metrics = evaluate_epoch(model, test_loader, criterion, device)

    # Compute parameter summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    param_summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params
    }

    best_saved["test_metrics"] = test_metrics
    best_saved["parameter_summary"] = param_summary
    with open(checkpoint_pkl_path, "wb") as f_pkl:
        pickle.dump(best_saved, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(best_saved, checkpoint_pt_path)

    total_duration = time.time() - start_time
    print("="*90, flush=True)
    print(f"[Training Complete] Total Time: {total_duration:.1f}s | Best Checkpoints: {checkpoint_pt_path} & {checkpoint_pkl_path}", flush=True)
    print(f"[Final Test Metrics] RMSE_V: {test_metrics['rmse_v']:.4f} | RMSE_A: {test_metrics['rmse_a']:.4f} | Overall RMSE: {test_metrics['rmse_avg']:.4f} | Polarity Acc: {test_metrics['acc_polarity']:.2f}% | 4-Quadrant Acc: {test_metrics['acc_quadrant']:.2f}%", flush=True)
    return checkpoint_pt_path


if __name__ == "__main__":
    run_training(epochs=8, batch_size=16, unfreeze_layers=2)

