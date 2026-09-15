"""
Hybrid CCC and Smooth L1 Loss Optimization Objective
=====================================================
Optimizes Concordance Correlation Coefficient (CCC) simultaneously with
Smooth L1 Loss (or MSE/RMSE) across continuous Valence and Arousal dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def concordance_correlation_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Concordance Correlation Coefficient (CCC):
        CCC = (2 * cov(y_true, y_pred)) / (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))^2)
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    if len(y_pred) < 2:
        return torch.tensor(0.0, device=y_pred.device)

    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)

    var_pred = torch.var(y_pred, unbiased=False)
    var_true = torch.var(y_true, unbiased=False)

    cov = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

    numerator = 2.0 * cov
    denominator = var_pred + var_true + (mean_pred - mean_true) ** 2 + eps

    ccc = numerator / denominator
    ccc = torch.nan_to_num(ccc, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.clamp(ccc, -1.0, 1.0)


class HybridCCCSmoothL1Loss(nn.Module):
    """
    End-to-End Hybrid Loss Objective:
        L_total = lambda_ccc * ((1 - CCC_V) + (1 - CCC_A)) + lambda_mse * (MSE_V + MSE_A) + lambda_l1 * (SmoothL1_V + SmoothL1_A)
    """
    def __init__(
        self,
        lambda_ccc: float = 1.0,
        lambda_mse: float = 2.0,
        lambda_l1: float = 0.5,
        beta: float = 0.05
    ):
        super().__init__()
        self.lambda_ccc = lambda_ccc
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
        self.mse = nn.MSELoss()

    def forward(
        self,
        v_pred: torch.Tensor,
        v_true: torch.Tensor,
        a_pred: torch.Tensor,
        a_true: torch.Tensor
    ) -> torch.Tensor:
        # MSE and Smooth L1 Regressions (directly minimizes RMSE)
        mse_v = self.mse(v_pred, v_true)
        mse_a = self.mse(a_pred, a_true)
        mse_loss = mse_v + mse_a

        l1_v = self.smooth_l1(v_pred, v_true)
        l1_a = self.smooth_l1(a_pred, a_true)
        l1_loss = l1_v + l1_a

        # Concordance Correlation Losses
        if len(v_pred) > 2:
            ccc_v = concordance_correlation_coefficient(v_pred, v_true)
            ccc_a = concordance_correlation_coefficient(a_pred, a_true)
            ccc_loss = (1.0 - ccc_v) + (1.0 - ccc_a)
        else:
            ccc_loss = torch.tensor(0.0, device=v_pred.device)

        total_loss = self.lambda_ccc * ccc_loss + self.lambda_mse * mse_loss + self.lambda_l1 * l1_loss
        return total_loss


# Alias
HybridCCCMSELoss = HybridCCCSmoothL1Loss

