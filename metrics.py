"""
Evaluation Metrics for Dimensional ABSA (NSSG-DimNet)
=====================================================
Calculates:
- Root Mean Squared Error (RMSE_V, RMSE_A, RMSE_avg)
- Concordance Correlation Coefficient (CCC_V, CCC_A, CCC_avg)
- Pearson Correlation Coefficient (r_V, r_A, r_avg)
- Mean Absolute Error (MAE_V, MAE_A, MAE_avg)
- Coefficient of Determination (R2_V, R2_A)
- Categorical & Quadrant Accuracies:
  * Polarity Accuracy (Valence > 0.5 vs <= 0.5)
  * Arousal State Accuracy (Arousal > 0.5 vs <= 0.5)
  * 4-Quadrant Circumplex Accuracy (Q1: HV-HA, Q2: LV-HA, Q3: LV-LA, Q4: HV-LA)
  * Continuous Tolerance Accuracies (|pred - true| <= threshold) for epsilon = 0.05, 0.10, 0.15
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Concordance Correlation Coefficient (CCC) for numpy arrays."""
    if len(y_true) < 2:
        return 0.0
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(2.0 * cov / denominator, -1.0, 1.0))


def compute_metrics(
    v_true_list: List[float],
    v_pred_list: List[float],
    a_true_list: List[float],
    a_pred_list: List[float]
) -> Dict[str, float]:
    """Computes comprehensive regression, correlation, and accuracy metrics."""
    y_v_true = np.array(v_true_list, dtype=np.float64)
    y_v_pred = np.array(v_pred_list, dtype=np.float64)
    y_a_true = np.array(a_true_list, dtype=np.float64)
    y_a_pred = np.array(a_pred_list, dtype=np.float64)

    # 1. Root Mean Squared Error (RMSE)
    rmse_v = float(np.sqrt(np.mean((y_v_true - y_v_pred) ** 2)))
    rmse_a = float(np.sqrt(np.mean((y_a_true - y_a_pred) ** 2)))
    rmse_avg = float((rmse_v + rmse_a) / 2.0)

    # 2. Mean Absolute Error (MAE)
    mae_v = float(np.mean(np.abs(y_v_true - y_v_pred)))
    mae_a = float(np.mean(np.abs(y_a_true - y_a_pred)))
    mae_avg = float((mae_v + mae_a) / 2.0)

    # 3. Concordance Correlation Coefficient (CCC)
    ccc_v = compute_ccc(y_v_true, y_v_pred)
    ccc_a = compute_ccc(y_a_true, y_a_pred)
    ccc_avg = float((ccc_v + ccc_a) / 2.0)

    # 4. Pearson Correlation (r)
    try:
        r_v, _ = pearsonr(y_v_true, y_v_pred)
        r_v = float(r_v) if not np.isnan(r_v) else 0.0
    except Exception:
        r_v = 0.0

    try:
        r_a, _ = pearsonr(y_a_true, y_a_pred)
        r_a = float(r_a) if not np.isnan(r_a) else 0.0
    except Exception:
        r_a = 0.0
    r_avg = float((r_v + r_a) / 2.0)

    # 5. R^2 Score
    ss_tot_v = np.sum((y_v_true - np.mean(y_v_true)) ** 2)
    ss_res_v = np.sum((y_v_true - y_v_pred) ** 2)
    r2_v = float(1.0 - ss_res_v / (ss_tot_v + 1e-12))

    ss_tot_a = np.sum((y_a_true - np.mean(y_a_true)) ** 2)
    ss_res_a = np.sum((y_a_true - y_a_pred) ** 2)
    r2_a = float(1.0 - ss_res_a / (ss_tot_a + 1e-12))

    # 6. Polarity & Arousal Binary Accuracies (Threshold = 0.50)
    acc_polarity = float(np.mean((y_v_true >= 0.5) == (y_v_pred >= 0.5))) * 100.0
    acc_arousal_state = float(np.mean((y_a_true >= 0.5) == (y_a_pred >= 0.5))) * 100.0

    # 7. 4-Quadrant Circumplex Classification Accuracy
    # Q1: High V, High A | Q2: Low V, High A | Q3: Low V, Low A | Q4: High V, Low A
    quad_true = (y_v_true >= 0.5).astype(int) * 2 + (y_a_true >= 0.5).astype(int)
    quad_pred = (y_v_pred >= 0.5).astype(int) * 2 + (y_a_pred >= 0.5).astype(int)
    acc_quadrant = float(np.mean(quad_true == quad_pred)) * 100.0

    # 8. Tolerance / Proximity Accuracies (prediction error <= epsilon)
    v_diff = np.abs(y_v_true - y_v_pred)
    a_diff = np.abs(y_a_true - y_a_pred)
    
    # Both V and A within tolerance
    acc_tol_05 = float(np.mean((v_diff <= 0.05) & (a_diff <= 0.05))) * 100.0
    acc_tol_10 = float(np.mean((v_diff <= 0.10) & (a_diff <= 0.10))) * 100.0
    acc_tol_15 = float(np.mean((v_diff <= 0.15) & (a_diff <= 0.15))) * 100.0

    # Individual V & A tolerance
    acc_v_tol_10 = float(np.mean(v_diff <= 0.10)) * 100.0
    acc_a_tol_10 = float(np.mean(a_diff <= 0.10)) * 100.0

    return {
        "rmse_v": rmse_v,
        "rmse_a": rmse_a,
        "rmse_avg": rmse_avg,
        "mae_v": mae_v,
        "mae_a": mae_a,
        "mae_avg": mae_avg,
        "ccc_v": ccc_v,
        "ccc_a": ccc_a,
        "ccc_avg": ccc_avg,
        "pearson_v": r_v,
        "pearson_a": r_a,
        "pearson_avg": r_avg,
        "r2_v": r2_v,
        "r2_a": r2_a,
        "acc_polarity": acc_polarity,
        "acc_arousal_state": acc_arousal_state,
        "acc_quadrant": acc_quadrant,
        "acc_tol_05": acc_tol_05,
        "acc_tol_10": acc_tol_10,
        "acc_tol_15": acc_tol_15,
        "acc_v_tol_10": acc_v_tol_10,
        "acc_a_tol_10": acc_a_tol_10
    }
