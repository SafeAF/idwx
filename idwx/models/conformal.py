from __future__ import annotations

import numpy as np


def residual_quantile(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.2) -> float:
    residuals = np.abs(y_true - y_pred)
    if residuals.size == 0:
        return float("nan")
    q = np.quantile(residuals, 1.0 - alpha)
    return float(q)
