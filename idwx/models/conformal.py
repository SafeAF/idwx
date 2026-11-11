from __future__ import annotations

import numpy as np

CALIBRATION_VERSION = 2


class InsufficientCalibrationError(ValueError):
    """Held-out data cannot support the requested finite interval."""


def calibration_rank(n: int, alpha: float) -> int:
    """Finite-sample split-conformal order statistic; never cap an unsupported rank."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Calibration alpha must be finite and strictly between 0 and 1")
    if n < 1:
        raise InsufficientCalibrationError("Calibration requires nonempty held-out residuals")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        raise InsufficientCalibrationError(
            f"Calibration sample of {n} cannot support a finite interval at alpha={alpha}; required rank {rank} exceeds {n}"
        )
    return rank

def residual_quantile(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.2) -> float:
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("Calibration actuals and predictions must be equal-length one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Calibration actuals and predictions must be finite")
    rank = calibration_rank(actual.size, alpha)
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = np.abs(actual - predicted)
    if not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals overflowed; cannot construct a finite interval")
    return float(np.partition(residuals, rank - 1)[rank - 1])
