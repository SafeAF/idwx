from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from idwx.models.base import BaseModel
from idwx.models.conformal import (
    CALIBRATION_VERSION,
    InsufficientCalibrationError,
    calibration_rank,
    residual_quantile,
)

NON_FEATURE_COLS = {"station_id", "season_year", "target_doy"}


class RFConformalModel(BaseModel):
    name = "rf"

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        random_state: int = 1337,
        alpha: float = 0.2,
        calibration_fraction: float = 0.2,
        min_calibration_years: int = 5,
    ) -> None:
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("Calibration alpha must be finite and strictly between 0 and 1")
        if not np.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
            raise ValueError("Calibration fraction must be finite and strictly between 0 and 1")
        if isinstance(min_calibration_years, bool) or not isinstance(min_calibration_years, (int, np.integer)) or min_calibration_years < 1:
            raise ValueError("Minimum calibration years must be a positive integer")
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.min_calibration_years = int(min_calibration_years)
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: list[str] = []
        self.feature_medians: dict[str, float] = {}
        self.qhat: float = np.nan
        self.metadata: dict[str, object] = {}
        self._is_fitted = False

    def _prep_X(self, X: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        if fit:
            self.feature_cols = [c for c in X.columns if c not in NON_FEATURE_COLS]
        out = X[self.feature_cols].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        if fit:
            med = out.median(numeric_only=True)
            self.feature_medians = {k: float(v) for k, v in med.items() if pd.notna(v)}
        for c in out.columns:
            out[c] = out[c].fillna(self.feature_medians.get(c, 0.0))
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame | None, config: dict) -> RFConformalModel:
        # A failed refit must never leave the old model usable for prediction.
        self._is_fitted = False
        self.qhat = np.nan
        self.feature_cols = []
        self.feature_medians = {}
        self.metadata = {}
        frame = X.copy()
        if len(y) != len(frame):
            raise ValueError("RF training features and targets must have equal length")
        if not {"station_id", "season_year"}.issubset(frame.columns):
            raise ValueError("RF training requires station_id and season_year")
        if frame[["station_id", "season_year"]].isna().any().any():
            raise ValueError("RF station IDs and season years must be nonmissing")
        years = pd.to_numeric(frame.season_year, errors="raise")
        if not (np.isfinite(years) & years.eq(np.floor(years))).all():
            raise ValueError("RF season years must be finite integers")
        frame["season_year"] = years.astype(int)
        if frame.duplicated(["station_id", "season_year"]).any():
            raise ValueError("RF training requires unique station-years")
        frame["y"] = np.asarray(y, dtype=float)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame[frame["y"].notna()].copy()
        frame = frame.sort_values(["season_year", "station_id"])
        if frame.empty:
            raise ValueError("RF model received empty training targets")

        years = np.sort(frame.season_year.unique())
        n_calibration_years = min(len(years), max(self.min_calibration_years, int(np.ceil(len(years) * self.calibration_fraction))))
        first_calibration_year = int(years[-n_calibration_years])
        # Keep the full preceding calendar year out of both partitions. This
        # respects the opt-in January 1 protocol's label availability boundary.
        train = frame[frame.season_year <= first_calibration_year - 2]
        calib = frame[frame.season_year >= first_calibration_year]
        self.metadata = {
            "model_name": self.name, "calibration_version": CALIBRATION_VERSION,
            "calibration_method": "chronological_split_absolute_residual",
            "interval_semantics": "symmetric_absolute_residual_interval",
            "validation_status": "unvalidated_research", "fit_status": "unavailable",
            "alpha": self.alpha, "nominal_coverage": 1 - self.alpha,
            "calibration_fraction": self.calibration_fraction, "min_calibration_years": self.min_calibration_years,
            "calibration_embargo_years": 1,
            "n_total": len(frame), "n_train": len(train), "n_calib": len(calib),
            "n_embargo": len(frame) - len(train) - len(calib),
            "n_fitting_years": train.season_year.nunique(), "n_calibration_years": calib.season_year.nunique(),
            "fitting_start_year": int(train.season_year.min()) if len(train) else None,
            "fitting_end_year": int(train.season_year.max()) if len(train) else None,
            "calibration_start_year": int(calib.season_year.min()), "calibration_end_year": int(calib.season_year.max()),
            "random_state": self.random_state,
        }
        if n_calibration_years < self.min_calibration_years:
            raise InsufficientCalibrationError(f"RF requires at least {self.min_calibration_years} held-out calibration years")
        if train.season_year.nunique() < 8:
            raise ValueError("RF requires at least 8 fitting years after reserving calibration years and a one-calendar-year embargo")
        rank = calibration_rank(len(calib), self.alpha)

        Xtr = self._prep_X(train.drop(columns=["y"]), fit=True)
        ytr = train["y"]
        self.model.fit(Xtr, ytr)

        Xc = self._prep_X(calib.drop(columns=["y"]))
        yc = calib["y"].to_numpy(dtype=float)
        yhat = self.model.predict(Xc)
        self.qhat = residual_quantile(yc, yhat, alpha=self.alpha)
        self.metadata.update(fit_status="fitted", feature_count=len(self.feature_cols), calibration_rank=rank, qhat=self.qhat)
        self._is_fitted = True

        return self

    def predict(self, X_future: pd.DataFrame, config: dict) -> pd.DataFrame:
        self._require_fitted()
        if X_future.empty:
            return pd.DataFrame(columns=["p10", "p50", "p90"])
        point = self.model.predict(self._prep_X(X_future))
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = pd.DataFrame({"p10": point - self.qhat, "p50": point, "p90": point + self.qhat})
        if not np.isfinite(predictions.to_numpy()).all():
            raise ValueError("RF produced nonfinite interval bounds; no fallback interval is available")
        return predictions

    def _require_fitted(self) -> None:
        if (not getattr(self, "_is_fitted", False) or self.metadata.get("calibration_version") != CALIBRATION_VERSION
                or not np.isfinite(self.qhat) or self.qhat < 0):
            raise ValueError("RF has no compatible held-out calibration; retrain before prediction")

    def save(self, path: Path) -> None:
        self._require_fitted()
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.pkl")

    @classmethod
    def load(cls, path: Path) -> RFConformalModel:
        model = joblib.load(path / "model.pkl")
        if not isinstance(model, cls):
            raise ValueError("Artifact is not an RF model")  # noqa: TRY004 - Invalid file content, not an API argument type.
        model._require_fitted()
        return model
