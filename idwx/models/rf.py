from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from idwx.models.base import BaseModel
from idwx.models.conformal import residual_quantile


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
    ) -> None:
        self.alpha = alpha
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

    def _prep_X(self, X: pd.DataFrame) -> pd.DataFrame:
        use = [c for c in X.columns if c not in NON_FEATURE_COLS]
        self.feature_cols = use if not self.feature_cols else self.feature_cols
        out = X[self.feature_cols].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        if not self.feature_medians:
            med = out.median(numeric_only=True)
            self.feature_medians = {k: float(v) for k, v in med.items() if pd.notna(v)}
        for c in out.columns:
            out[c] = out[c].fillna(self.feature_medians.get(c, 0.0))
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame | None, config: dict) -> "RFConformalModel":
        frame = X.copy()
        frame["y"] = y.values
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame[frame["y"].notna()].copy()
        frame = frame.sort_values("season_year")
        if frame.empty:
            raise ValueError("RF model received empty training targets")

        n = len(frame)
        split = int(max(min(n - 1, np.floor(n * 0.8)), 1))
        train = frame.iloc[:split]
        calib = frame.iloc[split:]
        if train["y"].notna().sum() < 8:
            raise ValueError("RF model requires at least 8 non-null training rows")

        Xtr = self._prep_X(train.drop(columns=["y"]))
        ytr = train["y"]
        self.model.fit(Xtr, ytr)

        qhat = np.nan
        if not calib.empty and calib["y"].notna().sum() > 0:
            Xc = self._prep_X(calib.drop(columns=["y"]))
            yc = calib["y"].to_numpy(dtype=float)
            yhat = self.model.predict(Xc)
            qhat = residual_quantile(yc, yhat, alpha=self.alpha)
        if not np.isfinite(qhat):
            yhat_tr = self.model.predict(Xtr)
            qhat = residual_quantile(ytr.to_numpy(dtype=float), yhat_tr, alpha=self.alpha)
        self.qhat = float(qhat)
        self.metadata = {
            "model_name": self.name,
            "alpha": self.alpha,
            "feature_count": len(self.feature_cols),
            "n_total": int(n),
            "n_train": int(len(train)),
            "n_calib": int(len(calib)),
            "qhat": self.qhat,
            "random_state": self.random_state,
        }

        return self

    def predict(self, X_future: pd.DataFrame, config: dict) -> pd.DataFrame:
        rows = []
        for _, r in X_future.iterrows():
            x = self._prep_X(pd.DataFrame([r]))
            p50 = float(self.model.predict(x)[0])
            q = self.qhat if np.isfinite(self.qhat) else 10.0
            p10 = p50 - q
            p90 = p50 + q
            rows.append({"p10": float(min(p10, p50)), "p50": p50, "p90": float(max(p50, p90))})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.pkl")

    @classmethod
    def load(cls, path: Path) -> "RFConformalModel":
        return joblib.load(path / "model.pkl")
