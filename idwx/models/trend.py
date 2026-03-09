from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from idwx.models.base import BaseModel


class TrendQuantileModel(BaseModel):
    name = "trend"

    def __init__(self, quantiles: list[float] | None = None) -> None:
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.models: dict[str, dict[float, QuantileRegressor]] = {}
        self.global_models: dict[float, QuantileRegressor] = {}
        self.fallback_quantiles: dict[str, float] = {"p10": np.nan, "p50": np.nan, "p90": np.nan}
        self.metadata: dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame | None, config: dict) -> "TrendQuantileModel":
        frame = X.copy()
        frame["y"] = y.values
        fitted_stations = 0
        for sid, g in frame.groupby("station_id"):
            gx = g[["year_index"]].copy()
            gy = g["y"]
            valid = gy.notna() & gx["year_index"].notna()
            gx = gx[valid]
            gy = gy[valid]
            if len(gx) < 6:
                continue

            station_models: dict[float, QuantileRegressor] = {}
            for q in self.quantiles:
                m = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
                m.fit(gx, gy)
                station_models[q] = m
            self.models[sid] = station_models
            fitted_stations += 1

        gvalid = frame["y"].notna() & frame["year_index"].notna()
        gx_all = frame.loc[gvalid, ["year_index"]]
        gy_all = frame.loc[gvalid, "y"]
        if len(gx_all) >= 6:
            for q in self.quantiles:
                gm = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
                gm.fit(gx_all, gy_all)
                self.global_models[q] = gm

        vals = gy_all.dropna()
        if not vals.empty:
            self.fallback_quantiles = {
                "p10": float(vals.quantile(0.10)),
                "p50": float(vals.quantile(0.50)),
                "p90": float(vals.quantile(0.90)),
            }
        self.metadata = {
            "model_name": self.name,
            "quantiles": self.quantiles,
            "fitted_station_count": fitted_stations,
            "global_model_available": bool(self.global_models),
        }

        return self

    def predict(self, X_future: pd.DataFrame, config: dict) -> pd.DataFrame:
        rows = []
        for _, r in X_future.iterrows():
            sid = str(r["station_id"])
            x = pd.DataFrame({"year_index": [r["year_index"]]})
            station_models = self.models.get(sid)
            if not station_models:
                if self.global_models:
                    preds = {q: float(m.predict(x)[0]) for q, m in self.global_models.items()}
                    rows.append({"p10": preds.get(0.1), "p50": preds.get(0.5), "p90": preds.get(0.9)})
                else:
                    rows.append(self.fallback_quantiles.copy())
                continue
            preds = {q: float(m.predict(x)[0]) for q, m in station_models.items()}
            rows.append({"p10": preds.get(0.1), "p50": preds.get(0.5), "p90": preds.get(0.9)})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.pkl")

    @classmethod
    def load(cls, path: Path) -> "TrendQuantileModel":
        return joblib.load(path / "model.pkl")
