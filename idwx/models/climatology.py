from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from idwx.models.base import BaseModel


class ClimatologyModel(BaseModel):
    name = "climatology"

    def __init__(self) -> None:
        self.by_station: dict[str, dict[str, float]] = {}
        self.global_stats: dict[str, float] = {"p10": np.nan, "p50": np.nan, "p90": np.nan}
        self.metadata: dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame | None, config: dict) -> "ClimatologyModel":
        frame = X.copy()
        frame["y"] = y.values
        for sid, g in frame.groupby("station_id"):
            vals = g["y"].dropna()
            if vals.empty:
                continue
            self.by_station[sid] = {
                "p10": float(vals.quantile(0.10)),
                "p50": float(vals.quantile(0.50)),
                "p90": float(vals.quantile(0.90)),
            }

        vals = y.dropna()
        if not vals.empty:
            self.global_stats = {
                "p10": float(vals.quantile(0.10)),
                "p50": float(vals.quantile(0.50)),
                "p90": float(vals.quantile(0.90)),
            }
        self.metadata = {
            "model_name": self.name,
            "station_count": len(self.by_station),
            "global_n_obs": int(vals.notna().sum()),
        }
        return self

    def predict(self, X_future: pd.DataFrame, config: dict) -> pd.DataFrame:
        rows = []
        for _, r in X_future.iterrows():
            sid = str(r.get("station_id", ""))
            stats = self.by_station.get(sid, self.global_stats)
            rows.append({"p10": stats["p10"], "p50": stats["p50"], "p90": stats["p90"]})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.pkl")

    @classmethod
    def load(cls, path: Path) -> "ClimatologyModel":
        return joblib.load(path / "model.pkl")
