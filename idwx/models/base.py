from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame | None, config: dict) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_future: pd.DataFrame, config: dict) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        raise NotImplementedError
