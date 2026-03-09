from idwx.models.base import BaseModel
from idwx.models.climatology import ClimatologyModel
from idwx.models.rf import RFConformalModel
from idwx.models.trend import TrendQuantileModel


def create_model(model_name: str, config: dict) -> BaseModel:
    if model_name == "climatology":
        return ClimatologyModel()
    if model_name == "trend":
        quantiles = config.get("trend", {}).get("quantiles", [0.1, 0.5, 0.9])
        return TrendQuantileModel(quantiles=quantiles)
    if model_name == "rf":
        rf_cfg = config.get("rf", {})
        alpha = float(config.get("conformal", {}).get("alpha", 0.2))
        return RFConformalModel(
            n_estimators=int(rf_cfg.get("n_estimators", 500)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
            max_features=rf_cfg.get("max_features", "sqrt"),
            random_state=int(rf_cfg.get("random_state", 1337)),
            alpha=alpha,
        )
    raise ValueError(f"Unsupported model: {model_name}")
