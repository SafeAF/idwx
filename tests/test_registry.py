import json
from dataclasses import replace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from idwx.cli import app
from idwx.config import load_config
from idwx.models.climatology import ClimatologyModel
from idwx.registry import (
    hash_config,
    latest_model_dir,
    model_run_dir,
    save_model_artifacts,
    threshold_key,
)


@pytest.fixture
def cfg(tmp_path):
    stations = tmp_path / "stations.yml"
    stations.write_text("stations: [{station_id: demo, timezone: America/Boise}]", encoding="utf-8")
    return replace(load_config("configs/idaho.yml"), models_dir=tmp_path / "models", stations_file=stations)


def dataset():
    return pd.DataFrame({"station_id": "demo", "season_year": range(2000, 2006),
                         "year_index": range(2000, 2006), "target_doy": range(100, 106)})


def save(cfg, threshold, run_id=None):
    data = dataset()
    model = ClimatologyModel().fit(data.drop(columns="target_doy"), data.target_doy, None, {})
    return save_model_artifacts(model, cfg, "last_spring_frost", "demo", "climatology", data,
                                threshold, run_id=run_id)


def lookup(cfg, threshold, data=None):
    return latest_model_dir(cfg, "last_spring_frost", "demo", "climatology", threshold_c=threshold, dataset=data)


def test_thresholds_are_isolated_and_metadata_is_checked(cfg):
    zero = save(cfg, 0., "first")
    secondary = save(cfg, -2., "second")
    assert zero.parent != secondary.parent
    assert lookup(cfg, 0., dataset()) == zero
    assert lookup(cfg, -2., dataset()) == secondary
    with pytest.raises(FileNotFoundError, match="threshold-specific"):
        lookup(cfg, 1.)
    metadata = yaml.safe_load((zero / "config.yml").read_text(encoding="utf-8"))
    assert metadata["threshold_c"] == 0. and metadata["artifact_version"] == 2
    assert metadata["validation_status"] == "unvalidated_research"
    metadata["threshold_c"] = -2.
    (zero / "config.yml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        lookup(cfg, 0.)


def test_legacy_and_incomplete_runs_are_never_selected(cfg):
    legacy = cfg.models_dir / "last_spring_frost/demo/climatology/legacy"
    legacy.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Retrain legacy"):
        lookup(cfg, 0.)
    complete = save(cfg, 0., "complete")
    incomplete = model_run_dir(cfg, "last_spring_frost", "demo", "climatology", threshold_c=0., run_id="zzz")
    incomplete.mkdir()
    assert lookup(cfg, 0.) == complete


def test_new_runs_do_not_collide_and_explicit_ids_cannot_overwrite(cfg):
    first, second = save(cfg, 0.), save(cfg, 0.)
    assert first != second and lookup(cfg, 0.) == second
    chosen = save(cfg, 0., "chosen")
    before = (chosen / "model.pkl").read_bytes()
    with pytest.raises(FileExistsError):
        save(cfg, 0., "chosen")
    assert (chosen / "model.pkl").read_bytes() == before


def test_lookup_rejects_changed_configuration_and_dataset(cfg):
    save(cfg, 0.)
    overrides = replace(cfg, station_windows={"demo": {"last_spring_frost": {"start": "01-01", "end": "07-31"}}})
    assert hash_config(cfg) != hash_config(overrides)
    with pytest.raises(ValueError, match="Incompatible"):
        lookup(overrides, 0.)
    changed = dataset()
    changed.loc[0, "target_doy"] = 999
    with pytest.raises(ValueError, match="Incompatible"):
        lookup(cfg, 0., changed)
    with pytest.raises(ValueError, match="Incompatible"):
        lookup(cfg, 0., dataset().assign(new_feature=0))
    cfg.stations_file.write_text("stations: [{station_id: demo, timezone: America/Los_Angeles}]", encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        lookup(cfg, 0.)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_thresholds_fail(threshold):
    with pytest.raises(ValueError, match="finite"):
        threshold_key(threshold)


def test_threshold_keys_preserve_exact_values_and_normalize_negative_zero():
    assert threshold_key(0.) == threshold_key(-0.)
    assert threshold_key(None) != threshold_key(0.)
    assert threshold_key(-2.) != threshold_key(-2.000000000000001)


def test_predict_rejects_legacy_artifact_before_loading_pickle(cfg, monkeypatch):
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    monkeypatch.setattr("idwx.cli.load_or_build_dataset", lambda *args, **kwargs: dataset())
    result = CliRunner().invoke(app, ["predict", "--target", "last_spring_frost", "--station", "demo",
                                     "--model", "climatology"])
    assert result.exit_code != 0 and "No threshold-specific model directory" in result.output


def test_train_and_predict_use_the_requested_threshold(cfg, monkeypatch):
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    monkeypatch.setattr("idwx.cli.load_or_build_dataset", lambda *args, **kwargs: dataset())
    runner = CliRunner()
    trained = runner.invoke(app, ["train", "--target", "last_spring_frost", "--models", "climatology",
                                  "--threshold", "-2"])
    assert trained.exit_code == 0, trained.output
    assert json.loads(trained.output)["count"] == 1
    args = ["predict", "--target", "last_spring_frost", "--station", "demo", "--model", "climatology"]
    predicted = runner.invoke(app, [*args, "--threshold", "-2"])
    assert predicted.exit_code == 0, predicted.output
    result = json.loads(predicted.output)
    assert result["threshold_c"] == -2. and result["validation_status"] == "unvalidated_research"
    assert result["prediction"]["p50"] == 102.5
    wrong_threshold = runner.invoke(app, args)
    assert wrong_threshold.exit_code != 0 and "No threshold-specific model directory" in wrong_threshold.output


def test_unreadable_latest_metadata_fails_without_selecting_an_older_run(cfg):
    save(cfg, 0., "first")
    latest = save(cfg, 0., "second")
    (latest / "config.yml").write_text("invalid: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreadable model metadata"):
        lookup(cfg, 0.)
