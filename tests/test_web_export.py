import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from idwx.config import StationMeta, load_config
from idwx.web_export import demo_daily, export_web_data


@pytest.fixture
def source(tmp_path):
    cfg = replace(load_config("configs/idaho.yml"), cache_dir=tmp_path / "private",
                  backtest={"start_year": 2000, "end_year": 2022}, database_path=None)
    daily_dir = cfg.cache_dir / "daily"
    daily_dir.mkdir(parents=True)
    stations = [StationMeta("alpha", "Alpha", 42.0, -114.0),
                StationMeta("beta", "Beta", 47.0, -116.0, timezone="America/Los_Angeles")]
    for station in stations:
        demo_daily(station.station_id).to_parquet(daily_dir / f"{station.station_id}.parquet", index=False)
    return cfg, stations, tmp_path / "public"


def read_json(path):
    # Reject nonstandard NaN/Infinity even though Python's default parser permits them.
    def reject(value):
        raise ValueError(f"Invalid JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def test_release_contract_csv_consistency_and_privacy(source):
    cfg, stations, output = source
    manifest = export_web_data(cfg, stations, output)
    assert read_json(output / "data/manifest.json") == manifest
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["data_kind"] == "historical_observations"
    for name in ("index.html", "css/styles.css", "js/app.js", "js/charts.js", "js/format.js"):
        assert (output / name).read_bytes() == (Path("web") / name).read_bytes()
    assert not (output / "package.json").exists()
    assert manifest["source_coverage"] == {"first_date": "2000-01-01", "last_date": "2022-12-31"}
    index = read_json(output / manifest["stations_url"])
    assert index["release_id"] == manifest["release_id"]
    assert len(index["stations"]) == 2
    for url, digest in manifest["files"].items():
        content = (output / url).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest
        assert str(cfg.cache_dir).encode() not in content
        assert b"tmean_c" not in content  # No raw daily observations.
    documents = [read_json(output / entry["url"]) for entry in index["stations"]]
    assert documents[1]["station"]["timezone"] == "America/Los_Angeles"
    assert documents[0]["station"]["timezone"] == cfg.timezone
    for doc in documents:
        assert doc["release_id"] == manifest["release_id"]
        assert len(doc["annual"]) == 46
        assert doc["summaries"][0]["last_spring_frost"]["status"] == "available"
    partial = next(row for row in documents[0]["annual"] if row["season_year"] == 2021 and row["threshold_c"] == 0)
    assert partial["last_spring_frost"]["date"] is None
    assert partial["last_spring_frost"]["candidate_date"] == "2021-05-02"
    assert partial["last_spring_frost"]["missing_days"] == 1
    assert partial["interval"]["frost_free_days"] is None
    with (output / manifest["history_csv_url"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 92
    csv_partial = next(row for row in rows if row["station_id"] == "alpha" and row["season_year"] == "2021" and row["threshold_c"] == "0.0")
    assert csv_partial["last_spring_frost_date"] == ""
    assert csv_partial["last_spring_frost_candidate_date"] == partial["last_spring_frost"]["candidate_date"]
    assert csv_partial["release_id"] == manifest["release_id"]


def test_reproducible_immutable_releases_and_retention(source):
    cfg, stations, output = source
    first = export_web_data(cfg, stations, output)
    initial = {name: (output / name).read_bytes() for name in first["files"]}
    second = export_web_data(cfg, list(reversed(stations)), output)
    assert first["release_id"] == second["release_id"]
    assert all((output / name).read_bytes() == value for name, value in initial.items())
    third = export_web_data(cfg, stations, output, baseline=(2010, 2022))
    assert third["release_id"] != first["release_id"]
    assert all((output / name).read_bytes() == value for name, value in initial.items())


def test_missing_station_and_corrupted_release_leave_manifest_untouched(source):
    cfg, stations, output = source
    manifest = export_web_data(cfg, stations, output)
    old_manifest = (output / "data/manifest.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        export_web_data(cfg, [*stations, StationMeta("absent", "Absent")], output)
    assert (output / "data/manifest.json").read_bytes() == old_manifest
    (output / manifest["stations_url"]).write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable release"):
        export_web_data(cfg, stations, output)
    assert (output / "data/manifest.json").read_bytes() == old_manifest


def test_failed_manifest_switch_preserves_previous_release(source, monkeypatch):
    import os

    cfg, stations, output = source
    first = export_web_data(cfg, stations, output)
    old_manifest = (output / "data/manifest.json").read_bytes()
    original_replace = os.replace

    def fail_manifest(src, dst):
        if dst.name == "manifest.json":
            raise OSError("Simulated manifest replacement failure")
        original_replace(src, dst)

    monkeypatch.setattr("idwx.web_export.os.replace", fail_manifest)
    with pytest.raises(OSError, match="Simulated"):
        export_web_data(cfg, stations, output, min_years=3)
    assert (output / "data/manifest.json").read_bytes() == old_manifest
    assert all((output / name).is_file() for name in first["files"])
    assert not list((output / "data").glob(".*-*/"))


def test_demo_is_explicit_and_does_not_read_private_caches(source):
    cfg, _, output = source
    manifest = export_web_data(cfg, [StationMeta("demo", "Demo")], output, demo=True)
    assert manifest["data_kind"] == "synthetic_demo"
    assert read_json(output / manifest["stations_url"])["data_kind"] == "synthetic_demo"
    with (output / manifest["history_csv_url"]).open(newline="", encoding="utf-8") as stream:
        assert all(row["data_kind"] == "synthetic_demo" for row in csv.DictReader(stream))


@pytest.mark.parametrize("station_ids", [["../escape"], ["same", "same"]])
def test_unsafe_or_duplicate_station_ids_rejected(source, station_ids):
    cfg, _, output = source
    with pytest.raises(ValueError, match="snake_case"):
        export_web_data(cfg, [StationMeta(s, s) for s in station_ids], output, demo=True)
    assert not output.exists()


def test_nonfinite_metadata_is_null_and_nonfinite_threshold_is_rejected(source):
    cfg, _, output = source
    station = StationMeta("demo", "Demo", lat=float("nan"), elevation_m=float("inf"))
    manifest = export_web_data(cfg, [station], output, demo=True)
    assert read_json(output / manifest["stations_url"])["stations"][0]["station"]["lat"] is None
    old_manifest = (output / "data/manifest.json").read_bytes()
    cfg.frost_thresholds_c = [float("inf")]
    with pytest.raises(ValueError, match="finite Celsius"):
        export_web_data(cfg, [station], output, demo=True)
    assert (output / "data/manifest.json").read_bytes() == old_manifest


def test_ignores_old_target_caches(source):
    cfg, stations, output = source
    targets = cfg.cache_dir / "targets"
    targets.mkdir()
    pd.DataFrame({"corrupt": ["stale"]}).to_parquet(targets / "alpha.parquet")
    manifest = export_web_data(cfg, stations, output)
    assert manifest["release_id"]


def test_all_missing_minima_remain_null_and_insufficient(source):
    cfg, stations, output = source
    path = cfg.cache_dir / "daily/alpha.parquet"
    daily = pd.read_parquet(path)
    daily["tmin_c"] = float("nan")
    daily.to_parquet(path, index=False)
    manifest = export_web_data(cfg, stations, output)
    index = read_json(output / manifest["stations_url"])
    station = read_json(output / index["stations"][0]["url"])
    assert station["source_coverage"]["last_valid_minimum_date"] is None
    assert station["source_coverage"]["valid_minimum_days"] == 0
    assert all(row["last_spring_frost"]["status"] == "insufficient_data" for row in station["annual"])
    assert station["summaries"][0]["last_spring_frost"]["p50"] is None


def test_outside_baseline_and_extra_release_files_are_rejected(source):
    cfg, stations, output = source
    with pytest.raises(ValueError, match="within the exported"):
        export_web_data(cfg, stations, output, baseline=(1999, 2022))
    assert not output.exists()
    manifest = export_web_data(cfg, stations, output)
    old_manifest = (output / "data/manifest.json").read_bytes()
    (output / "data" / manifest["release_id"] / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable release"):
        export_web_data(cfg, stations, output)
    assert (output / "data/manifest.json").read_bytes() == old_manifest
