import hashlib
import json
import tempfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from idwx.cli import app
from idwx.config import StationMeta, load_config
from idwx.data_audit import audit_data, write_audit_report
from idwx.io import build_data_cache, ingest_era5_hourly_csv


@pytest.fixture
def station():
    return StationMeta("twin_falls", "Synthetic fixture", 42.5629, -114.4609, timezone="UTC")


@pytest.fixture
def cfg(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    return replace(load_config("configs/idaho.yml"), data_root=raw, cache_dir=tmp_path / "cache",
                   reports_dir=tmp_path / "reports", timezone="UTC",
                   backtest={"start_year": 2019, "end_year": 2020})


def source(cfg, timestamps=None, *, name="42.5629---114.4609-source.csv", temperatures=5., offsets=0.):
    if timestamps is None:
        timestamps = pd.date_range("2020-01-01T00:00:00Z", periods=24, freq="h")
    path = cfg.data_root / name
    pd.DataFrame({"timestamp": timestamps, "temperature (degC)": temperatures,
                  "utc_offset (hrs)": offsets}).to_csv(path, index=False)
    return path


def codes(report):
    return {issue["code"] for issue in report["blocking_issues"]}


def test_missing_or_empty_root_produces_blocked_private_report(cfg, station, tmp_path):
    for root in (cfg.data_root, cfg.data_root / "absent"):
        current = replace(cfg, data_root=root)
        report = audit_data(current, [station])
        assert report["audit_status"] == "blocked" and report["publication_ready"] is False
        assert report["files"] == [] and "missing_station_source" in codes(report)
        output = tmp_path / "reports/audit.json"
        write_audit_report(current, report, output)
        assert json.loads(output.read_text(encoding="utf-8"))["private_report"] is True
        with pytest.raises(FileNotFoundError):
            build_data_cache(current, [station])
        assert not cfg.cache_dir.exists()


def test_audit_hashes_sources_recomputes_history_and_never_certifies_claims(cfg, station):
    times = pd.date_range("2020-01-01T00:00:00Z", "2020-12-31T23:00:00Z", freq="h")
    temperatures = np.full(len(times), 5.)
    temperatures[np.isin(times.strftime("%m-%d"), ["04-30", "07-04", "09-15"])] = -3.
    path = source(cfg, times, temperatures=temperatures)
    before = path.read_bytes()
    report = audit_data(cfg, [station], chunksize=333)
    assert report["audit_status"] == "review_required" and not report["publication_ready"]
    assert set(report["manual_review_required"]) >= {"source_provenance", "temperature_units", "reporting_day", "data_vintage"}
    entry = report["files"][0]
    assert entry["sha256"] == hashlib.sha256(before).hexdigest()
    assert entry["diagnostics"]["complete"] and entry["diagnostics"]["rows"] == 8784
    assert entry["diagnostics"]["temperature_min_c"] == -3.
    assert entry["historical_review"]["source_coverage"]["recorded_days"] == 366
    records = entry["historical_review"]["annual"]
    selected = next(row for row in records if row["season_year"] == 2020 and row["threshold_c"] == 0.)
    assert selected["last_spring_frost"]["date"] == "2020-04-30"
    assert selected["first_fall_frost"]["date"] == "2020-09-15"
    assert selected["interval"]["status"] == "intervening_freeze"
    assert entry["review_summary"]["entirely_missing_years"] == [2019]
    assert entry["review_summary"]["missing_daily_minimum_ranges"] == [
        {"start_date": "2019-01-01", "end_date": "2019-12-31", "days": 365}]
    assert len(entry["review_summary"]["summer_freeze_records"]) == 2
    assert entry["daily_cache_comparison"]["status"] == "missing"
    assert path.read_bytes() == before and not cfg.cache_dir.exists()
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("corrupt", [False, True])
def test_cache_comparison_detects_stale_values_without_modifying_them(cfg, station, corrupt):
    source(cfg)
    assert build_data_cache(cfg, [station], rebuild=True) == [station.station_id]
    assert audit_data(cfg, [station])["files"][0]["daily_cache_comparison"]["status"] == "matches_current_aggregation"
    cache = cfg.cache_dir / "daily/twin_falls.parquet"
    if corrupt:
        cache.write_bytes(b"not a parquet file")
    else:
        daily = pd.read_parquet(cache)
        daily["tmin_c"] = -99.
        daily.to_parquet(cache, index=False)
    old = cache.read_bytes()
    report = audit_data(cfg, [station])
    assert "daily_cache_mismatch" in codes(report) and report["audit_status"] == "blocked"
    assert cache.read_bytes() == old
    assert build_data_cache(cfg, [station], rebuild=False) == []
    assert cache.read_bytes() == old
    build_data_cache(cfg, [station], rebuild=True)
    assert audit_data(cfg, [station])["audit_status"] == "review_required"


@pytest.mark.parametrize("chunksize", [1, 100])
def test_duplicate_timestamps_block_audit_with_partial_diagnostics(cfg, station, chunksize):
    source(cfg, ["2020-01-01T12:00:00Z"] * 2)
    report = audit_data(cfg, [station], chunksize=chunksize)
    assert report["audit_status"] == "blocked" and "source_processing_error" in codes(report)
    stats = report["files"][0]["diagnostics"]
    assert not stats["complete"] and stats["duplicate_timestamp_rows"] == 1
    assert report["files"][0]["historical_review"] is None


@pytest.mark.parametrize("timestamps, temperatures, offsets, code", [
    (["invalid"], [5.], [0.], "source_processing_error"),
    (["2020-01-01T00:00:00Z"], [5.], ["bad"], "source_processing_error"),
    (["2020-01-01T00:00:00Z"], [5.], [24.], "source_processing_error"),
    (["2020-01-01T00:00:00Z"], [5.], [float("inf")], "source_processing_error"),
    (["2020-01-01T00:00:00Z"], ["bad"], [0.], "invalid_temperature_values"),
    (["2020-01-01T00:00:00Z"], [float("inf")], [0.], "invalid_temperature_values"),
    (["2020-01-01", "2020-01-02"], [5., 5.], [0., 0.], "unsupported_daily_timestamps"),
])
def test_unusable_source_values_are_explicit(cfg, station, timestamps, temperatures, offsets, code):
    source(cfg, timestamps, temperatures=temperatures, offsets=offsets)
    report = audit_data(cfg, [station])
    assert report["audit_status"] == "blocked" and code in codes(report)
    if offsets[0] in ("bad", 24., float("inf")):
        assert report["files"][0]["diagnostics"]["invalid_offset_rows"] == 1
    json.dumps(report, allow_nan=False)


def test_missing_values_are_distinguished_from_invalid_temperature_values(cfg, station):
    source(cfg, temperatures=[None] + [5.] * 23, offsets=[None] + [0.] * 23)
    report = audit_data(cfg, [station])
    assert report["audit_status"] == "review_required"
    stats = report["files"][0]["diagnostics"]
    assert stats["missing_temperature_rows"] == 1 and stats["invalid_temperature_rows"] == 0
    assert stats["fallback_timezone_rows"] == 1 and stats["explicit_offset_rows"] == 23


@pytest.mark.parametrize("header", ["timestamp,temperature (degF)\n", "timestamp,temperature (degC),temperature (degC)\n", ""])
def test_unsupported_duplicate_or_empty_headers_block(cfg, station, header):
    path = cfg.data_root / "42.5629---114.4609-source.csv"
    path.write_text(header, encoding="utf-8")
    report = audit_data(cfg, [station])
    assert "source_processing_error" in codes(report)
    assert not report["files"][0]["diagnostics"]["complete"]


def test_unconfigured_or_ambiguous_station_mapping_is_not_silently_accepted(cfg, station):
    source(cfg)
    unmatched = replace(station, lat=43.)
    report = audit_data(cfg, [unmatched])
    assert {"missing_station_source", "unconfigured_source"} <= codes(report)
    nearby = replace(station, station_id="nearby", lat=station.lat + .00001)
    report = audit_data(cfg, [station, nearby])
    assert "station_mapping_error" in codes(report)
    with pytest.raises(ValueError, match="multiple configured stations"):
        build_data_cache(cfg, [station, nearby], rebuild=True)


@pytest.mark.parametrize("rebuild", [False, True])
def test_multiple_station_sources_fail_before_existing_cache_can_be_overwritten(cfg, station, rebuild):
    source(cfg, name="42.5629---114.4609-first.csv")
    source(cfg, name="42.5629---114.4609-second.csv")
    cache = cfg.cache_dir / "daily/twin_falls.parquet"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"previous cache")
    report = audit_data(cfg, [station])
    assert "multiple_station_sources" in codes(report)
    assert all(entry["historical_review"] is None for entry in report["files"])
    assert all(entry["sha256"] for entry in report["files"])
    with pytest.raises(ValueError, match="Multiple CSV files"):
        build_data_cache(cfg, [station], rebuild=rebuild)
    assert cache.read_bytes() == b"previous cache"


@pytest.mark.parametrize("failure", ["parse", "serialization"])
def test_failed_cache_build_preserves_all_old_station_files(cfg, station, monkeypatch, failure):
    source(cfg)
    second = replace(station, station_id="second", lat=43.)
    source(cfg, ["invalid"] if failure == "parse" else None, name="43.0---114.4609-second.csv")
    daily_dir = cfg.cache_dir / "daily"
    daily_dir.mkdir(parents=True)
    for sid in (station.station_id, second.station_id):
        (daily_dir / f"{sid}.parquet").write_bytes(sid.encode())
    if failure == "serialization":
        original = pd.DataFrame.to_parquet
        calls = []

        def fail_second(frame, *args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise ValueError("Serialization failed")
            return original(frame, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_second)
    with pytest.raises(ValueError):
        build_data_cache(cfg, [station, second], rebuild=True)
    for sid in (station.station_id, second.station_id):
        assert (daily_dir / f"{sid}.parquet").read_bytes() == sid.encode()
    assert not list(daily_dir.glob(".staging-*"))


def test_timestamp_order_timezone_and_offset_diagnostics_are_chunk_independent(cfg, station):
    cfg = replace(cfg, hourly_min_coverage=1)
    station = replace(station, timezone="America/Los_Angeles")
    source(cfg, ["2020-01-01T08:30:00Z", "2020-01-01T07:30:00Z"], offsets=[None, -7.])
    report = audit_data(cfg, [station], chunksize=1)
    stats = report["files"][0]["diagnostics"]
    assert stats["timestamp_order_violations"] == 1
    assert stats["source_offset_min"] == stats["source_offset_max"] == -7.
    assert stats["explicit_offset_rows"] == stats["fallback_timezone_rows"] == 1
    assert report["configured_stations"][0]["effective_timezone"] == "America/Los_Angeles"


def test_nonfinite_metadata_is_blocked_and_serialized_as_null(cfg, station):
    source(cfg)
    report = audit_data(cfg, [replace(station, elevation_m=float("inf"))])
    assert "invalid_station_metadata" in codes(report)
    assert report["configured_stations"][0]["elevation_m"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["write", "replace", "serialize"])
def test_failed_report_write_keeps_previous_report_and_cleans_temporary_files(cfg, tmp_path, monkeypatch, failure):
    output = tmp_path / "reports/audit.json"
    output.parent.mkdir()
    output.write_bytes(b"previous report")
    report = {"value": float("nan")} if failure == "serialize" else {"value": 1}
    if failure == "replace":
        def fail_replace(*args):
            raise OSError("Switch failed")

        monkeypatch.setattr("idwx.data_audit.os.replace", fail_replace)
    if failure == "write":
        original = tempfile.NamedTemporaryFile

        class FailWrite:
            def __init__(self, *args, **kwargs):
                self.file = original(*args, **kwargs)

            def __enter__(self):
                self.file.__enter__()
                self.name = self.file.name
                return self

            def write(self, content):
                self.file.write("partial")
                raise OSError("Write failed")

            def __exit__(self, *args):
                return self.file.__exit__(*args)

        monkeypatch.setattr("idwx.data_audit.tempfile.NamedTemporaryFile", FailWrite)
    with pytest.raises((ValueError, OSError)):
        write_audit_report(cfg, report, output)
    assert output.read_bytes() == b"previous report"
    assert not list(output.parent.glob(".audit-*"))


def test_private_report_cannot_be_written_to_source_or_publication_directories(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    publication = tmp_path / "export"
    (publication / "data").mkdir(parents=True)
    (publication / "data/manifest.json").write_text("{}", encoding="utf-8")
    for destination in (cfg.data_root / "audit.json", tmp_path / "web/audit.json", publication / "audit.json"):
        with pytest.raises(ValueError, match="outside"):
            write_audit_report(cfg, {}, destination)
        assert not destination.exists()


@pytest.mark.parametrize("explicit_output", [False, True])
def test_cli_writes_blocked_report_with_nonzero_exit_and_never_builds_caches(cfg, station, monkeypatch, explicit_output):
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    monkeypatch.setattr("idwx.cli.load_stations", lambda *args: [station])
    output = cfg.reports_dir / ("audit.json" if explicit_output else "data-audit.json")
    result = CliRunner().invoke(app, ["data", "audit", *(["--output", str(output)] if explicit_output else [])])
    assert result.exit_code == 1
    assert json.loads(result.output)["audit_status"] == "blocked"
    assert output.is_file() and not cfg.cache_dir.exists()


def test_ingestion_rejects_invalid_times_and_offsets_without_audit_mode(cfg, station):
    for times, offsets in [(["invalid"], [0.]), (["2020-01-01T00:00:00Z"], ["bad"])]:
        path = source(cfg, times, offsets=offsets)
        with pytest.raises(ValueError):
            ingest_era5_hourly_csv(path, cfg, [station])


def test_source_changed_during_audit_cannot_supply_historical_review(cfg, station, monkeypatch):
    path = source(cfg)
    original = ingest_era5_hourly_csv

    def simulate_external_update(*args, **kwargs):
        result = original(*args, **kwargs)
        with path.open("a", encoding="utf-8") as changed:
            changed.write("\n")
        return result

    monkeypatch.setattr("idwx.data_audit.ingest_era5_hourly_csv", simulate_external_update)
    report = audit_data(cfg, [station])
    assert report["audit_status"] == "blocked"
    assert "changed during audit" in report["blocking_issues"][0]["detail"]
    assert report["files"][0]["historical_review"] is None
