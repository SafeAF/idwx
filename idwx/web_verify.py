"""Read-only static publication checks and a verified, portable site archive."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from idwx.history import summarize_events
from idwx.web_export import EVENT_NAMES, SCHEMA_VERSION, WEB_ASSETS

COMMON = {"schema_version", "release_id", "data_kind", "methodology"}
STATION_FIELDS = {"station", "source_coverage", "season_years", "windows", "thresholds_c", "annual", "summaries"}
EVENT_FIELDS = {"status", "date", "tmin_c", "candidate_date", "candidate_tmin_c", "window_start", "window_end",
                "expected_days", "valid_days", "missing_days", "coverage_fraction"}
EVENT_STATUSES = {"observed", "no_event_in_complete_window", "incomplete_window", "insufficient_data"}
STATION_ID = r"[a-z0-9]+(?:_[a-z0-9]+)*"
RELEASE_ID = r"[a-f0-9]{64}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value, expected: set[str], label: str) -> None:
    _require(isinstance(value, dict) and set(value) == expected, f"Invalid {label} fields")


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value, minimum=0) -> bool:
    return type(value) is int and value >= minimum


def _date(value) -> date:
    _require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), "Invalid calendar date")
    return date.fromisoformat(value)


def _read_json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        parsed = float(value)
        _require(math.isfinite(parsed), "Nonfinite JSON number")
        return parsed

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                       parse_float=finite_float, object_pairs_hook=pairs)
    _require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def _release_path(root: Path, name: str, release: str) -> Path:
    _require(isinstance(name, str) and not any(char in name for char in "\\%?#:"), "Unsafe release reference")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and ".." not in path.parts and name == path.as_posix()
             and name.startswith(f"data/{release}/"), "Reference is outside the pinned release")
    result = root / name
    _require(result.resolve().is_relative_to(root) and result.is_file(), f"Missing or unsafe release file: {name}")
    return result


def _inventory(root: Path) -> list[Path]:
    files = []
    file_pattern = rf"data/{RELEASE_ID}/(?:stations\.json|stations/{STATION_ID}\.json|downloads/frost-history\.csv)"
    directory_pattern = rf"(?:css|js|data|data/{RELEASE_ID}(?:/(?:stations|downloads))?)"
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        _require(not path.is_symlink(), f"Publication contains a symlink: {name}")
        if path.is_dir():
            _require(re.fullmatch(directory_pattern, name), f"Unexpected publication directory: {name}")
        else:
            _require(path.is_file() and (name in WEB_ASSETS or name == "data/manifest.json" or re.fullmatch(file_pattern, name)),
                     f"Unexpected/private publication file: {name}")
            files.append(path)
    for asset in (*WEB_ASSETS, "data/manifest.json"):
        _require((root / asset).is_file(), f"Missing publication asset: {asset}")
    return files


def _common(document: dict, manifest: dict) -> None:
    _require(all(document.get(key) == manifest[key] for key in COMMON), "Mixed or incompatible release documents")


def _event(event: dict, year: int, threshold: float, window: dict) -> None:
    _keys(event, EVENT_FIELDS, "event")
    start, end = (_date(f"{year:04d}-{window[key]}") for key in ("start", "end"))
    _require(start <= end and event["window_start"] == start.isoformat() and event["window_end"] == end.isoformat(), "Event window mismatch")
    expected = (end - start).days + 1
    valid, missing, fraction, status = (event[key] for key in ("valid_days", "missing_days", "coverage_fraction", "status"))
    _require(status in EVENT_STATUSES and _integer(valid) and _integer(missing) and _integer(event["expected_days"], 1) and event["expected_days"] == expected
             and valid + missing == expected and _number(fraction) and abs(fraction - valid / expected) < 1e-12,
             "Invalid event coverage")
    candidate, temperature = event["candidate_date"], event["candidate_tmin_c"]
    if candidate is None:
        _require(temperature is None, "Candidate temperature without date")
    else:
        _require(valid > 0 and start <= _date(candidate) <= end and _number(temperature) and temperature <= threshold, "Invalid provisional candidate")
    if status == "observed":
        _require(valid == expected and candidate is not None and event["date"] == candidate and event["tmin_c"] == temperature,
                 "Confirmed event is not supported by its coverage/candidate")
    else:
        _require(event["date"] is None and event["tmin_c"] is None, "Unconfirmed event exposes a confirmed date")
        if status == "no_event_in_complete_window":
            _require(valid == expected and candidate is None, "Invalid complete no-event window")
        elif status == "insufficient_data":
            _require(valid == 0 and candidate is None, "Invalid insufficient-data window")
        else:
            _require(0 < valid < expected, "Invalid incomplete window")


def _station(document: dict, manifest: dict, expected_station: dict) -> None:
    _keys(document, COMMON | STATION_FIELDS, "station document")
    _common(document, manifest)
    station = document["station"]
    _keys(station, {"station_id", "name", "lat", "lon", "elevation_m", "source", "timezone"}, "station metadata")
    _require(station == expected_station and re.fullmatch(STATION_ID, station["station_id"]), "Station index/document mismatch")
    _require(isinstance(station["name"], str) and station["name"] and isinstance(station["timezone"], str), "Invalid station name/timezone")
    try:
        ZoneInfo(station["timezone"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Unknown station timezone") from exc
    for key in ("lat", "lon", "elevation_m"):
        _require(station[key] is None or _number(station[key]), "Invalid numeric station metadata")
    _require((station["lat"] is None or -90 <= station["lat"] <= 90) and
             (station["lon"] is None or -180 <= station["lon"] <= 180), "Station coordinates out of range")
    _require(station["source"] is None or isinstance(station["source"], str), "Invalid source declaration")
    coverage = document["source_coverage"]
    _keys(coverage, {"first_date", "last_date", "first_valid_minimum_date", "last_valid_minimum_date", "recorded_days", "valid_minimum_days"}, "source coverage")
    first, last = _date(coverage["first_date"]), _date(coverage["last_date"])
    _require(first <= last and _integer(coverage["recorded_days"], 1) and _integer(coverage["valid_minimum_days"])
             and coverage["valid_minimum_days"] <= coverage["recorded_days"] <= (last - first).days + 1, "Invalid source coverage counts")
    if coverage["valid_minimum_days"]:
        _require(first <= _date(coverage["first_valid_minimum_date"]) <= _date(coverage["last_valid_minimum_date"]) <= last, "Invalid minimum coverage dates")
    else:
        _require(coverage["first_valid_minimum_date"] is None and coverage["last_valid_minimum_date"] is None, "Minimum dates without valid observations")
    _keys(document["season_years"], {"start_year", "end_year"}, "season years")
    start, end = (document["season_years"][key] for key in ("start_year", "end_year"))
    _require(_integer(start, 1) and _integer(end, start) and end <= 9999, "Invalid season year range")
    thresholds = document["thresholds_c"]
    _require(isinstance(thresholds, list) and thresholds and all(_number(value) for value in thresholds)
             and len(thresholds) == len(set(thresholds)), "Invalid thresholds")
    _keys(document["windows"], set(EVENT_NAMES), "station windows")
    for window in document["windows"].values():
        _keys(window, {"start", "end"}, "window")
    annual, summaries = document["annual"], document["summaries"]
    _require(isinstance(annual, list) and isinstance(summaries, list), "Missing historical arrays")
    _require(all(isinstance(summary, dict) for summary in summaries), "Invalid threshold summaries")
    seen = set()
    for row in annual:
        _keys(row, {"season_year", "threshold_c", *EVENT_NAMES, "interval"}, "annual record")
        year, threshold = row["season_year"], row["threshold_c"]
        _require(_integer(year, start) and year <= end and _number(threshold) and threshold in thresholds, "Unexpected season/threshold")
        _require((year, threshold) not in seen, "Duplicate season/threshold")
        seen.add((year, threshold))
        for name in EVENT_NAMES:
            _event(row[name], year, threshold, document["windows"][name])
        interval = row["interval"]
        _keys(interval, {"date_difference_days", "frost_free_days", "status", "summer_freeze_observed"}, "interval")
        _require(type(interval["summer_freeze_observed"]) is bool, "Invalid summer-freeze flag")
        _require((interval["date_difference_days"] is None or _number(interval["date_difference_days"])) and
                 (interval["frost_free_days"] is None or _number(interval["frost_free_days"]) and
                  interval["frost_free_days"] >= 0 and interval["frost_free_days"] == math.floor(interval["frost_free_days"])), "Invalid interval numbers")
        spring, fall = (row[name]["date"] for name in EVENT_NAMES)
        if spring is None or fall is None:
            _require(interval["status"] == "missing_dependency" and interval["date_difference_days"] is None and interval["frost_free_days"] is None, "Invalid missing interval dependency")
        elif fall <= spring:
            _require(interval["status"] == "invalid_event_order" and interval["date_difference_days"] is None and interval["frost_free_days"] is None, "Invalid event order handling")
        else:
            difference = (_date(fall) - _date(spring)).days
            _require(interval["date_difference_days"] == difference, "Interval date difference mismatch")
            if interval["status"] == "verified":
                _require(interval["frost_free_days"] == max(0, difference - 1), "Verified interval count mismatch")
            else:
                _require(interval["status"] in {"incomplete_interval", "intervening_freeze"} and interval["frost_free_days"] is None, "Invalid unverified interval")
    _require(len(seen) == (end - start + 1) * len(thresholds), "Missing annual season/threshold records")
    _require(len(summaries) == len(thresholds), "Missing historical summaries")
    for threshold in thresholds:
        matches = [summary for summary in summaries if summary.get("threshold_c") == threshold]
        _require(len(matches) == 1, "Duplicate or missing threshold summary")
        summary = matches[0]
        _keys(summary, {"threshold_c", *EVENT_NAMES}, "threshold summary")
        for name in EVENT_NAMES:
            distribution = summary[name]
            period = distribution["baseline_period"]
            _keys(period, {"start_year", "end_year"}, "baseline")
            a, b, minimum = period["start_year"], period["end_year"], distribution["minimum_valid_years"]
            _require(_integer(a, start) and _integer(b, a) and b <= end and _integer(minimum, 1), "Invalid historical baseline")
            records = [{"season_year": row["season_year"], **row[name]} for row in annual if row["threshold_c"] == threshold]
            _require(distribution == summarize_events(records, a, b, minimum), "Historical summary disagrees with annual records")


def _verify_release(directory: Path, *, allow_demo: bool, supplied_manifest: dict | None = None) -> dict:
    root = directory.resolve()
    _require(root.is_dir(), "Publication directory does not exist")
    _inventory(root)
    manifest = supplied_manifest if supplied_manifest is not None else _read_json(root / "data/manifest.json")
    _keys(manifest, COMMON | {"generated_at_utc", "stations_url", "history_csv_url", "source_coverage", "files"}, "manifest")
    release = manifest["release_id"]
    _require(manifest["schema_version"] == SCHEMA_VERSION and isinstance(release, str) and re.fullmatch(RELEASE_ID, release), "Unsupported schema or release ID")
    _require(manifest["data_kind"] in {
        "synthetic_demo", "historical_observations", "historical_reanalysis",
    }, "Unsupported data kind")
    _require(allow_demo or manifest["data_kind"] != "synthetic_demo", "Synthetic data requires explicit --allow-demo")
    _require(isinstance(manifest["methodology"], str), "Missing methodology")
    _require(isinstance(manifest["generated_at_utc"], str), "Missing generation timestamp")
    generated = datetime.fromisoformat(manifest["generated_at_utc"].replace("Z", "+00:00"))
    _require(generated.tzinfo is not None and generated.utcoffset() == timedelta(0), "Generation timestamp must be UTC-aware")
    _require(manifest["stations_url"] == f"data/{release}/stations.json" and manifest["history_csv_url"] == f"data/{release}/downloads/frost-history.csv", "Invalid release entry points")
    _require(isinstance(manifest["files"], dict), "Missing release checksums")
    for name, digest in manifest["files"].items():
        path = _release_path(root, name, release)
        _require(isinstance(digest, str) and re.fullmatch(RELEASE_ID, digest) and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                 f"Checksum mismatch: {name}")
    index = _read_json(_release_path(root, manifest["stations_url"], release))
    _keys(index, COMMON | {"stations"}, "station index")
    _common(index, manifest)
    _require(isinstance(index["stations"], list) and index["stations"], "No stations in release")
    documents, ids, expected_files, rows = [], set(), {manifest["stations_url"], manifest["history_csv_url"]}, []
    for entry in index["stations"]:
        _keys(entry, {"station", "source_coverage", "url"}, "station index entry")
        sid = entry["station"]["station_id"]
        _require(isinstance(sid, str) and re.fullmatch(STATION_ID, sid) and sid not in ids, "Invalid or duplicate station ID")
        ids.add(sid)
        _require(entry["url"] == f"data/{release}/stations/{sid}.json", "Station URL mismatch")
        expected_files.add(entry["url"])
        document = _read_json(_release_path(root, entry["url"], release))
        _station(document, manifest, entry["station"])
        _require(entry["source_coverage"] == document["source_coverage"], "Index source coverage mismatch")
        documents.append({key: value for key, value in document.items() if key not in COMMON})
        for record in document["annual"]:
            row = {key: manifest[key] for key in ("schema_version", "release_id", "data_kind")}
            row.update(station_id=sid, season_year=record["season_year"], threshold_c=record["threshold_c"])
            for name in (*EVENT_NAMES, "interval"):
                row.update({f"{name}_{key}": value for key, value in record[name].items()})
            rows.append({key: "" if value is None else str(value) for key, value in row.items()})
    actual_files = {path.relative_to(root).as_posix() for path in (root / "data" / release).rglob("*") if path.is_file()}
    _require(expected_files == set(manifest["files"]) == actual_files, "Release inventory/checksum references disagree")
    with (root / manifest["history_csv_url"]).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
                 and set(reader.fieldnames) == set(rows[0]) and list(reader) == rows, "CSV disagrees with station JSON")
    expected_coverage = {"first_date": min(doc["source_coverage"]["first_date"] for doc in documents),
                         "last_date": max(doc["source_coverage"]["last_date"] for doc in documents)}
    _require(manifest["source_coverage"] == expected_coverage, "Manifest source coverage mismatch")
    identity = {key: manifest[key] for key in ("schema_version", "data_kind", "methodology")}
    identity["stations"] = documents
    content = (json.dumps(identity, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _require(hashlib.sha256(content).hexdigest() == release, "Release ID disagrees with canonical content")
    return {"status": "valid_static_package", "release_id": release, "data_kind": manifest["data_kind"],
            "stations": len(ids), "annual_records": len(rows), "forecasts_included": False}


def verify_web(directory: Path, *, allow_demo: bool = False) -> dict:
    """Verify active and retained releases; this does not certify source provenance."""
    try:
        root = directory.resolve()
        result = _verify_release(root, allow_demo=allow_demo)
        retained = 0
        for release in sorted((root / "data").iterdir()):
            if not release.is_dir() or release.name == result["release_id"]:
                continue
            index = _read_json(release / "stations.json")
            _require(index.get("release_id") == release.name, "Retained release ID mismatch")
            files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in release.rglob("*") if path.is_file()}
            coverage = [entry["source_coverage"] for entry in index["stations"]]
            previous = {key: index[key] for key in COMMON}
            # The original timestamp is absent after manifest rollback; it is
            # not part of content identity. Verify retained content independently.
            previous.update(generated_at_utc="2000-01-01T00:00:00+00:00", files=files,
                            stations_url=f"data/{release.name}/stations.json",
                            history_csv_url=f"data/{release.name}/downloads/frost-history.csv",
                            source_coverage={"first_date": min(item["first_date"] for item in coverage),
                                             "last_date": max(item["last_date"] for item in coverage)})
            _verify_release(root, allow_demo=allow_demo, supplied_manifest=previous)
            retained += 1
        return {**result, "retained_releases_verified": retained}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed static publication: {exc}") from exc


def package_web(directory: Path, output: Path, *, allow_demo: bool = False) -> dict:
    result = verify_web(directory, allow_demo=allow_demo)
    root, output = directory.resolve(), output.resolve()
    _require(output.suffix.lower() == ".zip" and not output.is_relative_to(root), "Choose a .zip output outside the publication directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".site-", suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w") as archive:
            for path in sorted(_inventory(root)):
                entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2000, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, path.read_bytes())
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".site-check-") as check:
            with zipfile.ZipFile(temporary) as archive:
                archive.extractall(check)  # Entries were produced from the validated allowlist above.
            _require(verify_web(Path(check), allow_demo=allow_demo) == result, "Packaged content changed during verification")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {**result, "archive": str(output), "sha256": digest}
