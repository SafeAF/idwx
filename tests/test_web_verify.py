import csv
import hashlib
import json
import zipfile
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from idwx.cli import app
from idwx.config import StationMeta, load_config
from idwx.web_export import export_web_data
from idwx.web_verify import package_web, verify_web


@pytest.fixture
def publication(tmp_path):
    cfg = replace(load_config("configs/idaho.yml"), backtest={"start_year": 2000, "end_year": 2022})
    stations = [StationMeta("demo", "Synthetic fixture", timezone="UTC")]
    root = tmp_path / "public"
    manifest = export_web_data(cfg, stations, root, demo=True)
    return cfg, stations, root, manifest


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def update_digest(root, relative):
    path = root / "data/manifest.json"
    manifest = read_json(path)
    manifest["files"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    write_json(path, manifest)


def station_path(root, manifest):
    return root / f"data/{manifest['release_id']}/stations/demo.json"


def test_verifier_requires_explicit_demo_and_cli_reports_its_scope(publication):
    _, _, root, manifest = publication
    with pytest.raises(ValueError, match="--allow-demo"):
        verify_web(root)
    report = verify_web(root, allow_demo=True)
    assert report["release_id"] == manifest["release_id"]
    assert report["stations"] == 1 and report["annual_records"] == 46
    assert report["forecasts_included"] is False
    assert report["climatology_projection_included"] is True
    runner = CliRunner()
    invalid = runner.invoke(app, ["verify-web", str(root)])
    assert invalid.exit_code != 0 and "--allow-demo" in invalid.output
    valid = runner.invoke(app, ["verify-web", str(root), "--allow-demo"])
    assert valid.exit_code == 0, valid.output
    assert json.loads(valid.output) == report


def test_retained_releases_are_verified_independently(publication):
    cfg, stations, root, first = publication
    second = export_web_data(cfg, stations, root, demo=True, min_years=2)
    report = verify_web(root, allow_demo=True)
    assert report["release_id"] == second["release_id"] and report["retained_releases_verified"] == 1
    old = station_path(root, first)
    data = read_json(old)
    data["forecasts"] = {"secret": "not a historical field"}
    write_json(old, data)
    with pytest.raises(ValueError, match="station document fields"):
        verify_web(root, allow_demo=True)


@pytest.mark.parametrize("relative", ["reports/audit.json", "model.pkl", ".staging-leftover/private.json", "data/private.csv"])
def test_private_or_unexpected_files_are_rejected(publication, relative):
    _, _, root, _ = publication
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        verify_web(root, allow_demo=True)


def test_missing_files_and_checksum_corruption_are_rejected(publication):
    _, _, root, manifest = publication
    document = station_path(root, manifest)
    original = document.read_bytes()
    document.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_web(root, allow_demo=True)
    document.unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        verify_web(root, allow_demo=True)


@pytest.mark.parametrize("mutation, message", [
    ("private_field", "station document fields"), ("missing_year", "Missing annual"),
    ("coverage", "Invalid event coverage"), ("candidate", "Invalid provisional"),
    ("summary", "Historical summary disagrees"), ("interval", "Verified interval count"),
])
def test_matching_checksum_does_not_hide_contract_errors(publication, mutation, message):
    _, _, root, manifest = publication
    path = station_path(root, manifest)
    document = read_json(path)
    if mutation == "private_field":
        document["raw_observations"] = []
    elif mutation == "missing_year":
        document["annual"].pop()
    elif mutation == "coverage":
        document["annual"][0]["last_spring_frost"]["valid_days"] = 1
    elif mutation == "candidate":
        document["annual"][0]["last_spring_frost"]["candidate_tmin_c"] = 100.
    elif mutation == "summary":
        document["summaries"][0]["last_spring_frost"]["valid_year_count"] = 999
    else:
        row = next(row for row in document["annual"] if row["interval"]["status"] == "verified")
        row["interval"]["frost_free_days"] += 1
    write_json(path, document)
    update_digest(root, path.relative_to(root).as_posix())
    with pytest.raises(ValueError, match=message):
        verify_web(root, allow_demo=True)


def test_csv_cannot_disagree_even_when_its_checksum_is_updated(publication):
    _, _, root, manifest = publication
    path = root / manifest["history_csv_url"]
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["last_spring_frost_date"] = "2000-01-01"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    update_digest(root, manifest["history_csv_url"])
    with pytest.raises(ValueError, match="CSV disagrees"):
        verify_web(root, allow_demo=True)


def test_canonical_release_identity_is_verified_after_document_checks(publication):
    _, _, root, manifest = publication
    for relative in [manifest["stations_url"], station_path(root, manifest).relative_to(root).as_posix()]:
        path = root / relative
        document = read_json(path)
        document["methodology"] = "Changed canonical content"
        write_json(path, document)
        update_digest(root, relative)
    current = read_json(root / "data/manifest.json")
    current["methodology"] = "Changed canonical content"
    write_json(root / "data/manifest.json", current)
    with pytest.raises(ValueError, match="Release ID disagrees"):
        verify_web(root, allow_demo=True)


@pytest.mark.parametrize("relative", ["../outside.json", "data/RELEASE/../../outside.json", "data/RELEASE/%2e%2e/outside.json"])
def test_release_references_cannot_escape_the_publication(publication, relative):
    _, _, root, manifest = publication
    manifest["files"][relative.replace("RELEASE", manifest["release_id"])] = "0" * 64
    write_json(root / "data/manifest.json", manifest)
    with pytest.raises(ValueError, match="outside|Unsafe"):
        verify_web(root, allow_demo=True)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_values_are_rejected(publication, constant):
    _, _, root, manifest = publication
    path = station_path(root, manifest)
    content = path.read_text(encoding="utf-8")
    changed = content.replace('"lat": null', f'"lat": {constant}', 1)
    assert changed != content
    path.write_text(changed, encoding="utf-8")
    update_digest(root, path.relative_to(root).as_posix())
    with pytest.raises(ValueError, match="Nonfinite"):
        verify_web(root, allow_demo=True)


def test_duplicate_json_keys_are_rejected(publication):
    _, _, root, _ = publication
    path = root / "data/manifest.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('{', '{"schema_version":"1.0.0",', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        verify_web(root, allow_demo=True)


def test_zip_is_deterministic_portable_and_contains_only_verified_content(publication, tmp_path):
    _, _, root, _ = publication
    first = package_web(root, tmp_path / "first.zip", allow_demo=True)
    second = package_web(root, tmp_path / "second.zip", allow_demo=True)
    assert first["sha256"] == second["sha256"]
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(first["archive"]) as archive:
        assert "index.html" in archive.namelist() and "data/manifest.json" in archive.namelist()
        assert all(not name.endswith((".py", ".parquet", ".pkl")) for name in archive.namelist())
        archive.extractall(destination)
    assert verify_web(destination, allow_demo=True)["release_id"] == first["release_id"]
    with pytest.raises(ValueError, match="outside"):
        package_web(root, root / "site.zip", allow_demo=True)


def test_failed_archive_switch_preserves_previous_artifact(publication, tmp_path, monkeypatch):
    _, _, root, _ = publication
    output = tmp_path / "site.zip"
    output.write_bytes(b"previous")

    def fail_replace(*args):
        raise OSError("Switch failed")

    monkeypatch.setattr("idwx.web_verify.os.replace", fail_replace)
    with pytest.raises(OSError, match="Switch failed"):
        package_web(root, output, allow_demo=True)
    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".site-*"))
