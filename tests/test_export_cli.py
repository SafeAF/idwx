import json

import yaml
from typer.testing import CliRunner

from idwx.cli import app


def test_export_web_cli_demo_and_invalid_baseline(tmp_path):
    stations = tmp_path / "stations.yml"
    stations.write_text(yaml.safe_dump({"stations": [{"station_id": "demo", "name": "Demo"}]}), encoding="utf-8")
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({
        "stations_file": str(stations), "cache_dir": str(tmp_path / "absent"),
        "backtest": {"start_year": 2000, "end_year": 2022},
    }), encoding="utf-8")
    runner = CliRunner()
    base_args = ["export-web", "--config", str(config), "--output", str(tmp_path / "web")]
    result = runner.invoke(app, [*base_args, "--demo"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["data_kind"] == "synthetic_demo"
    assert (tmp_path / "web/data/manifest.json").is_file()
    invalid = runner.invoke(app, [*base_args, "--demo", "--baseline-start", "2010"])
    assert invalid.exit_code != 0
    assert "Supply both" in invalid.output
    missing = runner.invoke(app, base_args)
    assert missing.exit_code != 0  # Synthetic output is never an implicit fallback.
