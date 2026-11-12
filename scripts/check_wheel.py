"""Exercise an extracted wheel using the caller's installed runtime dependencies."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    wheels = list(source.glob("*.whl")) if source.is_dir() else [source]
    if len(wheels) != 1:
        raise ValueError("Pass one wheel or a directory containing exactly one wheel")
    repository = Path(__file__).resolve().parents[1]
    temporary_root = repository / ".tmp"
    temporary_root.mkdir(exist_ok=True)
    installed = "--installed" in sys.argv[2:]
    offline = "--offline" in sys.argv[2:]
    with tempfile.TemporaryDirectory(prefix="wi-", dir=temporary_root) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(wheels[0]) as archive:
            for name in archive.namelist():
                if not (stage / name).resolve().is_relative_to(stage):
                    raise ValueError("Wheel contains an unsafe path")
                if not name.startswith("idwx/") and not re.match(r"^idwx-[^/]+\.dist-info/", name):
                    raise ValueError(f"Unexpected wheel content: {name}")
            if not installed:
                archive.extractall(stage)
        executable = Path(sys.executable)
        if installed:
            uv = shutil.which("uv")
            if uv is None:
                raise ValueError("Install uv for the isolated wheel check")
            prefix = [uv, "--cache-dir", str(temporary_root / "uv-cache"), *(["--offline"] if offline else [])]
            constraints = stage / "constraints.txt"
            subprocess.run([*prefix, "export", "--locked", "--no-dev", "--no-emit-project", "--no-hashes",
                            "--output-file", str(constraints)], cwd=repository, check=True, stdout=subprocess.DEVNULL)
            environment_root = stage / "venv"
            subprocess.run([*prefix, "venv", "--python", sys.executable, str(environment_root)], check=True)
            executable = environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run([*prefix, "pip", "install", "--python", str(executable), "--constraint", str(constraints),
                            str(wheels[0])], check=True)
        (stage / "stations.yml").write_text("stations: [{station_id: wheel_demo, name: Wheel demonstration}]\n", encoding="utf-8")
        (stage / "config.yml").write_text("stations_file: stations.yml\nbacktest: {start_year: 2000, end_year: 2022}\n", encoding="utf-8")
        environment = {**os.environ, "PYTHONPATH": str(stage)}

        def run(*arguments: str) -> str:
            return subprocess.check_output([str(executable), *(["-I"] if installed else []), *arguments],
                                           cwd=stage, env=environment, text=True)

        imported = Path(run("-c", "import idwx;print(idwx.__file__)").strip()).resolve()
        if not imported.is_relative_to(stage / "venv" if installed else stage):
            raise ValueError("Wheel verification imported a checkout instead of the extracted wheel")
        run("-m", "idwx", "export-web", "--config", "config.yml", "--demo", "--output", "site")
        report = json.loads(run("-m", "idwx", "verify-web", "site", "--allow-demo"))
        run("-m", "idwx", "package-web", "site", "--allow-demo", "--output", "demo.zip")
        print(json.dumps({"wheel": wheels[0].name, "installed_in_isolated_environment": installed,
                          "imported_from_extracted_wheel": not installed, **report}))


if __name__ == "__main__":
    main()
