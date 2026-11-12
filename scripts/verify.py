"""Run the complete local verification and create a verified synthetic delivery."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--pnpm", type=Path, help="pnpm executable or pnpm.cjs")
    parser.add_argument("--nginx", type=Path, help="Optional installed Nginx executable for loopback smoke checks")
    parser.add_argument("--offline", action="store_true", help="Use only existing uv and pnpm caches")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install uv before running verification")
    node = args.node or shutil.which("node")
    if node is None and os.name == "nt":
        node = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    if node is None or not Path(node).is_file():
        raise ValueError("Install Node 24.15+ or supply --node")
    node = Path(node).resolve()
    pnpm = args.pnpm or shutil.which("pnpm") or node.parent.parent / "node_modules/pnpm/bin/pnpm.cjs"
    pnpm = Path(pnpm).resolve()
    if pnpm.suffix == ".cmd":
        adjacent_cli = pnpm.parent / "node_modules/pnpm/bin/pnpm.cjs"
        if adjacent_cli.is_file():
            pnpm = adjacent_cli
    if not pnpm.is_file():
        raise ValueError("Install pnpm 11.25.0 or supply --pnpm")
    environment = {**os.environ, "PATH": str(node.parent) + os.pathsep + os.environ.get("PATH", "")}
    uv_prefix = [uv, "--cache-dir", str(root / ".tmp/uv-cache"), *(["--offline"] if args.offline else [])]

    def run(command: list[str | Path], *, env: dict | None = None) -> None:
        subprocess.run([str(value) for value in command], cwd=root, env=env or environment, check=True)

    run([*uv_prefix, "sync", "--locked"])
    run([*uv_prefix, "run", "ruff", "check", "."])
    # Short names keep nested Windows model fixtures below legacy MAX_PATH.
    scratch = root / ".tmp" / f"t-{uuid4().hex[:8]}"
    if scratch.exists() or not scratch.resolve().is_relative_to(root / ".tmp"):
        raise ValueError("Unsafe verification scratch directory")
    run([*uv_prefix, "run", "pytest", "-q", "--maxfail=1", "-p", "no:cacheprovider", "--basetemp", scratch])
    prefix = [node, pnpm] if pnpm.suffix in {".cjs", ".js"} else [pnpm]
    run([*prefix, "--dir", "web", "install", "--frozen-lockfile", "--ignore-scripts",
         "--store-dir", root / ".tmp/pnpm-store", *(["--offline"] if args.offline else [])])
    run([*prefix, "--dir", "web", "test"])
    if args.nginx:
        nginx_scratch = root / ".tmp" / f"n-{uuid4().hex[:8]}"
        run([*uv_prefix, "run", "pytest", "-q", "--maxfail=1", "-p", "no:cacheprovider", "--basetemp", nginx_scratch, "tests/test_nginx.py"],
            env={**environment, "IDWX_NGINX": str(args.nginx.resolve())})
    run([*uv_prefix, "run", "idwx", "export-web", "--demo", "--output", ".tmp/verified-site"])
    run([*uv_prefix, "run", "idwx", "verify-web", ".tmp/verified-site", "--allow-demo"])
    run([*uv_prefix, "run", "idwx", "package-web", ".tmp/verified-site", "--allow-demo", "--output", ".tmp/delivery/idwx-demo-site.zip"])
    run([*uv_prefix, "build", "--wheel", "--out-dir", ".tmp/delivery/wheels"])
    run([*uv_prefix, "run", "python", "scripts/check_wheel.py", ".tmp/delivery/wheels"])
    run([*uv_prefix, "run", "python", "scripts/check_wheel.py", ".tmp/delivery/wheels", "--installed",
         *(["--offline"] if args.offline else [])])
    run(["git", "diff", "--check"])
    delivery = root / ".tmp/delivery"
    artifacts = [delivery / "idwx-demo-site.zip", *sorted((delivery / "wheels").glob("*.whl"))]
    (delivery / "SHA256SUMS.txt").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(delivery).as_posix()}\n"
        for path in artifacts), encoding="utf-8")
    print("Verified synthetic site: .tmp/verified-site\nPortable ZIP and wheel: .tmp/delivery", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
