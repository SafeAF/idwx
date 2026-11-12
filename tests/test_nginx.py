"""Optional live smoke test: set IDWX_NGINX to an installed Nginx executable."""
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import time
from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest

from idwx.config import StationMeta, load_config
from idwx.web_export import export_web_data


@pytest.mark.skipif(not os.environ.get("IDWX_NGINX"), reason="Set IDWX_NGINX for live server checks")
def test_nginx_static_delivery_and_manifest_rollback(tmp_path):
    executable = Path(os.environ["IDWX_NGINX"]).resolve()
    cfg = replace(load_config("configs/idaho.yml"), backtest={"start_year": 2000, "end_year": 2022})
    output = tmp_path / "public"
    station = StationMeta("demo", "Demo")
    first = export_web_data(cfg, [station], output, demo=True)
    old_manifest = (output / "data/manifest.json").read_bytes()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    template = Path("deploy/nginx.conf").read_text(encoding="utf-8")
    # Exercise production static locations over loopback HTTP; production TLS
    # certificates and redirect remain deployment-specific checks.
    server = template[template.index("server {", template.index("server {") + 1):]
    server = server.replace("listen 443 ssl;", f"listen 127.0.0.1:{port};")
    server = re.sub(r"^\s*ssl_.*;", "", server, flags=re.MULTILINE)
    server = server.replace("root /srv/idwx/current;", f'root "{output.as_posix()}";')
    mime = executable.parent / "conf/mime.types"
    if not mime.exists():
        mime = Path("/etc/nginx/mime.types")
    server = server.replace("include mime.types;", f'include "{mime.as_posix()}";')
    prefix = tmp_path / "nginx"
    (prefix / "logs").mkdir(parents=True)
    (prefix / "temp").mkdir()
    config = prefix / "nginx.conf"
    config.write_text("daemon off;\nmaster_process off;\nevents {}\nhttp {\naccess_log off;\n" + server + "\n}", encoding="utf-8")
    args = [str(executable), "-p", prefix.as_posix() + "/", "-c", config.as_posix()]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    checked = subprocess.run([*args, "-t"], capture_output=True, text=True, creationflags=flags, check=False)
    assert checked.returncode == 0, checked.stderr
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    base = f"http://127.0.0.1:{port}/"
    try:
        for _ in range(100):
            try:
                with urlopen(base, timeout=.3) as response:
                    assert response.status == 200
                break
            except URLError:
                if process.poll() is not None:
                    pytest.fail((prefix / "logs/error.log").read_text())
                time.sleep(.05)
        else:
            pytest.fail("Nginx did not become ready")
        with urlopen(base) as response:
            assert response.headers.get_content_type() == "text/html"
            assert "no-cache" in response.headers["Cache-Control"]
            assert "script-src 'self'" in response.headers["Content-Security-Policy"]
        for name in ("js/app.js", "css/styles.css", "data/manifest.json", first["stations_url"], first["history_csv_url"]):
            with urlopen(base + name) as response:
                content_type = response.headers.get_content_type()
                if name.endswith(".js"):
                    assert content_type in {"application/javascript", "text/javascript"}
                elif name.endswith(".css"):
                    assert content_type == "text/css"
                elif name.endswith(".json"):
                    assert content_type == "application/json"
                else:
                    assert content_type == "text/csv"
                assert ("immutable" in response.headers.get("Cache-Control", "")) == (name.startswith("data/") and name != "data/manifest.json")
        for name in ("missing.js", "data/missing.json", f"data/{first['release_id']}/missing.json"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + name)
            assert error.value.code == 404
            assert "immutable" not in error.value.headers.get("Cache-Control", "")
        (output / "data/.staging-test").mkdir()
        (output / "data/.staging-test/private.json").write_text("{}")
        with pytest.raises(HTTPError) as error:
            urlopen(base + "data/.staging-test/private.json")
        assert error.value.code in {403, 404}
        second = export_web_data(cfg, [station], output, min_years=2, demo=True)
        with urlopen(base + "data/manifest.json") as response:
            assert json.load(response)["release_id"] == second["release_id"]
        restore = output / "data/restore.json"
        restore.write_bytes(old_manifest)
        os.replace(restore, output / "data/manifest.json")
        with urlopen(base + "data/manifest.json") as response:
            assert json.load(response)["release_id"] == first["release_id"]
        with urlopen(base + second["stations_url"]) as response:
            assert response.status == 200  # Both pinned views remain available.
    finally:
        subprocess.run([*args, "-s", "quit"], capture_output=True, creationflags=flags, check=False)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.skipif(not os.environ.get("IDWX_NGINX"), reason="Set IDWX_NGINX for live server checks")
def test_nginx_tls_and_http_redirect(tmp_path):
    """Trust an ephemeral certificate only in this client, never system-wide."""
    openssl = os.environ.get("IDWX_OPENSSL") or shutil.which("openssl")
    if not openssl and os.name == "nt":
        candidate = Path("C:/Program Files/Git/usr/bin/openssl.exe")
        if candidate.is_file():
            openssl = str(candidate)
    if not openssl:
        pytest.skip("Install OpenSSL or set IDWX_OPENSSL for local TLS checks")
    executable = Path(os.environ["IDWX_NGINX"]).resolve()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    generated = subprocess.run([
        openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        "-keyout", str(key), "-out", str(cert),
    ], capture_output=True, text=True, creationflags=flags, check=False)
    assert generated.returncode == 0, generated.stderr
    # Reserve both ports simultaneously so they cannot be identical.
    with socket.socket() as http_sock, socket.socket() as https_sock:
        http_sock.bind(("127.0.0.1", 0))
        https_sock.bind(("127.0.0.1", 0))
        http_port, https_port = http_sock.getsockname()[1], https_sock.getsockname()[1]
    output = tmp_path / "public"
    export_web_data(load_config("configs/idaho.yml"), [StationMeta("demo", "Demo")], output, demo=True)
    template = Path("deploy/nginx.conf").read_text(encoding="utf-8")
    template = template.replace("listen 80;", f"listen 127.0.0.1:{http_port};")
    template = template.replace("listen 443 ssl;", f"listen 127.0.0.1:{https_port} ssl;")
    template = template.replace("https://frost.example.com$request_uri", f"https://localhost:{https_port}$request_uri")
    template = template.replace("root /srv/idwx/current;", f'root "{output.as_posix()}";')
    template = re.sub(r"ssl_certificate .+;", f'ssl_certificate "{cert.as_posix()}";', template)
    template = re.sub(r"ssl_certificate_key .+;", f'ssl_certificate_key "{key.as_posix()}";', template)
    mime = executable.parent / "conf/mime.types"
    if not mime.exists():
        mime = Path("/etc/nginx/mime.types")
    template = template.replace("include mime.types;", f'include "{mime.as_posix()}";')
    prefix = tmp_path / "nginx"
    (prefix / "logs").mkdir(parents=True)
    (prefix / "temp").mkdir()
    config = prefix / "nginx.conf"
    config.write_text("daemon off;\nmaster_process off;\nevents {}\nhttp {\naccess_log off;\n" + template + "\n}", encoding="utf-8")
    args = [str(executable), "-p", prefix.as_posix() + "/", "-c", config.as_posix()]
    checked = subprocess.run([*args, "-t"], capture_output=True, text=True, creationflags=flags, check=False)
    assert checked.returncode == 0, checked.stderr
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    try:
        context = ssl.create_default_context(cafile=str(cert))
        for _ in range(100):
            try:
                with urlopen(f"https://127.0.0.1:{https_port}/", context=context, timeout=.3) as response:
                    assert response.status == 200
                break
            except URLError:
                if process.poll() is not None:
                    pytest.fail((prefix / "logs/error.log").read_text())
                time.sleep(.05)
        else:
            pytest.fail("Nginx TLS did not become ready")
        connection = HTTPConnection("127.0.0.1", http_port, timeout=3)
        try:
            connection.request("GET", "/?station=demo&unit=f")
            response = connection.getresponse()
            assert response.status == 301
            assert response.headers["Location"] == f"https://localhost:{https_port}/?station=demo&unit=f"
            response.read()
        finally:
            connection.close()
        for version in (ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3):
            context.minimum_version = context.maximum_version = version
            with (
                socket.create_connection(("127.0.0.1", https_port), timeout=3) as sock,
                context.wrap_socket(sock, server_hostname="localhost") as secure,
            ):
                assert secure.version() == {ssl.TLSVersion.TLSv1_2: "TLSv1.2", ssl.TLSVersion.TLSv1_3: "TLSv1.3"}[version]
        # A normal trust store must reject the generated, untrusted certificate.
        with pytest.raises(URLError, match="CERTIFICATE_VERIFY_FAILED"):
            urlopen(f"https://127.0.0.1:{https_port}/", context=ssl.create_default_context(), timeout=3)
    finally:
        subprocess.run([*args, "-s", "quit"], capture_output=True, creationflags=flags, check=False)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
