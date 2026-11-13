# Frost dashboard: preview, deployment, and verification

## What is implemented

`web/` contains semantic HTML, responsive CSS, and browser JavaScript modules.
The exporter installs these assets alongside versioned JSON/CSV, both from
this checkout and from an installed Python wheel. No frontend build step,
third-party scripts, external fonts, or weather API calls are required.

The dashboard supports station/threshold/year selection, Celsius/Fahrenheit,
shareable URL parameters, observed dates alongside historical typical dates,
P10-P90 chart bands, an accessible history table, selected-season coverage,
provisional candidates, and downloads. A historical baseline table shows
earliest/latest confirmed month/day, typical dates and P10-P90 ranges, and
separate counts for confirmed events, complete no-event windows, incomplete
windows and years without valid minima. Counts use only the published baseline
and selected threshold; they are not future freeze probabilities. Short
baselines retain observed extremes but do not acquire unsupported quantiles. The year selector controls summary
cards and details; the history chart/table and filtered download retain all
years for the selected station and threshold. A second download includes all
stations and thresholds from the pinned release.

Unavailable data, incomplete seasons, missing events, insufficient history,
and failed requests have explicit states. Old source records and synthetic
data have prominent notices. Export time and observation freshness are
displayed separately. No forecasts are shown.

## Local preview

```powershell
uv sync
uv run idwx export-web --output .tmp/twin-falls-web
uv run idwx verify-web .tmp/twin-falls-web
uv run python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/twin-falls-web
```

Open `http://127.0.0.1:8765/`. Stop the server with Ctrl+C when finished.
Opening `index.html` directly through `file://` will not load modules/data.
This export reads the committed Twin Falls database; the raw hourly archive is
not required. Use `--demo` and a separate output directory only when an
explicitly labeled synthetic preview is wanted.

## Nginx configuration

`deploy/nginx.conf` is an include for the Nginx `http {}` context. It uses a
dedicated document root at `/srv/idwx/current`. Replace the example domain,
document root, and existing certificate/key paths before installation.
Provision and renew certificates using the server's normal certificate setup.
The configuration redirects HTTP to HTTPS and permits TLS 1.2/1.3. It is not
installed automatically, and no live server credentials have been assumed.

The configuration provides MIME types including an explicit CSV mapping,
compression, same-origin content restrictions, and real missing-file errors.
It revalidates frontend assets and the manifest. Immutable release paths
cache for a year on successful responses; missing files do not receive that
cache policy. Dotfiles and exporter staging directories are denied.

After configuring the server:

```sh
sudo nginx -t
sudo systemctl reload nginx
```

## Staged updates

Run Python on a data-processing machine or batch runner. Upload only the
complete generated publication directory. Keep source CSVs, daily caches,
models, Python environments, and the repository outside the document root.

Use a unique build directory, for example `/srv/idwx/builds/<build-id>`, and
upload the new complete export there. Before activating it, copy retained
versioned data from the current build while excluding its old manifest:

```sh
# Set these to explicitly chosen, existing build directories.
new_build=/srv/idwx/builds/BUILD_ID
previous_build=$(readlink -f /srv/idwx/current)
rsync -a --ignore-existing --exclude=/manifest.json "$previous_build/data/" "$new_build/data/"
```

For the first deployment there is no previous build; skip that copy. Run
`idwx verify-web NEW_BUILD` on the batch machine after retained releases are
merged; this rejects demo releases by default. Transfer a verified archive
created with `idwx package-web NEW_BUILD --output delivery/site.zip`, or
verify the final staged tree directly on a machine with idwx installed.
Keep both the archive and verification receipt outside the document root.
Then perform HTTP checks
on a staging virtual host pointing to the new directory. Do not make edits
inside immutable release directories. Retain the previous build path for
rollback. Switch the document root symlink on the same filesystem:

```sh
ln -s "$new_build" /srv/idwx/.current-next
mv -Tf /srv/idwx/.current-next /srv/idwx/current
```

This assumes GNU `mv`, a symlink at `current`, and no existing `.current-next`;
inspect and resolve a leftover temporary link before starting another update.
Nginx uses the new directory on subsequent requests without a reload when
the server configuration itself is unchanged. Keep one batch publisher at a
time. Release retention lets existing pages keep fetching their pinned data.

Regenerate after source updates or relevant configuration/code changes. A
schedule should follow the actual source delivery cadence; no scheduler is
installed while that cadence is unknown. Exporting again does not imply that
new observations arrived.

## HTTP smoke checks

```sh
curl -I http://YOUR_DOMAIN/
curl -I https://YOUR_DOMAIN/
curl -I https://YOUR_DOMAIN/js/app.js
curl -I https://YOUR_DOMAIN/data/manifest.json
curl -I https://YOUR_DOMAIN/data/RELEASE_ID/stations.json
curl -I https://YOUR_DOMAIN/data/RELEASE_ID/downloads/frost-history.csv
curl -I https://YOUR_DOMAIN/missing.js
curl -I https://YOUR_DOMAIN/data/RELEASE_ID/missing.json
curl -I https://YOUR_DOMAIN/data/.staging-test/private.json
```

Expect an HTTP-to-HTTPS redirect, then HTML/JavaScript/JSON/CSV content types,
revalidation for assets/manifest, long immutable caching only for existing
release files, 404 for missing files, and 403/404 for staging paths. Verify
the live certificate hostname, trust, and expiry using the browser or the
server's monitoring. Never bypass certificate warnings.

## Rollback

Keep the previous build and its original manifest. Before switching back,
copy newer immutable releases into the previous build so pages already pinned
to the newer release continue to work after rollback:

```sh
active_build=$(readlink -f /srv/idwx/current)
# previous_build must be the retained, explicitly chosen prior build.
rsync -a --ignore-existing --exclude=/manifest.json "$active_build/data/" "$previous_build/data/"
ln -s "$previous_build" /srv/idwx/.current-rollback
mv -Tf /srv/idwx/.current-rollback /srv/idwx/current
```

Repeat the HTTP checks. Prune old builds/releases only after an established
retention period; cleanup is intentionally not automated here.

## Verification report

Python regressions cover calculations, export contracts, atomic manifest
replacement, and copied frontend assets. Frontend tests cover date-only
formatting independent of local timezones, leap-calendar mapping, exact
threshold labels, insufficient history, pinned release URLs, and CSV output.

```powershell
uv run pytest -q
node --test tests/web.test.mjs
```

The optional live Nginx test accepts `IDWX_NGINX` pointing to an installed
executable. On Windows its adjacent `conf/mime.types` is used; on Linux the
test falls back to `/etc/nginx/mime.types`:

```powershell
$env:IDWX_NGINX = 'C:/path/to/nginx.exe'
uv run pytest -q tests/test_nginx.py
```

The two tests bind only to loopback and stop their servers. They run
`nginx -t`, check MIME/cache/error/denied-path behavior and manifest rollback,
then exercise the full template's HTTP redirect and TLS 1.2/1.3 handshakes
with an ephemeral certificate. OpenSSL must be installed or supplied through
`IDWX_OPENSSL`; Git for Windows' OpenSSL is discovered automatically. The
certificate is trusted only by that test client's explicit context; default
trust correctly rejects it. No system trust configuration is changed.
Both tests passed with the official Windows Nginx package. Production
hostname, public trust, renewal and expiry still require live deployment checks.

Browser preview was attempted but its access host was unavailable in the
implementation session. A native Firefox fallback on November 11, 2026 was
stopped by the computer-use tool because URL policy enforcement is unsupported
for this Windows browser. The companion page supplied no document session.
The completion pass rechecked browser inventory and the access host remained
unavailable. Desktop/mobile appearance, real browser keyboard behavior and
native download behavior still need live browser verification. No screenshots
or claims of browser-tested behavior are provided.

DOM integration tests now execute the actual dashboard modules against
exported synthetic fixtures using jsdom. Twenty frontend tests cover URL
restoration/normalization, controls and units, all data states, chart/table/CSV
agreement, malformed/mixed releases, failed requests and recovery, late-response
races, baseline filtering and missingness counts, observed extremes with
insufficient history, semantic controls and focus targets. These tests do not render layout
or replace desktop/mobile browser checks. Install test-only dependencies with
the pinned pnpm version (Node 24.15+ in the 24 series, or Node 26+):

```powershell
npm install --global pnpm@11.25.0
pnpm --dir web install --frozen-lockfile --ignore-scripts
pnpm --dir web test
```

Set `IDWX_PYTHON` to the Python executable if it is not in the checkout's
standard `.venv`. The test fixture is explicitly synthetic and is cleaned up.
jsdom/pnpm are development dependencies and are excluded from exports/wheels;
the production website remains buildless.

The initial forecast audit and input leakage corrections are recorded in
`16_FORECAST_CORRECTNESS_AUDIT.md`. Those synthetic regression checks do not
establish forecast accuracy or calibration. No forecasts are published.

The completion suite has 225 Python tests passing, two optional Nginx tests
skipped in the default run, 20 frontend tests passing, and both Nginx tests
passing separately. The source-audit command and its
blocked workspace report are documented in `20_SOURCE_DATA_AUDIT.md`. Reports
contain private paths/metadata and must remain outside the publication root.

## Repeatable delivery checks

With Python/uv, Node 24.15+ (24 series) or 26+, and pnpm 11.25.0 available:

```powershell
.venv/Scripts/python.exe scripts/verify.py
# Include an installed Nginx for loopback HTTP/TLS verification:
.venv/Scripts/python.exe scripts/verify.py --nginx C:/path/to/nginx.exe
```

The Python runner installs locked dependencies without frontend lifecycle
scripts, runs full Ruff/Python/frontend checks, builds and verifies the demo
archive, and checks wheel exports both extracted and installed in a fresh
isolated environment. It uses workspace caches and short Windows fixture
paths. `--node`, `--pnpm`, and `--offline` support local installations and
cached operation. It does not alter PowerShell execution policy.
Artifacts are `.tmp/verified-site`, `.tmp/delivery/idwx-demo-site.zip`, and
`.tmp/delivery/wheels/`. These are synthetic, ignored by Git, and intended
for preview/testing. Real deployment must use audited data and omit both
`--demo` and `--allow-demo`.

`.github/workflows/verify.yml` configures the same checks on Windows and
Linux, with the optional Nginx smoke checks on Linux. It verifies and uploads
both the real Twin Falls deployment ZIP and the synthetic demo ZIP, plus the
wheel. Action revisions, pnpm and dependencies are pinned.
The workflow is configured but has not been run remotely in this session.
See `21_DELIVERY_STATUS.md` for the launch checklist and current blockers.
