# Frost dashboard: preview, deployment, and verification

## What is implemented

`web/` contains semantic HTML, responsive CSS, and browser JavaScript modules.
The exporter installs these assets alongside versioned JSON/CSV, both from
this checkout and from an installed Python wheel. No frontend build step,
third-party scripts, external fonts, or weather API calls are required.

The dashboard supports station/threshold/year selection, Celsius/Fahrenheit,
shareable URL parameters, observed dates alongside historical typical dates,
P10-P90 chart bands, an accessible history table, selected-season coverage,
provisional candidates, and downloads. The year selector controls summary
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
uv run idwx export-web --demo --output .tmp/demo-web
uv run python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/demo-web
```

Open `http://127.0.0.1:8765/`. Stop the server with Ctrl+C when finished.
Opening `index.html` directly through `file://` will not load modules/data.
For real inputs, rebuild daily caches and omit `--demo` when exporting into
a new staging directory. Audit the source first; the current workspace lacks
the configured real weather directory.

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

For the first deployment there is no previous build; skip that copy. Confirm
the new manifest's referenced JSON and CSV exist, then perform HTTP checks
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

The test binds only to loopback, runs `nginx -t` on a generated HTTP test
configuration, checks MIME/cache/error/denied-path behavior and manifest
rollback, then stops its server. The production static locations passed with
the official Windows Nginx package. Production TLS and redirect checks still
require the deployment's actual certificates and hostname.

Browser preview was attempted but its access host was unavailable in the
implementation session. A native Firefox fallback on November 11, 2026 was
stopped by the computer-use tool because URL policy enforcement is unsupported
for this Windows browser. The companion page supplied no document session.
Desktop/mobile appearance, interactive controls,
keyboard navigation, chart/table agreement in the rendered page, downloads,
and failed-request recovery therefore still need live browser verification.
No screenshots or claims of browser-tested behavior are provided.

The initial forecast audit and input leakage corrections are recorded in
`16_FORECAST_CORRECTNESS_AUDIT.md`. Those synthetic regression checks do not
establish forecast accuracy or calibration. No forecasts are published.
