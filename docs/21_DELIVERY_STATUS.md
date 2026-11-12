# Frost website delivery status — November 11, 2026

The historical website implementation and synthetic delivery package are
complete locally. Deployment is deferred at the user's request. A real-data
production launch remains dependent on source files, access to the deployment
host, and browser verification. This report
does not certify a live deployment or forecast accuracy.

## Deliverables

- Coverage-aware calculations and historical summaries, with missing years,
  provisional candidates, shared leap-year calendar, station timezone/window
  overrides, strict hourly ingestion, and a private source/cache audit.
- Versioned JSON/CSV exporter with immutable content-hashed releases, atomic
  manifest replacement, fixed public assets, and installed-wheel support.
- Buildless responsive dashboard with station/threshold/year/unit controls,
  shareable URLs, chart/table/CSV views, coverage and freshness details, and
  explicit loading, error and unavailable states. The baseline table exposes
  observed extremes and confirmed/no-event/incomplete/unavailable year counts.
  Observed event dates and
  historical typical dates/ranges are distinguished. No predictions appear.
- Independent package verification, synthetic-data guard, portable ZIP
  packaging, Nginx configuration, staging/activation/rollback instructions,
  a repeatable verification runner, and Windows/Linux CI configuration.
- Research-only forecast corrections: explicit issue/year context, audited
  predictor availability, chronological preprocessing and calibration rules,
  shared calendar/date policy, and fail-closed artifact compatibility checks.

## Plan completion and remaining gates

| Primary plan phase | Implemented and verified locally | External or data-dependent gate |
| --- | --- | --- |
| 1: calculations | Conservative event/interval rules and synthetic regressions; private audit command | Obtain original station files; verify units, provenance, reporting days, coverage and hand-check real station-years |
| 2: summaries and prediction validation | Reproducible historical quantiles and research correctness safeguards | Real chronological backtests, baseline skill, bias, interval coverage/width, calibration and publication criteria; forecasts remain excluded |
| 3: data contract | Schema 1.0.0, pinned releases, JSON/CSV, atomic publication, independent verifier and portable archive | Generate a real release only after source review |
| 4: dashboard | Working modules; 20 automated utility/DOM checks against synthetic exports | Actual desktop/mobile appearance, keyboard behavior, native downloads and recovery in a supported browser |
| 5: deployment | Nginx HTTP/TLS template and loopback checks, staging/rollback procedures | Reach proxy1, identify public domain/document root, preserve existing sites, install valid certificates, deploy and verify public HTTPS; choose schedule after source cadence is known |
| 6: delivery | Full local regressions, source and deploy documentation, verified demo ZIP and wheel | Run configured CI remotely; finish live browser, source and production checks |

## Verification evidence

The completion pass runs 225 Python tests successfully, with the two optional
Nginx tests skipped in the default run. Twenty frontend tests pass.
Both optional Nginx tests pass separately with official Windows Nginx 1.30.6.
The latter cover MIME types, cache policy, missing-file responses, denied
staging paths, manifest rollback, HTTP-to-HTTPS redirects, and TLS 1.2/1.3.
The TLS test uses a one-day ephemeral certificate trusted solely by its test
client; normal trust rejects it. It does not validate a production certificate.

Full-repository Ruff and `git diff --check` pass. Wheel checks export,
verify and package from an extracted wheel and from a fresh isolated install
using locked runtime dependencies, without importing this checkout.
ZIP packaging verifies the extracted archive before final replacement.
Malformed, private, inconsistent or mixed-release packages fail verification.
Synthetic packages require explicit `--allow-demo`.

Run the complete local checks with:

```powershell
.venv/Scripts/python.exe scripts/verify.py --nginx C:/path/to/nginx.exe
```

Omit `--nginx` when unavailable; add `--offline` when all dependency caches
are already populated. The CI workflow is committed configuration; no remote
CI result is claimed before it is pushed and executed.

Generated, Git-ignored deliverables:

| Location | Contents |
| --- | --- |
| `.tmp/verified-site/` | Complete static synthetic preview |
| `.tmp/delivery/idwx-demo-site.zip` | Verified portable preview, including JSON/CSV |
| `.tmp/delivery/wheels/` | Python wheel including frontend assets |
| `.tmp/final-verification.log` | Local verification output |
| `.tmp/baseline-verification.log` | Baseline-table follow-up verification |
| `.tmp/delivery/SHA256SUMS.txt` | Final ZIP/wheel byte checksums |

The ZIP can be extracted into a dedicated static document root. Python and
Node are unnecessary on the serving host. Synthetic notices must remain
visible; do not represent this example as actual Idaho observations.

## Precise unresolved inputs

**Source data:** The configured
`C:/Users/kerrc/historical_weather_data-1980-2022/idaho` directory is absent.
The private `.tmp/data-audit.json` is blocked with zero source files.
Targeted checks of likely user/download/document/data locations found no
matching source directory; this was not an exhaustive disk/network search.
Local audit reports contain paths/metadata and stay outside the public root.

**Browser:** The supported browser inventory is empty and its access host
reports unavailable. The earlier native Firefox attempt was stopped by the
tool's unsupported URL-policy enforcement. DOM tests exercise actual modules
but do not render layouts or establish native browser accessibility/download
behavior. No screenshot-based or real-browser verification is claimed.

**Production:** The user identified `proxy1` as the host. A noninteractive SSH
check with existing credentials and strict host-key verification failed before
connecting: `Could not resolve hostname proxy1: No such host is known.` No SSH
alias is configured in this workspace's user profile. An address/full hostname,
SSH username, site domain and weather-data path remain unknown. The user
subsequently deferred deployment so local development could continue. No remote
configuration, certificate, data upload or activation has occurred.

**Forecasts:** No real-data accuracy, chronological baseline comparison,
calibration or current-year feature-availability evidence is available.
Forecast publication stays disabled. The historical first release does not
depend on completing this subsequent research release.

## Launch sequence after the inputs arrive

1. Audit original sources privately, resolve every provenance/reporting-day
   question, rebuild caches and hand-check representative real station-years.
2. Export into a new real-data staging root, verify without `--allow-demo`,
   retain compatible real releases and package outside the document root.
3. Inspect proxy1's existing configuration and certificates, choose the
   dedicated domain/root, and stage without replacing other hosted services.
4. Check the real dashboard on desktop/mobile and keyboard, then activate
   atomically. Verify public HTTPS, MIME/cache/error behavior and rollback.
5. Set the batch cadence and retention policy from the actual source delivery
   and operating requirements. Keep research predictions out until validated.

See [deployment procedures](15_DASHBOARD_DEPLOYMENT.md),
[the data contract](14_WEB_DATA_CONTRACT.md), and
[the source audit](20_SOURCE_DATA_AUDIT.md) for commands and boundaries.
