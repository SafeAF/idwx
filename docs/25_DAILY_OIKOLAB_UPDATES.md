# Daily Oikolab updates

The owner authorized daily historical-data updates on November 21, 2026.
This extends the earlier static-only deployment authorization with a bounded
batch job on `proxy1`. Nginx continues to serve static HTML/JS/JSON/CSV; there
is no persistent application/API server, model training, or new station.

## Schedule and runtime

`idwx-daily-update.timer` runs at **04:00 America/Denver daily**, within systemd's
one-minute timer accuracy. The timezone follows daylight saving time. Persistent
scheduling catches a missed run after the host returns. A one-shot service
uses a pinned wheel and locked dependencies in a private virtual environment.
It does not pull source updates or upgrade dependencies on the daily schedule.
See the [systemd timer reference](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html).

| Location on proxy1 | Purpose |
| --- | --- |
| `/etc/idwx/oikolab.env` | Root-only `OIKOLAB_API_KEY`; not Nginx configuration |
| `/srv/idwx/automation/releases/<runtime-id>/` | Pinned runtime, configuration, original seed inputs, full source/wheel identity |
| `/srv/idwx/automation/runtime` | Selected runtime symlink |
| `/srv/idwx/automation/state/snapshots/<run-id>/` | Private refreshed SQLite, reviewed precipitation, source archive, receipt |
| `/srv/idwx/automation/state/current` | Last successfully published input snapshot |
| `/srv/idwx/publish.lock` | Shared exclusive publication lock |
| `/srv/idwx/builds/daily-<run-id>/` | Verified public static build |
| `/srv/idwx/current` | Public site symlink |

Private runtime and input directories are outside the document root. Secret
values never enter public assets, command arguments, deployment records, or
logs. The service reads its environment file without a shell. The checkout's
`.env` is separate, private and Git-ignored; it is not needed on the server.

## Refresh and publication

1. Acquire the publication lock and verify the existing public build.
2. Read Oikolab ERA5 metadata. If both temperature and reviewed precipitation
   already reach the latest complete reporting day, and runtime/site identities
   match the last successful snapshot, skip downloading and publishing.
3. Otherwise download continuous hourly Twin Falls temperature and total
   precipitation from the oldest required date to the published endpoint.
   The first catch-up includes a complete boundary month for independent audit.
4. Stage updates to a copy of SQLite and the reviewed precipitation supplement.
   Validate coordinates, ERA5 model, fixed UTC−07 offset, continuous hours,
   finite nonnegative precipitation, and independent 24-hour coverage. Retain
   compressed raw source bytes privately for reproducibility, never in Git or
   public builds. No SQLite schema migration is added by this feature.
5. Compare complete-month precipitation totals using an independent
   `csv`/`datetime` calculation, separate from the pandas daily reducer.
6. Export static data and assets; retain previous immutable data via hard links.
   Independently verify the full new publication, including retained releases.
7. Atomically switch the site, then check the manifest through the existing
   Nginx HTTPS vhost using trusted TLS and loopback SNI. If that check fails,
   restore the previous site. Advance the private input snapshot only after
   that check succeeds. Provider/export/verification errors preserve the live
   site and previously successful input state, and fail the systemd service.

The initial precipitation supplement stays version 1 in Git. Runtime snapshots
use its backward-compatible private version 2, adding fingerprinted
`refresh_sources` (`sha256`, `start_utc`, `end_utc`) to the content digest. Every
fragment must match a SQLite `source_refreshes` provenance record before its
amounts can be published. The public schema remains 1.2.0: daily units, method,
coverage rules and fields are unchanged. Public `source_sha256` identifies the
original source anchor; subsequent fragments are bound in private provenance
and their resulting public values in the release identity. The published
precipitation cutoff advances only with complete reviewed reporting days.

Oikolab ERA5 remains delayed historical reanalysis, not live observations or
a forecast. The live NWS/Open-Meteo section is separate. Soil fertility is still
not assessed. The job does not recompute model predictions or change thresholds.
[Oikolab authentication/usage documentation](https://docs.oikolab.com/authentication/)
describes API-key authentication and metered data units. Its documented
`/account` endpoint returned HTTP 410 during setup, so account quota was not
confirmed there; weather request success is checked separately.

## Operations and rollback

```sh
sudo systemctl status idwx-daily-update.timer
sudo systemctl list-timers idwx-daily-update.timer
sudo journalctl -u idwx-daily-update.service -n 30 --no-pager
sudo systemctl start idwx-daily-update.service
```

A manual start uses the same lock and checks as the timer. Other manual
publishers must also acquire `/srv/idwx/publish.lock`. Do not edit the active
snapshot or files inside a public immutable release.

To stop scheduling, run `sudo systemctl disable --now idwx-daily-update.timer`.
That does not terminate an already-running service: wait for it to finish, or
stop it before rollback. Under the publication lock, restore both `/srv/idwx/current`
and `/srv/idwx/automation/state/current` to the preceding paths recorded in the
successful snapshot receipt, using temporary symlinks and atomic `mv -Tf`.
For the first-run rollback, restore the original public site and remove only
the new state symlink, so a later authorized run uses the seed again. Recheck
the served manifest after rollback. No Nginx reload is required.

Builds, state snapshots and source fragments are retained for rollback; this
increment performs no automatic pruning. Monitor available disk space before
setting a separate retention policy. The batch job runs only on proxy1; no
second updater is scheduled on the workstation. The checked-in seed database
and precipitation archive are unchanged by production refreshes.

## Verification

Tests exercise successful publication, independent reporting-day/month totals,
no-download current runs, provenance binding, rejected offsets/missing hours,
provider failures, post-promotion HTTPS-check failures and rollback, verification
without provider access, and exclusive locking. Local delivery and deployed
run evidence are recorded separately; unit tests do not establish production
activation or native browser visual acceptance.


## Activated November 21, 2026 (America/Denver)

The first production run succeeded, published complete temperature and reviewed
precipitation through **2026-11-15**, and retained the former public build.
The provider endpoint was 2026-11-16 15:00 UTC. Independent complete-month
references included December 2022 (61.03 mm / 744 hours), January 2023
(39.72 mm / 744 hours), and leap February 2024 (52.95 mm / 696 hours).
A second service run succeeded with `already_current` and skipped weather
download/publication. The timer is enabled and active; its next scheduled run
is November 22 at 04:00 America/Denver (11:00 UTC).

Full local Python checks: **251 passed, 2 optional Nginx tests skipped**.
The actual production Nginx configuration test passed. All 23 Nginx file
fingerprints were unchanged, and weather.kerrco.us, kerrco.us and order.kerrco.us
returned HTTPS 200 before/after. Twelve public HTTP checks confirmed the new
data, explorer links, asset headers and private-path denial. The installed
runtime verified the publication and ZIP extraction/reverification, including
the retained 1.0.0 release. Native browser visual review was not performed.

Production release:
`7bff6ee9d7170955e41a0461442903b7cb6554e6394b9211520af9a6feb9e02c`.
Runtime: `37ba3a9-daily-fc87665008c7`, built from `37ba3a9` plus the tested
uncommitted updater changes. Wheel SHA-256:
`fc87665008c75ade2fda397a1d1e08b12ecd051222b7fe405cba4d9967d5c5f6`.
Published static ZIP SHA-256:
`6df2efd8e80478f44d2a6a238d966c72cb94e18035d27f96b8255801dccb9c01`.
The complete local receipt is `.tmp/idwx-daily-delivery/verification-receipt.json`;
the server snapshot contains its publication/source receipt. The private
SafeAF/deployabot repository records exact unit content, health and rollback.
