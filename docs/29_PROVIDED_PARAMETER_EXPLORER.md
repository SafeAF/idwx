# Additional parameters in supplied units

Implemented locally November 22, 2026. This follows the owner's instruction to
bring in the other parameters and keep values as supplied when their time
interval is unspecified. It extends the [workspace foundation](28_MULTILOCATION_IMPLEMENTATION_STATUS.md).

The current SQLite database still provides the existing daily Twin Falls
products. The additional fields come from the existing, fingerprinted hourly
ERA5/Oikolab archive. No new weather collection, database migration, model
training, production deployment or persistent service was performed.

## Explorer behavior

Choose **Additional parameters · supplied units**, then select a location and
parameter. All 27 catalog parameters are available at all 21 allowlisted
locations. Other than the existing Twin Falls label, coordinates are used as
location names. These are reanalysis points, not 21 independent observing
stations or providers.

The additional view displays unaggregated source samples with:

- Original supplied numeric values and unit strings. Pressure stays in **Pa**;
  fractions stay in **0-1**; radiation stays in **W/m^2**. Snow water equivalent
  remains water equivalent. No hPa, percentage, energy or physical snow-depth
  conversion is applied.
- An explicit **unspecified value interval**. The source timestamps have an
  hourly sampling cadence; that does not establish each variable's measurement
  or accumulation interval. No rate, mean, sum, daily total or integration is
  inferred from that cadence.
- UTC timestamps and a separately labeled fixed-offset reporting time. Up to
  31 inclusive UTC days can be selected per graph; longer selections are
  explicitly limited. Monthly partitions load only for that bounded selection.
- Exact graph/table/CSV agreement. Missing values stay unavailable and blank
  in CSV. Finite values outside screening bounds remain visible and downloadable,
  with a quality note; they are not clamped or removed. Wind directions use
  unconnected points rather than an arithmetic direction mean or wraparound line.
- Independent selections in one to three graph cards, plus workspace URLs,
  studies and backup/import. Parameter presets retain archive-only locations
  and supplied-value semantics. Parameter graphs can link UTC date ranges and
  exact UTC inspection timestamps. Mixed UTC/local/overlay calendars remain
  independent. Location links explain when a location has no daily products.

Existing daily temperature, frost and reviewed precipitation views retain
previous meanings and values. Additional archived parameters stop in 2022;
that cutoff is displayed separately from the later Twin Falls daily refresh.
A parameter-only location is not offered as if it already had published frost
or other derived daily products.

## Rebuild from existing local data

```sh
.venv/bin/python -m idwx data build-parameters \
  --source-root ~/source/idaho-weather-data/archive \
  --manifest configs/climate_sources.yml \
  --output data_cache/parameters

.venv/bin/python -m idwx export-web \
  --parameter-snapshot data_cache/parameters/current \
  --output .tmp/parameter-site

.venv/bin/python -m idwx verify-web .tmp/parameter-site
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/parameter-site
```

Open `http://127.0.0.1:8765/explore.html?view=parameters`. Stop the temporary
server after review. `--station twin_falls` on `build-parameters` can explicitly
limit a development import; the default imports the complete allowlist.
The established export without a parameter snapshot remains available for
legacy-only publications and does not silently discover private inputs.

The importer streams sources and stages the complete snapshot before changing
`data_cache/parameters/current`. All source filenames, fingerprints, coordinates,
model identity, offsets, timestamp continuity and endpoints must pass the
existing full-file identity audit. Individual screening findings do not block
as-supplied viewing. Malformed/nonfinite unrepresentable values or identity
failures stop the import and preserve the previous snapshot pointer.

Private snapshot contents are normalized, allowlisted JSON month matrices and
a content-addressed catalog, under an immutable snapshot directory. No private
filesystem paths, raw CSVs, SQLite databases or audit reports enter the static
package. Source bytes and the tracked daily database remain unchanged.

## Publication and refresh compatibility

A publication with supplied samples uses **schema 2.0.0** and adds a pinned
`parameters_url`. Existing daily structures retain their meanings. The release
identity binds the parameter snapshot ID; its catalog binds source fingerprints,
units, UTC ranges, sample coverage, quality summaries and every monthly chunk's
SHA-256. This avoids putting thousands of leaf hashes in the top manifest.
The browser verifies catalog and sample-file bytes using SHA-256 before using
those values. It requires HTTPS or localhost browser cryptography support.

The package verifier independently recomputes sample counts, continuity,
missing/out-of-range counts and extrema. It rejects inconsistent summaries even
when file hashes and snapshot IDs have been updated. It also checks actual
JavaScript module dependencies. Retained 1.0/1.1/1.2 data releases continue to
verify; existing release contents are not rewritten.

The daily updater still fetches only its established Twin Falls inputs. For an
expanded active site, its private configuration must include:

```yaml
parameter_snapshot: /absolute/private/path/to/parameters/<snapshot-id>
```

Use the reviewed immutable snapshot directory. The updater resolves and pins
that directory before work, includes the snapshot ID in its no-op decision and
receipt, and preserves all archive locations and parameters in a later build.
An absent or invalid required snapshot stops publication before a provider
request; it cannot publish a reduced site. Identical immutable monthly files
are reused through verified hard links within retained publications, rather
than duplicating the entire archive on every daily refresh. Existing publication
locking, health checks and rollback remain in effect.

## Real-data evidence

| Property | Verified local result |
| --- | --- |
| Source locations / parameters | 21 / 27 |
| Archived timestamp rows | 7,915,824 |
| UTC range | 1980-01-01 00:00 through 2022-12-31 23:00 |
| Monthly files | 10,836 |
| Missing supplied values | 184,655, retained as null |
| Finite values outside screening bounds | 2,645 relative-humidity values, retained unchanged |
| Snapshot ID | `95f07fe573f1e19c6422f955f2111d705cd6573755895f27982719a1f4c1a4c6` |
| Expanded release ID | `68e728664baabfcd26c3a13d86bb55d7d126475394b0fb7a98521926b43422bb` |
| Snapshot JSON bytes | 1,224,823,251 |
| Catalog gzip bytes / default Twin Falls month gzip bytes | 557,654 / 37,596 |

Compressed sizes are file inspections, not measured browser throughput or heap
usage. The archive is loaded lazily; opening a graph does not download the full
1.22 GB snapshot.

An independent standard-library `csv`/`datetime` comparison checked January and
July 2000 at Twin Falls (UTC-07) and the northern coordinate
46.694995, -116.942834 (UTC-08). **80,352 values** matched their original source
values and header units exactly, including missing values, winter/snow and
summer/radiation periods. Reproduce:

```sh
.venv/bin/python scripts/check_parameter_source.py \
  --snapshot data_cache/parameters/current \
  --source-root ~/source/idaho-weather-data/archive \
  --station twin_falls --station point_46_694995_116_942834 \
  --month 2000-01 --month 2000-07
```

The full Python suite passed **276 tests**, with two optional Nginx tests skipped.
The full frontend suite passed **77 tests**, including all 27 parameter choices,
original Pa/radiation/fraction values, CSV parity, independent archive locations,
study persistence, presets and checksum failure/retry. Subsequent targeted
checks cover the final snapshot pinning and unchanged-file reuse. A simulated
later Twin Falls refresh preserves both fixture parameter locations and every
parameter, reuses their immutable files and refuses a missing required snapshot.

The real static publication and portable wheel were built and verified locally.
Artifact hashes and the final delivery receipt are recorded with the local
package. The browser tool returned **“The browser access host is unavailable”**,
so no native screenshots, visual acceptance or native download checks are
claimed. Nothing was deployed.

This delivery completes supplied-value access for the additional parameters.
Reviewed daily reductions, radiation energy, wind roses, paired/trend analyses
and the other advanced work in the broader plan remain separate. Missing time
intervals are not a prerequisite for displaying the original supplied samples.


## Final delivery artifacts

Source base on `main`: `b2b0a0e6dcb9c8b433802e15318cfd9e0bcbfe56`.
Implementation changes remain local and uncommitted.

- Static ZIP: `.tmp/delivery/idwx-provided-parameters.zip` (417,989,433 bytes).
  SHA-256: `6b65ca39012cc863bed774d899308d6ed50de747dcceac5b80acfff064582ece`.
- Wheel: `.tmp/parameter-wheels/idwx-0.1.0-py3-none-any.whl`.
  SHA-256: `ae002e341a1952b9948f313b4aaec8a3dfad1f8e21d88509757673f3b770baf7`.
- Receipt: `.tmp/delivery/idwx-provided-parameters-receipt.json`, recording the
  source-file digests, immutable data/snapshot identities, artifact hashes and
  verification limits.

The final ZIP was extracted and verified by the package tool. Every shipped
runtime asset matches the finalized source byte-for-byte, as do the wheel's
Python modules and web assets. The tracked SQLite database matches its base
revision. These artifact checks do not constitute production deployment or
native visual acceptance.
