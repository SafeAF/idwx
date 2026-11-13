# Codex instructions — idwx

## Current authorized task (2026-11-13)

Deploy the **existing verified, real-data, static idwx website** to **https://weather.kerrco.us** on **proxy1**. No model training, new weather data collection, API server, or other deployment is part of this task.

### Production access already provisioned

- Use SSH as **`keystone@proxy1`**, with the **existing** Steakies key `~/.ssh/id_ed25519_deployabot` (or an unlocked SSH agent serving that identity).
- The owner previously verified `keystone` SSH and `sudo -n true && sudo -n -l` on **both hosts**. Verify current access using `ssh -i ~/.ssh/id_ed25519_deployabot -o IdentitiesOnly=yes keystone@proxy1 'id; sudo -n -l'`.
- `codex-audit` is strictly for read-only inventory; **do not use it for deployment**. Do **not** create additional users, keys, sudoers rules, deployment groups, or prompt for administrator passwords. If Keystone access is unexpectedly unavailable, report the precise error.
- `app1` is **out of scope**. Do not change it.

### Local artifact to deploy

- The preceding Codex session reported `/home/sam/source/idwx/.tmp/delivery/weather-kerrco-us.zip` as a verified, real-data deployment ZIP and `/home/sam/source/idwx/.tmp/weather.kerrco.us/nginx.conf` as a prepared domain-specific configuration.
- **These are Git-ignored local artifacts, not GitHub-tracked files.** Check that they exist in the present checkout; verify their full SHA-256 against the local receipt and independently confirm they are not demo/synthetic, not partially generated, and include the complete site and data manifest. The conversation only includes an abbreviated checksum; do not treat it as sufficient validation. If absent or inconsistent, regenerate and verify using `README.md` and `docs/15_DASHBOARD_DEPLOYMENT.md` before deploying.
- The website is historical **ERA5/Oikolab Twin Falls frost climatology**, *not* real-time weather forecasts or independently validated future-event predictions. Preserve these distinctions in the page.

### Narrow production deployment steps authorized

1. From Steakies, check the existing Nginx virtual hosts, active HTTPS names/certificates, and `weather.kerrco.us` resolution. Only the **weather.kerrco.us** virtual host and `/srv/idwx` tree may be created or changed. Preserve every other vhost, including `kerrco.us` and `order.kerrco.us`.
2. Stage the verified site under a new `/srv/idwx/builds/<unique-id>` directory, not within the default document root. Validate manifest, static asset types and data files. Do not serve the ZIP, Python source, SQLite or raw data. Use a `/srv/idwx/current` symlink to promote a validated release; retain prior builds for rollback.
3. Review the prepared `.tmp/weather.kerrco.us/nginx.conf` against `deploy/nginx.conf` and actual Debian Nginx paths. Prepare a dedicated HTTP-only ACME/redirect-ready vhost before certificate issuance. Test `sudo nginx -t` before any reload.
4. Obtain a Let's Encrypt certificate for **weather.kerrco.us only** using the existing Certbot mechanism after verifying the challenge path/DNS and current unrelated renewal state. A prior general Certbot failure was observed: do not run broad renewals or edit unrelated vhosts/certificates to fix it. If the domain-specific certificate cannot be issued safely, stop and report the precise blocker.
5. Add HTTPS only after a valid domain certificate exists. Validate `nginx -t`, reload Nginx, and ensure correct SNI, trusted certificate, HTTP -> HTTPS redirect, HTML/JS/JSON/CSV MIME handling, manifest and immutable data caching, 404/denied hidden paths, UI controls and browser behavior.
6. Verify `https://kerrco.us`, `https://order.kerrco.us`, and other known active vhosts still behave as before. Keep backups of changed Nginx files and the previous `current` target; on failed checks, safely restore the prior state.
7. Record the source revision, full artifact digest, exact non-secret config diff, health checks, timestamp, and rollback procedure in the private **SafeAF/deployabot** repository. Commit reviewed repo documentation directly to `main` where appropriate. Do not commit generated ZIPs, secrets, machine-specific keys, or raw databases.

### Boundaries

No `app1` changes, no unrelated TLS repairs, DNS or firewall edits, no SQL migrations, no modifications to ActiveFundPerf or Salmon Scheduler, and no persistent Python/Node/Rails service. Never paste SSH private keys, passwords, env dumps, or credential values into output. This task ends after **weather.kerrco.us** is deployed and verified, or a specific blocker is documented.

The owner prefers **main only**. Do not create a branch or PR unless explicitly requested. Preserve existing local uncommitted work; do not force push, clean or hard-reset.
