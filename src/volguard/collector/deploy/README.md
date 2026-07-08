# VolGuard Collector — VPS Deployment

The live collector polls Deribit every 5 minutes for BTC option quotes (book
summary + tickers for the most liquid instruments + index price) and writes
date-partitioned NDJSON snapshots. A daily systemd timer syncs completed files
to object storage (Cloudflare R2 or Backblaze B2) via `rclone`, which you then
pull to your laptop for analysis.

Deploy this in **week 1** — the longer it runs, the denser your recent-history
quote surfaces and the intraday stretch-goal dataset.

## Recommended setup

- **VPS:** Hetzner CX22 (2 vCPU / 4 GB / 40 GB, ~€3.79/mo). Any Ubuntu 24.04
  box works — the code is provider-agnostic. A $5 DigitalOcean/Vultr droplet is
  an equivalent drop-in.
- **Storage:** Cloudflare R2 free tier (10 GB, zero egress fees — free to pull
  the archive down repeatedly). Backblaze B2 free tier also works.

Data volume is ~50 MB/day compressed, so the free tiers last months.

## One-time setup

1. Create the VPS (Ubuntu 24.04), note its IP, and SSH in as root.

2. Get the repo onto the box, then run the bootstrap:

   ```bash
   # option A: bootstrap clones for you
   sudo REPO_URL=<your-git-url> bash /path/to/bootstrap.sh
   # option B: rsync the repo to /opt/volguard first, then
   sudo bash /opt/volguard/src/volguard/collector/deploy/bootstrap.sh
   ```

   This creates a `volguard` service user, installs `uv` + `rclone`, runs
   `uv sync`, and installs the systemd units.

3. Configure the rclone remote (named `volguard`) and create the bucket — see
   the printed instructions at the end of bootstrap. For R2:
   `type=s3, provider=Cloudflare, endpoint=https://<account_id>.r2.cloudflarestorage.com`.

4. Start it:

   ```bash
   sudo systemctl enable --now volguard-collector.service
   sudo systemctl enable --now volguard-sync.timer
   ```

## Operating

```bash
sudo systemctl status volguard-collector      # is it running?
journalctl -u volguard-collector -f           # live logs
systemctl list-timers volguard-sync.timer     # next sync time
```

## Pulling data to your laptop

```bash
rclone copy volguard:volguard-collector/ticker_snapshots ./data/raw/ticker_snapshots
```

## Decommission (end of project)

```bash
sudo systemctl disable --now volguard-collector.service volguard-sync.timer
# final sync, then destroy the VPS from the provider dashboard.
```

## Local fallback (no VPS)

`uv run volguard collect` runs the same poller on your laptop. Windows Task
Scheduler can keep it alive across reboots — you accept gaps when the machine
sleeps. Combined with the Tardis free days this is a documented fallback.
