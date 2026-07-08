#!/usr/bin/env bash
# VolGuard collector bootstrap for a fresh Ubuntu 24.04 VPS.
# Run as root (or with sudo). Idempotent-ish: safe to re-run.
#
#   curl -fsSL <raw-url>/bootstrap.sh | sudo bash
# or copy this repo to the box and run: sudo bash bootstrap.sh
set -euo pipefail

REPO_URL="${REPO_URL:-}"          # optional: git clone URL of this repo
APP_USER="volguard"
APP_DIR="/opt/volguard"

echo "==> Creating service user '${APP_USER}'"
id -u "${APP_USER}" &>/dev/null || useradd --system --create-home --shell /bin/bash "${APP_USER}"

echo "==> Installing base packages (git, curl, rclone)"
apt-get update -y
apt-get install -y git curl rclone

echo "==> Installing uv for ${APP_USER}"
sudo -u "${APP_USER}" bash -lc 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

echo "==> Placing app in ${APP_DIR}"
mkdir -p "${APP_DIR}"
if [[ -n "${REPO_URL}" ]]; then
  if [[ ! -d "${APP_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${APP_DIR}"
  else
    git -C "${APP_DIR}" pull --ff-only
  fi
else
  echo "    REPO_URL not set — copy the repo into ${APP_DIR} manually (rsync/scp)."
fi
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"

echo "==> uv sync"
sudo -u "${APP_USER}" bash -lc "cd ${APP_DIR} && uv sync"

echo "==> Installing systemd units"
install -m 644 "${APP_DIR}/src/volguard/collector/deploy/volguard-collector.service" /etc/systemd/system/
install -m 644 "${APP_DIR}/src/volguard/collector/deploy/volguard-sync.service" /etc/systemd/system/
install -m 644 "${APP_DIR}/src/volguard/collector/deploy/volguard-sync.timer" /etc/systemd/system/
systemctl daemon-reload

cat <<'EOF'

==> Bootstrap complete. Remaining manual steps:

  1. Configure rclone remote named 'volguard' (Cloudflare R2 or Backblaze B2):
       sudo -u volguard rclone config
     For R2: type=s3, provider=Cloudflare, set access_key_id/secret,
     endpoint = https://<account_id>.r2.cloudflarestorage.com

  2. Create the bucket (e.g. 'volguard-collector') in the R2/B2 dashboard.

  3. Start the collector and enable the daily sync:
       sudo systemctl enable --now volguard-collector.service
       sudo systemctl enable --now volguard-sync.timer

  4. Verify:
       sudo systemctl status volguard-collector
       journalctl -u volguard-collector -f
EOF
