#!/usr/bin/env bash
# Install the local git hooks into .git/hooks (which git does not track).
# Run once per clone:  bash scripts/install-hooks.sh
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HOOK="$REPO_ROOT/.git/hooks/post-commit"

cat > "$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
# Restart the live agent after a commit on the deploy branch.
#
# /home/max/app is both this checkout and the systemd unit's WorkingDirectory, so the
# committed files are already the running code — the restart just reloads them.
#
# Managed by scripts/install-hooks.sh — edit there, not here.
set -euo pipefail

DEPLOY_BRANCH=master
SERVICE=max-agent

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "$DEPLOY_BRANCH" ]; then
    echo "post-commit: on '$BRANCH', not '$DEPLOY_BRANCH' — skipping $SERVICE restart."
    exit 0
fi

echo "post-commit: restarting $SERVICE…"
if ! sudo -n systemctl restart "$SERVICE"; then
    echo "post-commit: RESTART FAILED — commit is saved but $SERVICE still runs the old code." >&2
    echo "post-commit: recover with 'sudo systemctl restart $SERVICE' and check 'journalctl -u $SERVICE -n 50'." >&2
    exit 0
fi

sleep 1
if [ "$(systemctl is-active "$SERVICE")" = "active" ]; then
    echo "post-commit: $SERVICE is active."
else
    echo "post-commit: WARNING — $SERVICE is not active after restart." >&2
    systemctl --no-pager -n 20 status "$SERVICE" >&2 || true
fi
HOOK_EOF

chmod +x "$HOOK"
echo "Installed $HOOK"
