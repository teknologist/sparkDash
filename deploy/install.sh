#!/usr/bin/env bash
#
# install.sh — put this deploy tree on every node the composer launches on.
#
# The model launchers live in the repo (single source of truth), but the composer
# starts them over ssh on nodes that do NOT have the repo checked out. Bricks in
# config/models.json therefore reference ONE absolute path that must resolve
# identically everywhere: $DEPLOY_DIR (default /home/eric/sparkDash/deploy).
#
# On the repo host that path IS this directory, so nothing is copied. For every
# other node the tree is rsync'd to the same location.
#
# Usage:
#   ./deploy/install.sh                 # install to every node in config/sparks.json
#   ./deploy/install.sh Spark2          # install to specific ssh host(s)
#   DEPLOY_DIR=/opt/sparkdash/deploy ./deploy/install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${DEPLOY_DIR:-/home/eric/sparkDash/deploy}"

hosts=("$@")
if [ ${#hosts[@]} -eq 0 ]; then
  # Derive remote hosts from the spark registry; fall back to Spark2.
  cfg="$HERE/../config/sparks.json"
  if [ -f "$cfg" ] && command -v python3 >/dev/null; then
    mapfile -t hosts < <(python3 - "$cfg" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for s in d.get("sparks", []):
    if s.get("isLocal"):
        continue
    # prefer the fabric address when present — it is what the bricks ssh to
    print(s.get("cx7Ip") or (s.get("ssh") or {}).get("host") or s.get("lanIp"))
PY
    )
  fi
  [ ${#hosts[@]} -eq 0 ] && hosts=(Spark2)
fi

echo "source     : $HERE"
echo "target dir : $DEPLOY_DIR"

if [ "$HERE" != "$DEPLOY_DIR" ]; then
  echo "[local] $HERE -> $DEPLOY_DIR"
  mkdir -p "$DEPLOY_DIR"
  rsync -a --delete "$HERE/" "$DEPLOY_DIR/"
else
  echo "[local] already at the canonical path — nothing to copy"
fi

for h in "${hosts[@]}"; do
  [ -z "$h" ] && continue
  echo "[$h] -> $DEPLOY_DIR"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "mkdir -p '$DEPLOY_DIR'"
  rsync -a --delete -e "ssh -o BatchMode=yes -o ConnectTimeout=10" "$HERE/" "$h:$DEPLOY_DIR/"
  ssh -o BatchMode=yes "$h" "chmod +x '$DEPLOY_DIR'/*.sh '$DEPLOY_DIR'/*/*.sh 2>/dev/null || true"
done

echo "done."
