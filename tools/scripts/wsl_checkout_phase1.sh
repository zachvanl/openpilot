#!/bin/bash
set -euo pipefail
WIN=/mnt/c/Users/yari/Documents/CommaAi/mvl-boston-openpilot
cd /root/commaai/openpilot
echo "WSL before: $(git rev-parse --abbrev-ref HEAD) $(git log -1 --oneline)"
git remote remove win 2>/dev/null || true
git remote add win "$WIN"
git fetch win crv6g-mvl-202606
git checkout -B crv6g-mvl-202606 win/crv6g-mvl-202606
git submodule update --init opendbc_repo panda
echo "WSL after: $(git rev-parse --abbrev-ref HEAD) $(git log -1 --oneline)"
git -C opendbc_repo log -1 --oneline
git -C panda log -1 --oneline
