#!/bin/bash
set -euo pipefail
WIN="/mnt/c/Users/yari/Documents/CommaAi/mvl-boston-openpilot"
REPO="/root/commaai/openpilot"
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"

echo "=== Sync from Windows ==="
cp "$WIN/selfdrive/controls/lib/longitudinal_planner.py" selfdrive/controls/lib/longitudinal_planner.py
cp "$WIN/selfdrive/controls/tests/test_following_distance.py" selfdrive/controls/tests/test_following_distance.py
cp "$WIN/tools/scripts/validate_city_follow_replay.py" tools/scripts/validate_city_follow_replay.py

echo "=== Build ==="
scons -f SConstruct.params -j"$(nproc)" common/params_pyx.so
scons -f SConstruct.params -j"$(nproc)" selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so

echo "=== pytest city tests ==="
python3 -m pytest selfdrive/controls/tests/test_following_distance.py -k "city_" -q

OUT="$REPO/tools/scripts/regen_output"
mkdir -p "$OUT"
PROCS="radard,plannerd"

regen_one() {
  local route="$1"
  local seg="$2"
  echo ""
  echo "=== REGEN $route seg $seg ==="
  python3 selfdrive/test/process_replay/regen.py "$route" "$seg" \
    --whitelist-procs "$PROCS" --outdir "$OUT" --dummy-dcamera
}

regen_one "987638facc544f63/00000080--0684ca20c8" 7
regen_one "987638facc544f63/00000084--485bd88c34" 0
regen_one "987638facc544f63/00000071--f2a9720142" 1

echo ""
echo "=== REGEN OUTPUT (newest first) ==="
ls -1dt "$OUT"/*/
