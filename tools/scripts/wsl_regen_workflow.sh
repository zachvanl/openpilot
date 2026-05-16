#!/bin/bash
set -euo pipefail

WIN_REPO="/mnt/c/Users/yari/Documents/CommaAi/mvl-boston-openpilot"
REPO="/root/commaai/openpilot"
cd "$REPO"
. .venv/bin/activate
export PYTHONPATH="$REPO"

echo "=== Sync control files from Windows ==="
cp "$WIN_REPO/selfdrive/controls/lib/longitudinal_planner.py" selfdrive/controls/lib/longitudinal_planner.py
cp "$WIN_REPO/selfdrive/controls/radard.py" selfdrive/controls/radard.py
if [ -f "$WIN_REPO/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py" ]; then
  cp "$WIN_REPO/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py" selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py
fi

echo "=== Build MPC + params ==="
scons -j"$(nproc)" common/params_pyx.so
scons -j"$(nproc)" selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so

echo "=== Verify imports ==="
python3 -c "from openpilot.selfdrive.controls.lib.longitudinal_planner import city_closing_brake_active; print('OK', city_closing_brake_active)"
python3 -c "from openpilot.selfdrive.test.process_replay.regen import regen_and_save; print('regen OK')"

ROUTES=(
  "987638facc544f63/00000080--0684ca20c8:7"
  "987638facc544f63/0000007a--e54be11ada:1"
  "987638facc544f63/00000071--f2a9720142:1"
)
PROCS="radard,plannerd"
OUTBASE="$REPO/tools/scripts/regen_output"
mkdir -p "$OUTBASE"

for item in "${ROUTES[@]}"; do
  route="${item%%:*}"
  seg="${item##*:}"
  echo ""
  echo "=== REGEN $route seg $seg ==="
  python3 selfdrive/test/process_replay/regen.py "$route" "$seg" \
    --whitelist-procs "$PROCS" \
    --outdir "$OUTBASE" \
    --dummy-dcamera || echo "REGEN FAILED: $route $seg"
done

echo "=== Done. Output under $OUTBASE ==="
