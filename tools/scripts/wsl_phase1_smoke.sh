#!/bin/bash
set -euo pipefail
cd /root/commaai/openpilot
git fetch win crv6g-mvl-202606
git reset --hard win/crv6g-mvl-202606
. .venv/bin/activate
export PYTHONPATH=/root/commaai/openpilot
export LD_LIBRARY_PATH=/root/commaai/openpilot/.venv/lib/python3.12/site-packages/acados/install/lib
python - <<'PY'
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  is_city_fast_approach, cap_v_cruise_city, CITY_ESTABLISHED_FOLLOW_TIME,
)
class L:
  def __init__(self, s, d, v):
    self.status = s
    self.dRel = d
    self.vLead = v
    self.aLeadK = 0.0
s = L(True, 40.0, 0.0)
v = 40 * 0.44704
assert is_city_fast_approach(s, v, 0.0)
c = cap_v_cruise_city(30.0, s, v, 0.0, 0.0)
assert c < v and c <= 12
assert cap_v_cruise_city(20.0, s, 12.0, 0.0, CITY_ESTABLISHED_FOLLOW_TIME) == 20.0
print("city smoke OK", c)
PY
python -m pytest selfdrive/controls/tests/test_following_distance.py -k city_ -q -o addopts= --tb=line
