#!/usr/bin/env python3
"""Analyze experimental / DEC / e2e vs stop behavior on a route."""
import sys
import math
from collections import Counter
from openpilot.tools.lib.logreader import LogReader, ReadMode

def capget(o, n, d=None):
  try:
    return getattr(o, n)
  except Exception:
    return d

def f(o, n, d=float("nan")):
  try:
    return float(getattr(o, n))
  except Exception:
    return d

route = sys.argv[1]
seg_end = int(sys.argv[2]) if len(sys.argv) > 2 else 20

dec_state_ctr = Counter()
exp_on = 0
exp_off = 0
dec_active_on = 0
lp_sp_frames = 0

# Low speed + shouldStop false + weak braking (possible stoplight miss)
suspicious = []

for seg in range(0, seg_end):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    break

  latest = {}

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9

    if typ == "selfdriveState":
      sds = msg.selfdriveState
      latest["exp"] = bool(capget(sds, "experimentalMode", False))
      latest["active"] = bool(capget(sds, "active", False))

    elif typ == "carState":
      cs = msg.carState
      latest["v"] = f(cs, "vEgo")
      latest["brake"] = bool(capget(cs, "brakePressed", False))

    elif typ == "modelV2":
      md = msg.modelV2
      latest["e2e_a"] = f(capget(md, "action"), "desiredAcceleration")
      latest["should_stop"] = bool(capget(capget(md, "action"), "shouldStop", False))

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      latest["lead"] = bool(l0 and capget(l0, "status", False))

    elif typ == "longitudinalPlan":
      lp = msg.longitudinalPlan
      latest["a_target"] = f(lp, "aTarget")
      latest["should_stop_plan"] = bool(capget(lp, "shouldStop", False))
      latest["src"] = str(capget(lp, "longitudinalPlanSource", ""))

    elif typ == "longitudinalPlanSP":
      lp_sp_frames += 1
      lpsp = msg.longitudinalPlanSP
      dec = capget(lpsp, "dec")
      if dec:
        st = str(capget(dec, "state", ""))
        dec_state_ctr[st] += 1
        if bool(capget(dec, "active", False)):
          dec_active_on += 1
      if latest.get("exp"):
        exp_on += 1
      else:
        exp_off += 1

      v = latest.get("v", 0)
      active = latest.get("active", False)
      if active and v < 8.0 and v > 0.5:
        ss = latest.get("should_stop", False)
        ss_plan = latest.get("should_stop_plan", False)
        a_t = latest.get("a_target", 0)
        e2e_a = latest.get("e2e_a", float("nan"))
        if not ss_plan and a_t > -0.8 and not latest.get("lead", True):
          suspicious.append((t, seg, v * 2.237, a_t, e2e_a, ss, st if dec else "?"))

print("=" * 70)
print(f"E2E / DEC / experimental analysis: {route} segs 0-{seg_end-1}")
print("=" * 70)
print(f"\nlongitudinalPlanSP frames: {lp_sp_frames}")
print(f"DEC state distribution: {dict(dec_state_ctr)}")
print(f"DEC.active frames (approx): {dec_active_on}")
ratio_blended = dec_state_ctr.get("blended", 0) / max(lp_sp_frames, 1)
print(f"blended fraction: {ratio_blended*100:.1f}%")
print(f"experimentalMode True frames (paired with planSP): {exp_on}")
print(f"experimentalMode False: {exp_off}")

print("\n--- Interpretation ---")
if lp_sp_frames == 0:
  print("No longitudinalPlanSP in qlog (older segment or logging); check selfdriveState only.")
else:
  if dec_state_ctr.get("acc", 0) > dec_state_ctr.get("blended", 0) * 3:
    print("DEC spends most time in ACC mode.")
    print("With DynamicExperimentalControl ON, is_e2e = experimental AND (mode==blended).")
    print("So e2e red-light braking is OFF during ACC — only MPC runs (no model shouldStop min).")

print(f"\nSample 'city speed, no plan shouldStop, weak aTarget, no lead' frames: {len(suspicious)}")
for row in suspicious[:12]:
  print(f"  t={row[0]:.0f} seg={row[1]} v={row[2]:.0f}mph aTarget={row[3]:+.2f} e2e_desA={row[4]:+.2f} model_shouldStop={row[5]} dec={row[6]}")
print()
