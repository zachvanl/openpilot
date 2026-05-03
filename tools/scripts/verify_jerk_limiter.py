#!/usr/bin/env python3
"""
Verify that the universal jerk limiter would have smoothed
the worst source transitions in the latest drive.
"""
import sys
import math
import numpy as np
from collections import defaultdict
from openpilot.tools.lib.logreader import LogReader, ReadMode

def capget(obj, name, default=None):
  try:
    return getattr(obj, name)
  except Exception:
    return default

def f(obj, name, default=float("nan")):
  try:
    return float(getattr(obj, name))
  except Exception:
    return default

route = sys.argv[1]
seg_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
seg_end = int(sys.argv[3]) if len(sys.argv) > 3 else 30

OUTPUT_DECEL_JERK_LIMIT = -3.0
OUTPUT_DECEL_EMERGENCY_DIST = 4.0
DT = 0.05

prev_a_target = None
simulated_output = None
sds_latest = {}
cs_latest = {}
radar_latest = {}
engaged_since = 0

raw_jumps = []
smoothed_jumps = []
raw_a_series = []
smoothed_a_series = []
times = []

for seg in range(seg_start, seg_end):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9

    if typ == "carState":
      cs = msg.carState
      cs_latest = {"v_ego": f(cs, "vEgo"), "standstill": bool(capget(cs, "standstill", False))}

    elif typ == "selfdriveState":
      sds = msg.selfdriveState
      sds_latest = {"active": bool(capget(sds, "active", False))}

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      if l0 and capget(l0, "status", False):
        radar_latest = {"dRel": f(l0, "dRel"), "status": True}
      else:
        radar_latest = {"status": False}

    elif typ in ("longitudinalPlan", "longitudinalPlanSP"):
      if typ == "longitudinalPlanSP":
        lp = msg.longitudinalPlanSP
      else:
        lp = msg.longitudinalPlan
      a_target = f(lp, "aTarget")

      active = sds_latest.get("active", False)
      if not active or cs_latest.get("standstill", True):
        engaged_since = 0
        prev_a_target = None
        simulated_output = None
        continue

      engaged_since += 1
      if engaged_since < 50:
        prev_a_target = a_target
        simulated_output = a_target
        continue

      if prev_a_target is not None and simulated_output is not None:
        raw_jump = a_target - prev_a_target
        raw_jumps.append(raw_jump)

        d_rel = radar_latest.get("dRel", 40.0) if radar_latest.get("status") else 40.0
        emergency = radar_latest.get("status", False) and d_rel < OUTPUT_DECEL_EMERGENCY_DIST

        if a_target < simulated_output and not emergency:
          min_a = simulated_output + OUTPUT_DECEL_JERK_LIMIT * DT
          new_output = max(a_target, min_a)
        else:
          new_output = a_target

        smoothed_jump = new_output - simulated_output
        smoothed_jumps.append(smoothed_jump)

        raw_a_series.append(a_target)
        smoothed_a_series.append(new_output)
        times.append(t)

        simulated_output = new_output
      else:
        simulated_output = a_target

      prev_a_target = a_target

raw_jumps = np.array(raw_jumps)
smoothed_jumps = np.array(smoothed_jumps)

print("=" * 70)
print(f"UNIVERSAL JERK LIMITER SIMULATION: {route}")
print("=" * 70)

print(f"\nTotal engaged plan cycles analyzed: {len(raw_jumps)}")

# Deceleration jumps only (negative direction)
decel_raw = raw_jumps[raw_jumps < -0.1]
decel_smooth = smoothed_jumps[raw_jumps < -0.1]

print(f"\nDeceleration events (raw jump < -0.1 m/s^2): {len(decel_raw)}")
if len(decel_raw) > 0:
  print(f"\nRaw jumps (without limiter):")
  print(f"  Mean:   {np.mean(decel_raw):+.3f} m/s^2")
  print(f"  P95:    {np.percentile(decel_raw, 5):+.3f} m/s^2")
  print(f"  Worst:  {np.min(decel_raw):+.3f} m/s^2")
  harsh_raw = np.sum(decel_raw < -0.3)
  print(f"  Harsh (< -0.3): {harsh_raw}")

  print(f"\nSmoothed jumps (with limiter):")
  print(f"  Mean:   {np.mean(decel_smooth):+.3f} m/s^2")
  print(f"  P95:    {np.percentile(decel_smooth, 5):+.3f} m/s^2")
  print(f"  Worst:  {np.min(decel_smooth):+.3f} m/s^2")
  harsh_smooth = np.sum(decel_smooth < -0.3)
  print(f"  Harsh (< -0.3): {harsh_smooth}")

  print(f"\nImprovement:")
  print(f"  Harsh jumps eliminated: {harsh_raw - harsh_smooth} / {harsh_raw}")
  print(f"  Mean jump softened by: {abs(np.mean(decel_smooth)) - abs(np.mean(decel_raw)):+.3f} m/s^2")

  if len(raw_a_series) > 0:
    raw_arr = np.array(raw_a_series)
    smooth_arr = np.array(smoothed_a_series)
    raw_jerk = np.diff(raw_arr) / DT
    smooth_jerk = np.diff(smooth_arr) / DT

    decel_jerk_raw = raw_jerk[raw_jerk < -1.0]
    decel_jerk_smooth = smooth_jerk[smooth_jerk < -1.0]
    print(f"\n  Decel jerk events (< -1 m/s^3):")
    print(f"    Raw:      {len(decel_jerk_raw)}")
    print(f"    Smoothed: {len(decel_jerk_smooth)}")
    if len(decel_jerk_raw) > 0:
      print(f"    Raw P95 decel jerk:      {np.percentile(decel_jerk_raw, 5):.1f} m/s^3")
    if len(decel_jerk_smooth) > 0:
      print(f"    Smoothed P95 decel jerk: {np.percentile(decel_jerk_smooth, 5):.1f} m/s^3")

print()
