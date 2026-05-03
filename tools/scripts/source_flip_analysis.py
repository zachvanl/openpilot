#!/usr/bin/env python3
"""Detailed source flipping analysis - what transitions happen and when."""
import sys
import math
import numpy as np
from collections import defaultdict, Counter
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

prev_source = None
transitions = Counter()
flip_events = []
cs_latest = {}
sds_latest = {}
radar_latest = {}
engaged_since = 0
prev_a_target = None

speed_at_flip = {"cruise->lead0": [], "lead0->cruise": [], "cruise->e2e": [],
                 "e2e->cruise": [], "lead0->e2e": [], "e2e->lead0": []}
accel_jump_at_flip = {"cruise->lead0": [], "lead0->cruise": [], "cruise->e2e": [],
                      "e2e->cruise": [], "lead0->e2e": [], "e2e->lead0": []}

rapid_flip_bursts = []
flip_window = []

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
      cs_latest = {"v_ego": f(cs, "vEgo"), "a_ego": f(cs, "aEgo"),
                   "standstill": bool(capget(cs, "standstill", False)), "t": t}

    elif typ == "selfdriveState":
      sds = msg.selfdriveState
      sds_latest = {"active": bool(capget(sds, "active", False)), "t": t}

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      if l0 and capget(l0, "status", False):
        radar_latest = {"dRel": f(l0, "dRel"), "vLead": f(l0, "vLead"),
                        "vRel": f(l0, "vRel", 0), "status": True}
      else:
        radar_latest = {"status": False}

    elif typ in ("longitudinalPlan", "longitudinalPlanSP"):
      if typ == "longitudinalPlanSP":
        lp = msg.longitudinalPlanSP
      else:
        lp = msg.longitudinalPlan
      a_target = f(lp, "aTarget")
      source = str(capget(lp, "longitudinalPlanSource", ""))

      active = sds_latest.get("active", False)
      if not active or cs_latest.get("standstill", True):
        engaged_since = 0
        prev_source = None
        prev_a_target = None
        continue

      engaged_since += 1
      if engaged_since < 50:
        prev_source = source
        prev_a_target = a_target
        continue

      v_ego = cs_latest.get("v_ego", 0)

      if prev_source and source != prev_source:
        trans_key = f"{prev_source}->{source}"
        transitions[trans_key] += 1

        a_jump = abs(a_target - prev_a_target) if prev_a_target is not None else 0

        if trans_key in speed_at_flip:
          speed_at_flip[trans_key].append(v_ego)
          accel_jump_at_flip[trans_key].append(a_jump)
        else:
          speed_at_flip[trans_key] = [v_ego]
          accel_jump_at_flip[trans_key] = [a_jump]

        d = radar_latest.get("dRel", float("nan")) if radar_latest.get("status") else float("nan")
        v_rel = radar_latest.get("vRel", float("nan")) if radar_latest.get("status") else float("nan")

        flip_events.append({
          "t": t, "from": prev_source, "to": source,
          "v_ego": v_ego, "a_target": a_target, "prev_a": prev_a_target,
          "a_jump": a_jump, "d": d, "v_rel": v_rel
        })

        flip_window.append(t)
        flip_window = [ft for ft in flip_window if t - ft < 10.0]
        if len(flip_window) >= 6:
          rapid_flip_bursts.append({"t": t, "count_in_10s": len(flip_window),
                                     "v_ego": v_ego, "d": d})

      prev_source = source
      prev_a_target = a_target

print("=" * 80)
print(f"SOURCE FLIP ANALYSIS: {route}")
print("=" * 80)

print("\n--- Transition Counts ---")
for trans, count in transitions.most_common():
  print(f"  {trans}: {count}")

print("\n--- Speed Distribution at Flips ---")
for trans_key in sorted(speed_at_flip.keys()):
  speeds = speed_at_flip[trans_key]
  if len(speeds) < 3:
    continue
  s = np.array(speeds) * 2.237
  print(f"\n  {trans_key} ({len(speeds)} events):")
  print(f"    Speed: mean={np.mean(s):.0f} mph, median={np.median(s):.0f} mph, range=[{np.min(s):.0f}, {np.max(s):.0f}] mph")
  city = np.sum(s < 45)
  hwy = np.sum(s >= 45)
  print(f"    City (<45mph): {city}, Highway (>=45mph): {hwy}")

print("\n--- Acceleration Jump at Flips ---")
for trans_key in sorted(accel_jump_at_flip.keys()):
  jumps = accel_jump_at_flip[trans_key]
  if len(jumps) < 3:
    continue
  j = np.array(jumps)
  print(f"\n  {trans_key} ({len(jumps)} events):")
  print(f"    |a_jump|: mean={np.mean(j):.3f}, median={np.median(j):.3f}, P95={np.percentile(j, 95):.3f}, max={np.max(j):.3f} m/s2")
  harsh = np.sum(j > 0.3)
  print(f"    Harsh jumps (>0.3 m/s2): {harsh} ({harsh/len(j)*100:.0f}%)")

print("\n--- Rapid Flip Bursts (>= 6 flips in 10s window) ---")
if rapid_flip_bursts:
  for burst in rapid_flip_bursts[:20]:
    d_str = f"{burst['d']:.0f}m" if not math.isnan(burst['d']) else "no lead"
    print(f"  t={burst['t']:.1f}s: {burst['count_in_10s']} flips, {burst['v_ego']*2.237:.0f} mph, lead={d_str}")
  print(f"  Total rapid-flip bursts: {len(rapid_flip_bursts)}")
else:
  print("  None detected.")

print("\n--- Worst Acceleration Jumps (top 15 by |a_jump|) ---")
flip_events.sort(key=lambda x: -x["a_jump"])
for ev in flip_events[:15]:
  d_str = f"{ev['d']:.0f}m" if not math.isnan(ev['d']) else "no lead"
  print(f"  {ev['from']}->{ev['to']}: a_jump={ev['a_jump']:.3f} m/s2, "
        f"a={ev['prev_a']:+.2f}->{ev['a_target']:+.2f}, "
        f"v={ev['v_ego']*2.237:.0f}mph, lead={d_str}")

print()
