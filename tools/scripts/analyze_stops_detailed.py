#!/usr/bin/env python3
"""
Deep analysis of stopping behavior.
Find ALL brake presses, disengagements, and rapid decel events.
Track the full timeline leading up to each stop/brake.
"""
import sys
import math
import numpy as np
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
seg_end = int(sys.argv[3]) if len(sys.argv) > 3 else 40

timeline = []

for seg in range(seg_start, seg_end):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  cs_latest = {}
  sds_latest = {}
  radar_latest = {}

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9

    if typ == "carState":
      cs = msg.carState
      cs_latest = {
        "v_ego": f(cs, "vEgo"), "a_ego": f(cs, "aEgo"),
        "gas": bool(capget(cs, "gasPressed", False)),
        "brake": bool(capget(cs, "brakePressed", False)),
        "standstill": bool(capget(cs, "standstill", False)),
        "t": t,
      }

    elif typ == "selfdriveState":
      sds = msg.selfdriveState
      sds_latest = {"active": bool(capget(sds, "active", False)),
                    "enabled": bool(capget(sds, "enabled", False)), "t": t}

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      if l0 and capget(l0, "status", False):
        radar_latest = {"dRel": f(l0, "dRel"), "vLead": f(l0, "vLead"),
                        "status": True}
      else:
        radar_latest = {"status": False}

    elif typ in ("longitudinalPlan", "longitudinalPlanSP"):
      if typ == "longitudinalPlanSP":
        continue
      lp = msg.longitudinalPlan
      a_target = f(lp, "aTarget")
      source = str(capget(lp, "longitudinalPlanSource", ""))
      should_stop = bool(capget(lp, "shouldStop", False))

      v_ego = cs_latest.get("v_ego", 0)
      active = sds_latest.get("active", False) or sds_latest.get("enabled", False)

      timeline.append({
        "t": t, "seg": seg,
        "v_ego": v_ego, "a_ego": cs_latest.get("a_ego", 0),
        "a_target": a_target, "source": source, "should_stop": should_stop,
        "active": active,
        "brake": cs_latest.get("brake", False),
        "gas": cs_latest.get("gas", False),
        "standstill": cs_latest.get("standstill", False),
        "lead_status": radar_latest.get("status", False),
        "d_rel": radar_latest.get("dRel", float("nan")),
        "v_lead": radar_latest.get("vLead", float("nan")),
      })

print(f"Total timeline points: {len(timeline)}")

# Find all stops: v_ego goes from > 3 m/s to < 0.5 m/s
stops = []
for i in range(len(timeline)):
  if (timeline[i]["v_ego"] < 0.5 and
      i > 10 and timeline[i-10]["v_ego"] > 3.0):
    stops.append(i)

# Deduplicate stops (within 5s)
deduped = []
for idx in stops:
  if not deduped or timeline[idx]["t"] - timeline[deduped[-1]]["t"] > 5.0:
    deduped.append(idx)
stops = deduped

print(f"\nTotal stops found: {len(stops)}")

# Find brake presses (any time)
brake_events = []
prev_brake = False
for i, pt in enumerate(timeline):
  if pt["brake"] and not prev_brake:
    brake_events.append(i)
  prev_brake = pt["brake"]

print(f"Total brake press events: {len(brake_events)}")

# Find disengagements
disengage_events = []
prev_active = False
for i, pt in enumerate(timeline):
  if prev_active and not pt["active"]:
    disengage_events.append(i)
  prev_active = pt["active"]

print(f"Total disengagements: {len(disengage_events)}")

# For each stop, show the 15-second lead-up
print("\n" + "=" * 100)
print("STOP EVENT ANALYSIS (15s lead-up to each stop)")
print("=" * 100)

for stop_num, stop_idx in enumerate(stops):
  stop_t = timeline[stop_idx]["t"]

  # Find leadup (15s before stop)
  leadup = [pt for pt in timeline if stop_t - 15.0 <= pt["t"] <= stop_t + 2.0]

  if not leadup:
    continue

  print(f"\n--- Stop #{stop_num+1}: seg={timeline[stop_idx]['seg']}, t={stop_t:.1f} ---")

  # Check if there was a brake press or disengagement near this stop
  nearby_brake = [i for i in brake_events if abs(timeline[i]["t"] - stop_t) < 10.0]
  nearby_disengage = [i for i in disengage_events if abs(timeline[i]["t"] - stop_t) < 10.0]

  if nearby_brake:
    bi = nearby_brake[0]
    print(f"  BRAKE PRESS at t={timeline[bi]['t']:.1f} "
          f"(v={timeline[bi]['v_ego']*2.237:.0f}mph, active={timeline[bi]['active']})")
  if nearby_disengage:
    di = nearby_disengage[0]
    print(f"  DISENGAGEMENT at t={timeline[di]['t']:.1f} "
          f"(v={timeline[di]['v_ego']*2.237:.0f}mph)")

  # Sample every ~1s from leadup
  sampled = []
  last_t = -999
  for pt in leadup:
    if pt["t"] - last_t >= 1.0 or pt == leadup[-1]:
      sampled.append(pt)
      last_t = pt["t"]

  print(f"  {'t':>8} {'v_ego':>8} {'a_target':>9} {'source':>8} {'active':>7} {'brake':>6} {'lead':>6} {'d_rel':>6} {'v_lead':>7} {'stop':>5}")
  for pt in sampled:
    d_str = f"{pt['d_rel']:.0f}m" if not math.isnan(pt["d_rel"]) and pt["lead_status"] else "--"
    vl_str = f"{pt['v_lead']*2.237:.0f}mph" if not math.isnan(pt["v_lead"]) and pt["lead_status"] else "--"
    print(f"  {pt['t']:.1f} {pt['v_ego']*2.237:>7.1f}mph {pt['a_target']:>+8.2f} {pt['source']:>8} "
          f"{'Y' if pt['active'] else 'N':>7} {'BRAKE' if pt['brake'] else '':>6} "
          f"{d_str:>6} {vl_str:>7} {'STOP' if pt['should_stop'] else '':>5}")

# Also find segments where active but aTarget > 0 and lead is close/slow
print("\n" + "=" * 100)
print("CRITICAL: Active, approaching slow lead (d<50, vLead<3), but aTarget > -0.3")
print("=" * 100)

critical = []
for pt in timeline:
  if (pt["active"] and pt["lead_status"] and
      pt["d_rel"] < 50 and not math.isnan(pt["v_lead"]) and
      pt["v_lead"] < 3.0 and pt["v_ego"] > 3.0 and
      pt["a_target"] > -0.3):
    critical.append(pt)

if critical:
  for pt in critical:
    print(f"  seg={pt['seg']} t={pt['t']:.1f} v={pt['v_ego']*2.237:.0f}mph "
          f"aTarget={pt['a_target']:+.2f} source={pt['source']} "
          f"lead={pt['d_rel']:.0f}m vLead={pt['v_lead']*2.237:.0f}mph "
          f"shouldStop={pt['should_stop']}")
else:
  print("  None detected.")

print()
