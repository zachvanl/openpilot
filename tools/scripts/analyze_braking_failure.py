#!/usr/bin/env python3
"""
Analyze braking failures - find moments where:
1. Car approaching stopped/slow lead but aTarget stays near 0 or positive
2. Driver brake interventions (gasPressed or brakePressed)
3. Red light / stop scenarios where e2e should command stop but doesn't
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
seg_end = int(sys.argv[3]) if len(sys.argv) > 3 else 40

cs_latest = {}
sds_latest = {}
radar_latest = {}
plan_latest = {}
engaged_since = 0
prev_output_a = None

brake_interventions = []
near_miss_events = []
no_brake_events = []
all_plan_samples = []

for seg in range(seg_start, seg_end):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  print(f"  reading segment {seg}...")

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9

    if typ == "carState":
      cs = msg.carState
      v_ego = f(cs, "vEgo")
      a_ego = f(cs, "aEgo")
      gas = bool(capget(cs, "gasPressed", False))
      brake = bool(capget(cs, "brakePressed", False))
      standstill = bool(capget(cs, "standstill", False))
      cs_latest = {"v_ego": v_ego, "a_ego": a_ego, "gas": gas, "brake": brake,
                   "standstill": standstill, "t": t}

    elif typ == "selfdriveState":
      sds = msg.selfdriveState
      active = bool(capget(sds, "active", False))
      enabled = bool(capget(sds, "enabled", False))
      sds_latest = {"active": active or enabled, "t": t}

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      if l0 and capget(l0, "status", False):
        radar_latest = {"dRel": f(l0, "dRel"), "vLead": f(l0, "vLead"),
                        "vRel": f(l0, "vRel", 0), "aLeadK": f(l0, "aLeadK", 0),
                        "status": True, "t": t}
      else:
        radar_latest = {"status": False, "t": t}

    elif typ in ("longitudinalPlan", "longitudinalPlanSP"):
      if typ == "longitudinalPlanSP":
        lp = msg.longitudinalPlanSP
      else:
        lp = msg.longitudinalPlan
      a_target = f(lp, "aTarget")
      source = str(capget(lp, "longitudinalPlanSource", ""))
      should_stop = bool(capget(lp, "shouldStop", False))

      active = sds_latest.get("active", False)
      if not active:
        engaged_since = 0
        prev_output_a = None
        continue

      engaged_since += 1
      if engaged_since < 20:
        prev_output_a = a_target
        continue

      v_ego = cs_latest.get("v_ego", 0)
      brake_pressed = cs_latest.get("brake", False)

      plan_latest = {"a_target": a_target, "source": source,
                     "should_stop": should_stop, "t": t}

      sample = {
        "t": t, "seg": seg, "v_ego": v_ego, "a_target": a_target,
        "source": source, "should_stop": should_stop,
        "brake_pressed": brake_pressed,
        "lead_status": radar_latest.get("status", False),
        "d_rel": radar_latest.get("dRel", float("nan")),
        "v_lead": radar_latest.get("vLead", float("nan")),
        "v_rel": radar_latest.get("vRel", float("nan")),
      }
      all_plan_samples.append(sample)

      # Detect brake interventions while engaged
      if brake_pressed and v_ego > 2.0:
        brake_interventions.append(sample)

      # Detect: approaching stopped/slow lead, but aTarget is not braking
      if radar_latest.get("status"):
        d = radar_latest.get("dRel", 999)
        v_lead = radar_latest.get("vLead", 99)
        v_rel = radar_latest.get("vRel", 0)
        closing = v_ego - v_lead

        # Stopped or very slow lead, car still moving, not braking hard enough
        if v_lead < 2.0 and v_ego > 5.0 and d < 80 and a_target > -0.5:
          no_brake_events.append(sample)

        # Near miss: closing fast on slow lead, aTarget still mild
        if closing > 5.0 and d < 60 and v_lead < 5.0 and a_target > -1.0:
          near_miss_events.append(sample)

      prev_output_a = a_target

# ═══════════════════════════════════════
# REPORT
# ═══════════════════════════════════════

print("\n" + "=" * 80)
print(f"BRAKING FAILURE ANALYSIS: {route}")
print(f"Segments: {seg_start}-{seg_end}, Total plan samples: {len(all_plan_samples)}")
print("=" * 80)

print(f"\n--- Brake Interventions (driver pressed brake while engaged, v>2 m/s) ---")
if brake_interventions:
  print(f"Total: {len(brake_interventions)}")
  for ev in brake_interventions[:30]:
    d_str = f"{ev['d_rel']:.0f}m" if not math.isnan(ev['d_rel']) else "no lead"
    vl_str = f"{ev['v_lead']*2.237:.0f}mph" if not math.isnan(ev['v_lead']) else "?"
    print(f"  seg={ev['seg']} t={ev['t']:.1f} v={ev['v_ego']*2.237:.0f}mph "
          f"aTarget={ev['a_target']:+.2f} source={ev['source']} "
          f"lead={d_str} vLead={vl_str} shouldStop={ev['should_stop']}")
else:
  print("  None detected.")

print(f"\n--- Stopped/Slow Lead but No Braking (v_lead<2, v_ego>5, d<80, aTarget>-0.5) ---")
if no_brake_events:
  print(f"Total: {len(no_brake_events)}")
  for ev in no_brake_events[:30]:
    d_str = f"{ev['d_rel']:.0f}m" if not math.isnan(ev['d_rel']) else "no lead"
    vl_str = f"{ev['v_lead']*2.237:.0f}mph" if not math.isnan(ev['v_lead']) else "?"
    print(f"  seg={ev['seg']} t={ev['t']:.1f} v={ev['v_ego']*2.237:.0f}mph "
          f"aTarget={ev['a_target']:+.2f} source={ev['source']} "
          f"lead={d_str} vLead={vl_str} shouldStop={ev['should_stop']}")
else:
  print("  None detected.")

print(f"\n--- Near-Miss Events (closing>5 on slow lead, d<60, aTarget>-1.0) ---")
if near_miss_events:
  print(f"Total: {len(near_miss_events)}")
  for ev in near_miss_events[:30]:
    d_str = f"{ev['d_rel']:.0f}m" if not math.isnan(ev['d_rel']) else "no lead"
    vl_str = f"{ev['v_lead']*2.237:.0f}mph" if not math.isnan(ev['v_lead']) else "?"
    closing = ev['v_ego'] - ev['v_lead'] if not math.isnan(ev['v_lead']) else 0
    print(f"  seg={ev['seg']} t={ev['t']:.1f} v={ev['v_ego']*2.237:.0f}mph "
          f"aTarget={ev['a_target']:+.2f} source={ev['source']} "
          f"lead={d_str} vLead={vl_str} closing={closing*2.237:.0f}mph shouldStop={ev['should_stop']}")
else:
  print("  None detected.")

# Check if the jerk limiter could be responsible
print(f"\n--- Jerk Limiter Impact Check ---")
OUTPUT_DECEL_JERK_LIMIT = -3.0
DT = 0.05
suppressed_events = 0
for i in range(1, len(all_plan_samples)):
  prev = all_plan_samples[i-1]
  curr = all_plan_samples[i]
  if prev["seg"] != curr["seg"]:
    continue
  raw_drop = curr["a_target"] - prev["a_target"]
  # Check if jerk limiter would have restricted this
  max_drop_per_cycle = OUTPUT_DECEL_JERK_LIMIT * DT  # -0.15
  if raw_drop < max_drop_per_cycle and curr["a_target"] < -0.3:
    suppressed_events += 1

print(f"  Decel cycles where jerk limiter would constrain: {suppressed_events}")
print(f"  (This counts cycles where aTarget drops faster than {OUTPUT_DECEL_JERK_LIMIT*DT:.2f}/cycle)")

# Look for shouldStop behavior
stop_count = sum(1 for s in all_plan_samples if s["should_stop"])
print(f"\n--- shouldStop Analysis ---")
print(f"  shouldStop=True count: {stop_count} / {len(all_plan_samples)}")

# Source distribution
from collections import Counter
sources = Counter(s["source"] for s in all_plan_samples)
print(f"\n--- Source Distribution ---")
for src, cnt in sources.most_common():
  print(f"  {src}: {cnt} ({cnt/len(all_plan_samples)*100:.1f}%)")

print()
