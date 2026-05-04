#!/usr/bin/env python3
"""
Deep analysis of lead detection during braking failures.
Traces radar lead status, model leads, and what the planner saw
during every driver brake intervention and critical stop.
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

# High-frequency timeline: capture EVERY radar + plan update
timeline = []
radar_gaps = []
prev_radar_status = None
prev_radar_t = None
gap_start = None

for seg in range(seg_start, seg_end):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  print(f"  reading seg {seg}...")

  cs_latest = {}
  sds_latest = {}
  radar_latest = {}
  model_lead_latest = {}

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9

    if typ == "carState":
      cs = msg.carState
      cs_latest = {
        "v_ego": f(cs, "vEgo"), "a_ego": f(cs, "aEgo"),
        "brake": bool(capget(cs, "brakePressed", False)),
        "standstill": bool(capget(cs, "standstill", False)),
      }

    elif typ == "selfdriveState":
      sds = msg.selfdriveState
      sds_latest = {"active": bool(capget(sds, "active", False))}

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      l1 = capget(rs, "leadTwo")
      has_l0 = bool(l0 and capget(l0, "status", False))
      has_l1 = bool(l1 and capget(l1, "status", False))

      new_radar = {
        "l0_status": has_l0,
        "l0_dRel": f(l0, "dRel") if has_l0 else float("nan"),
        "l0_vLead": f(l0, "vLead") if has_l0 else float("nan"),
        "l0_aLeadK": f(l0, "aLeadK") if has_l0 else float("nan"),
        "l0_prob": f(l0, "modelProb") if has_l0 else 0,
        "l1_status": has_l1,
        "l1_dRel": f(l1, "dRel") if has_l1 else float("nan"),
        "l1_vLead": f(l1, "vLead") if has_l1 else float("nan"),
        "t": t,
      }
      radar_latest = new_radar

      # Track radar gaps (lead disappears then reappears)
      if prev_radar_status is True and not has_l0:
        gap_start = t
      elif prev_radar_status is False and has_l0 and gap_start is not None:
        radar_gaps.append({
          "gap_start": gap_start, "gap_end": t,
          "duration": t - gap_start, "seg": seg,
          "d_rel_after": f(l0, "dRel"),
          "v_lead_after": f(l0, "vLead"),
        })
        gap_start = None

      prev_radar_status = has_l0
      prev_radar_t = t

    elif typ == "modelV2":
      mdl = msg.modelV2
      leads_v3 = capget(mdl, "leadsV3")
      if leads_v3 and len(leads_v3) > 0:
        ml0 = leads_v3[0]
        model_lead_latest = {
          "prob": f(ml0, "prob"),
          "x": f(capget(ml0, "xyva")[0]) if capget(ml0, "xyva") else float("nan"),
          "v": f(capget(ml0, "xyva")[1]) if capget(ml0, "xyva") else float("nan"),
          "t": t,
        }

    elif typ in ("longitudinalPlan",):
      lp = msg.longitudinalPlan
      a_target = f(lp, "aTarget")
      source = str(capget(lp, "longitudinalPlanSource", ""))
      should_stop = bool(capget(lp, "shouldStop", False))

      timeline.append({
        "t": t, "seg": seg,
        "v_ego": cs_latest.get("v_ego", 0),
        "a_ego": cs_latest.get("a_ego", 0),
        "a_target": a_target,
        "source": source,
        "should_stop": should_stop,
        "active": sds_latest.get("active", False),
        "brake": cs_latest.get("brake", False),
        "standstill": cs_latest.get("standstill", False),
        # Radar state
        "l0_status": radar_latest.get("l0_status", False),
        "l0_dRel": radar_latest.get("l0_dRel", float("nan")),
        "l0_vLead": radar_latest.get("l0_vLead", float("nan")),
        "l0_aLeadK": radar_latest.get("l0_aLeadK", float("nan")),
        "l0_prob": radar_latest.get("l0_prob", 0),
        "l1_status": radar_latest.get("l1_status", False),
        "l1_dRel": radar_latest.get("l1_dRel", float("nan")),
        # Model lead
        "model_prob": model_lead_latest.get("prob", 0),
        "model_x": model_lead_latest.get("x", float("nan")),
      })

print(f"\nTotal timeline points: {len(timeline)}")

# ─────────────────────────────────────────────────────────
# 1. Radar gap analysis
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("RADAR LEAD GAPS (lead disappeared then reappeared)")
print("=" * 90)

if radar_gaps:
  print(f"\nTotal gaps: {len(radar_gaps)}")
  long_gaps = [g for g in radar_gaps if g["duration"] > 0.5]
  print(f"Gaps > 0.5s: {len(long_gaps)}")
  print(f"Gaps > 1.0s: {len([g for g in radar_gaps if g['duration'] > 1.0])}")
  print(f"Gaps > 2.0s: {len([g for g in radar_gaps if g['duration'] > 2.0])}")
  print(f"Gaps > 5.0s: {len([g for g in radar_gaps if g['duration'] > 5.0])}")

  if long_gaps:
    print(f"\n  Gaps > 0.5s detail:")
    for g in sorted(long_gaps, key=lambda x: -x["duration"])[:20]:
      print(f"    seg={g['seg']} duration={g['duration']:.1f}s "
            f"d_rel_after={g['d_rel_after']:.0f}m v_lead_after={g['v_lead_after']*2.237:.0f}mph")
else:
  print("  No radar gaps found.")

# ─────────────────────────────────────────────────────────
# 2. Lead status transitions during approaches
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("LEAD STATUS TRANSITIONS (moment-to-moment)")
print("=" * 90)

transitions = []
for i in range(1, len(timeline)):
  prev = timeline[i-1]
  curr = timeline[i]
  if prev["l0_status"] != curr["l0_status"]:
    transitions.append({
      "t": curr["t"], "seg": curr["seg"],
      "from_status": prev["l0_status"], "to_status": curr["l0_status"],
      "v_ego": curr["v_ego"], "active": curr["active"],
      "d_rel_before": prev["l0_dRel"], "d_rel_after": curr["l0_dRel"],
      "v_lead_before": prev["l0_vLead"], "v_lead_after": curr["l0_vLead"],
      "a_target": curr["a_target"], "source": curr["source"],
    })

print(f"\nTotal lead status transitions: {len(transitions)}")
appear = [t for t in transitions if t["to_status"]]
disappear = [t for t in transitions if not t["to_status"]]
print(f"  Lead appeared: {len(appear)}")
print(f"  Lead disappeared: {len(disappear)}")

# Focus on disappearances while active and moving
active_disappear = [t for t in disappear if t["active"] and t["v_ego"] > 2.0]
print(f"\n  Lead disappeared while ACTIVE and MOVING (v>2): {len(active_disappear)}")
if active_disappear:
  for tr in active_disappear[:20]:
    d_str = f"{tr['d_rel_before']:.0f}m" if not math.isnan(tr['d_rel_before']) else "?"
    vl_str = f"{tr['v_lead_before']*2.237:.0f}mph" if not math.isnan(tr['v_lead_before']) else "?"
    print(f"    seg={tr['seg']} t={tr['t']:.1f} v={tr['v_ego']*2.237:.0f}mph "
          f"last_d={d_str} last_vLead={vl_str} "
          f"aTarget={tr['a_target']:+.2f} source={tr['source']}")

# ─────────────────────────────────────────────────────────
# 3. Brake interventions with full lead context
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("BRAKE INTERVENTIONS (full lead context)")
print("=" * 90)

brake_starts = []
prev_brake = False
for i, pt in enumerate(timeline):
  if pt["brake"] and not prev_brake and pt["active"] and pt["v_ego"] > 1.0:
    brake_starts.append(i)
  prev_brake = pt["brake"]

print(f"\nTotal brake interventions while active: {len(brake_starts)}")

for bi_num, bi_idx in enumerate(brake_starts):
  bi_t = timeline[bi_idx]["t"]
  bi_seg = timeline[bi_idx]["seg"]

  # Get 10-second window before brake press + 3 seconds after
  window = [pt for pt in timeline if bi_t - 10.0 <= pt["t"] <= bi_t + 3.0]
  if not window:
    continue

  print(f"\n--- Brake #{bi_num+1}: seg={bi_seg}, v={timeline[bi_idx]['v_ego']*2.237:.0f}mph ---")

  # Summary of lead status in window
  lead_on = sum(1 for pt in window if pt["l0_status"])
  lead_off = len(window) - lead_on
  print(f"  Lead visible in window: {lead_on}/{len(window)} frames "
        f"({lead_on/len(window)*100:.0f}%)")

  # Did lead disappear in the 5s before brake?
  pre_window = [pt for pt in timeline if bi_t - 5.0 <= pt["t"] < bi_t]
  lead_drops = []
  for i in range(1, len(pre_window)):
    if pre_window[i-1]["l0_status"] and not pre_window[i]["l0_status"]:
      lead_drops.append(pre_window[i-1])
  if lead_drops:
    print(f"  ** Lead DROPPED {len(lead_drops)}x in 5s before brake! **")
    for ld in lead_drops:
      print(f"     at d={ld['l0_dRel']:.0f}m, vLead={ld['l0_vLead']*2.237:.0f}mph")

  # Sample the timeline at 0.5s intervals
  print(f"  {'dt':>5} {'v_ego':>8} {'aTarget':>9} {'src':>8} {'lead':>5} {'d_rel':>6} "
        f"{'vLead':>7} {'aLeadK':>7} {'l0prob':>7} {'mdlProb':>8} {'mdlX':>6} {'stop':>5}")
  sampled = []
  last_t = -999
  for pt in window:
    if pt["t"] - last_t >= 0.5 or pt == window[-1] or pt["t"] == bi_t:
      sampled.append(pt)
      last_t = pt["t"]

  for pt in sampled:
    dt = pt["t"] - bi_t
    d_str = f"{pt['l0_dRel']:.0f}m" if pt["l0_status"] and not math.isnan(pt['l0_dRel']) else "--"
    vl_str = f"{pt['l0_vLead']*2.237:.0f}mph" if pt["l0_status"] and not math.isnan(pt['l0_vLead']) else "--"
    al_str = f"{pt['l0_aLeadK']:+.1f}" if pt["l0_status"] and not math.isnan(pt['l0_aLeadK']) else "--"
    lp_str = f"{pt['l0_prob']:.2f}" if pt["l0_status"] else "--"
    mp_str = f"{pt['model_prob']:.2f}" if not math.isnan(pt.get('model_prob', float('nan'))) else "--"
    mx_str = f"{pt['model_x']:.0f}m" if not math.isnan(pt.get('model_x', float('nan'))) else "--"
    marker = " <-- BRAKE" if abs(pt["t"] - bi_t) < 0.3 else ""
    print(f"  {dt:>+5.1f} {pt['v_ego']*2.237:>7.1f}mph {pt['a_target']:>+8.2f} {pt['source']:>8} "
          f"{'Y' if pt['l0_status'] else 'N':>5} {d_str:>6} {vl_str:>7} {al_str:>7} "
          f"{lp_str:>7} {mp_str:>8} {mx_str:>6} "
          f"{'STOP' if pt['should_stop'] else '':>5}{marker}")

# ─────────────────────────────────────────────────────────
# 4. Critical: active, approaching, no braking
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("CRITICAL: Active, lead visible d<50m vLead<3, but aTarget > -0.3")
print("=" * 90)

critical = []
for pt in timeline:
  if (pt["active"] and pt["l0_status"] and
      not math.isnan(pt["l0_dRel"]) and pt["l0_dRel"] < 50 and
      not math.isnan(pt["l0_vLead"]) and pt["l0_vLead"] < 3.0 and
      pt["v_ego"] > 3.0 and pt["a_target"] > -0.3):
    critical.append(pt)

if critical:
  print(f"\nTotal critical frames: {len(critical)}")
  for pt in critical[:30]:
    print(f"  seg={pt['seg']} t={pt['t']:.1f} v={pt['v_ego']*2.237:.0f}mph "
          f"aTarget={pt['a_target']:+.2f} source={pt['source']} "
          f"d={pt['l0_dRel']:.0f}m vLead={pt['l0_vLead']*2.237:.0f}mph "
          f"stop={pt['should_stop']}")
else:
  print("  None detected.")

# ─────────────────────────────────────────────────────────
# 5. Critical: active, moving, NO lead, model sees car
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("MODEL SEES LEAD BUT RADAR DOESN'T (active, v>5)")
print("=" * 90)

model_only = []
for pt in timeline:
  if (pt["active"] and pt["v_ego"] > 2.0 and
      not pt["l0_status"] and
      not math.isnan(pt.get("model_prob", float("nan"))) and
      pt["model_prob"] > 0.3 and
      not math.isnan(pt.get("model_x", float("nan"))) and
      pt["model_x"] < 80):
    model_only.append(pt)

if model_only:
  print(f"\nTotal frames: {len(model_only)}")
  last_t = -999
  for pt in model_only:
    if pt["t"] - last_t >= 1.0:
      print(f"  seg={pt['seg']} t={pt['t']:.1f} v={pt['v_ego']*2.237:.0f}mph "
            f"model_prob={pt['model_prob']:.2f} model_x={pt['model_x']:.0f}m "
            f"aTarget={pt['a_target']:+.2f} source={pt['source']}")
      last_t = pt["t"]
else:
  print("  None detected.")

print()
