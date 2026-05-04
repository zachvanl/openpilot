#!/usr/bin/env python3
"""
Analyze model lead probability around radar gaps.
Since the CRV 6G has NO physical radar (radarUnavailable=True),
all lead detection depends on modelV2.leadsV3[0].prob > 0.5.
This script tracks model lead probability to understand why leads drop.
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

model_probs = []
radar_lead_timeline = []
v_ego_latest = 0
sds_active = False

for seg in range(seg_start, seg_end):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  print(f"  reading seg {seg}...")

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9

    if typ == "carState":
      v_ego_latest = f(msg.carState, "vEgo")

    elif typ == "selfdriveState":
      sds_active = bool(capget(msg.selfdriveState, "active", False))

    elif typ == "modelV2":
      mdl = msg.modelV2
      leads_v3 = capget(mdl, "leadsV3")
      if leads_v3 and len(leads_v3) > 0:
        l0 = leads_v3[0]
        prob = f(l0, "prob")
        xyva = capget(l0, "xyva")
        x_pos = float("nan")
        v_pos = float("nan")
        if xyva and len(xyva) >= 2:
          x_pos = f(xyva[0])
          v_pos = f(xyva[1])

        model_probs.append({
          "t": t, "seg": seg,
          "prob": prob, "x": x_pos, "v": v_pos,
          "v_ego": v_ego_latest, "active": sds_active,
        })

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      has_l0 = bool(l0 and capget(l0, "status", False))
      radar_lead_timeline.append({
        "t": t, "seg": seg,
        "has_lead": has_l0,
        "d_rel": f(l0, "dRel") if has_l0 else float("nan"),
        "v_lead": f(l0, "vLead") if has_l0 else float("nan"),
        "model_prob": f(l0, "modelProb") if has_l0 else 0,
        "radar": bool(capget(l0, "radar", False)) if has_l0 else False,
        "v_ego": v_ego_latest,
      })

print(f"\nTotal model frames: {len(model_probs)}")
print(f"Total radar frames: {len(radar_lead_timeline)}")

# Model probability distribution
probs = np.array([p["prob"] for p in model_probs if not math.isnan(p["prob"])])
if len(probs) > 0:
  print("\n" + "=" * 80)
  print("MODEL LEAD PROBABILITY DISTRIBUTION")
  print("=" * 80)
  bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
  hist, _ = np.histogram(probs, bins=bins)
  total = len(probs)
  for i in range(len(bins)-1):
    bar = "#" * int(hist[i] / total * 100)
    marker = " <-- THRESHOLD" if bins[i] == 0.5 else ""
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:>5} ({hist[i]/total*100:5.1f}%) {bar}{marker}")

  # How many frames fall in the "danger zone" (0.3-0.5)?
  danger = np.sum((probs >= 0.3) & (probs < 0.5))
  print(f"\n  Frames in danger zone (0.3-0.5): {danger} ({danger/total*100:.1f}%)")
  print(f"  Frames below threshold (<0.5): {np.sum(probs < 0.5)} ({np.sum(probs < 0.5)/total*100:.1f}%)")

# Find gaps in radar lead detection and correlate with model prob
print("\n" + "=" * 80)
print("RADAR LEAD GAPS + MODEL PROB CONTEXT")
print("=" * 80)

gaps = []
gap_start = None
prev_has_lead = None
prev_seg = None

for rt in radar_lead_timeline:
  if prev_has_lead is True and not rt["has_lead"]:
    gap_start = rt["t"]
    gap_seg = rt["seg"]
    gap_v_ego = rt["v_ego"]
  elif prev_has_lead is False and rt["has_lead"] and gap_start is not None:
    gap_end = rt["t"]
    gaps.append({
      "start": gap_start, "end": gap_end,
      "duration": gap_end - gap_start, "seg": gap_seg,
      "v_ego_start": gap_v_ego,
      "d_rel_after": rt["d_rel"],
      "v_lead_after": rt["v_lead"],
    })
    gap_start = None
  prev_has_lead = rt["has_lead"]

# For each significant gap, show the model probability during the gap
significant_gaps = [g for g in gaps if g["duration"] > 2.0 and g["v_ego_start"] > 3.0]
print(f"\nSignificant gaps (>2s, v_ego>3): {len(significant_gaps)}")

for gi, gap in enumerate(sorted(significant_gaps, key=lambda x: -x["duration"])[:10]):
  # Find model probs during and around gap
  probs_during = [p for p in model_probs
                  if gap["start"] - 2 <= p["t"] <= gap["end"] + 2]

  print(f"\n  Gap #{gi+1}: seg={gap['seg']} duration={gap['duration']:.1f}s "
        f"v_ego={gap['v_ego_start']*2.237:.0f}mph")
  d_str = f"{gap['d_rel_after']:.0f}m" if not math.isnan(gap['d_rel_after']) else "?"
  vl_str = f"{gap['v_lead_after']*2.237:.0f}mph" if not math.isnan(gap['v_lead_after']) else "?"
  print(f"       lead reappears at d={d_str}, vLead={vl_str}")

  if probs_during:
    print(f"       Model prob during gap ({len(probs_during)} frames):")
    print(f"         {'t':>8} {'prob':>6} {'model_x':>8} {'model_v':>10} {'active':>7}")
    last_t = -999
    for p in probs_during:
      if p["t"] - last_t >= 1.0 or p == probs_during[-1] or p == probs_during[0]:
        x_str = f"{p['x']:.0f}m" if not math.isnan(p['x']) else "?"
        v_str = f"{p['v']*2.237:.0f}mph" if not math.isnan(p['v']) else "?"
        marker = ""
        if p["prob"] >= 0.3 and p["prob"] < 0.5:
          marker = " <-- NEAR THRESHOLD"
        elif p["prob"] >= 0.5:
          marker = " <-- ABOVE THRESHOLD"
        print(f"         {p['t']:.1f} {p['prob']:>5.2f} {x_str:>8} {v_str:>10} "
              f"{'Y' if p['active'] else 'N':>7}{marker}")
        last_t = p["t"]

# Check if radarState ever reports radar=True (confirming no physical radar)
radar_flags = [rt for rt in radar_lead_timeline if rt["has_lead"] and rt["radar"]]
print(f"\n" + "=" * 80)
print(f"PHYSICAL RADAR CONFIRMATION")
print(f"=" * 80)
print(f"  Frames where lead came from physical radar: {len(radar_flags)}")
print(f"  Frames where lead came from vision only:    {len([r for r in radar_lead_timeline if r['has_lead'] and not r['radar']])}")
print(f"  Frames with no lead at all:                 {len([r for r in radar_lead_timeline if not r['has_lead']])}")

if len(radar_flags) == 0:
  print(f"\n  ** CONFIRMED: This car has NO physical radar. **")
  print(f"  ** All lead detection is 100% vision-model based. **")
  print(f"  ** Lead appears/disappears based on modelV2.leadsV3[0].prob > 0.5 **")

print()
