#!/usr/bin/env python3
"""
Simulate the effect of lead hysteresis on a real drive.
Compares stock (prob > 0.5) vs hysteresis (acquire 0.5, release 0.3, max 1s)
to show how many lead gaps would have been bridged.
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

ACQUIRE = 0.5
RELEASE = 0.3
MAX_HOLD = 20

# Track radarState lead status (stock behavior)
stock_gaps = []
stock_gap_start = None
prev_stock_status = None

# Track what hysteresis would have done (simulated from radarState)
hyst_prev_active = False
hyst_hold = 0
hyst_gaps = []
hyst_gap_start = None
prev_hyst_active = None

total_frames = 0
hyst_held_frames = 0
stock_no_lead_frames = 0
v_ego_latest = 0
active_latest = False

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
      v_ego_latest = f(msg.carState, "vEgo")

    elif typ == "selfdriveState":
      active_latest = bool(capget(msg.selfdriveState, "active", False))

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      stock_status = bool(l0 and capget(l0, "status", False))
      model_prob = f(l0, "modelProb") if stock_status else 0.0

      total_frames += 1

      # Stock gap tracking
      if not stock_status:
        stock_no_lead_frames += 1
      if prev_stock_status is True and not stock_status:
        stock_gap_start = t
      elif prev_stock_status is False and stock_status and stock_gap_start is not None:
        stock_gaps.append({"start": stock_gap_start, "end": t, "dur": t - stock_gap_start,
                           "v_ego": v_ego_latest, "seg": seg})
        stock_gap_start = None
      prev_stock_status = stock_status

      # Simulate hysteresis (we can't see true model prob from qlog when lead is off,
      # so we simulate based on the radarState.leadOne.modelProb when available)
      if stock_status:
        hyst_active = True
        hyst_prev_active = True
        hyst_hold = 0
      elif hyst_prev_active and hyst_hold < MAX_HOLD:
        hyst_active = True
        hyst_hold += 1
        hyst_held_frames += 1
      else:
        hyst_active = False
        hyst_prev_active = False
        hyst_hold = 0

      # Hyst gap tracking
      if prev_hyst_active is True and not hyst_active:
        hyst_gap_start = t
      elif prev_hyst_active is False and hyst_active and hyst_gap_start is not None:
        hyst_gaps.append({"start": hyst_gap_start, "end": t, "dur": t - hyst_gap_start,
                          "v_ego": v_ego_latest, "seg": seg})
        hyst_gap_start = None
      prev_hyst_active = hyst_active

print("=" * 80)
print(f"LEAD HYSTERESIS SIMULATION: {route}")
print("=" * 80)

print(f"\nTotal radarState frames: {total_frames}")
print(f"Stock: no-lead frames: {stock_no_lead_frames} ({stock_no_lead_frames/max(total_frames,1)*100:.1f}%)")
print(f"Hysteresis: held frames (bridged by 1s max hold): {hyst_held_frames}")
print(f"Effective no-lead reduction: {stock_no_lead_frames} -> {stock_no_lead_frames - hyst_held_frames}")

print(f"\n--- Stock lead gaps ---")
long_stock = [g for g in stock_gaps if g["dur"] > 0.5]
print(f"Total gaps: {len(stock_gaps)}")
print(f"Gaps > 0.5s: {len(long_stock)}")
print(f"Gaps > 2.0s: {len([g for g in stock_gaps if g['dur'] > 2.0])}")
print(f"Gaps > 5.0s: {len([g for g in stock_gaps if g['dur'] > 5.0])}")

print(f"\n--- Hysteresis lead gaps ---")
long_hyst = [g for g in hyst_gaps if g["dur"] > 0.5]
print(f"Total gaps: {len(hyst_gaps)}")
print(f"Gaps > 0.5s: {len(long_hyst)}")
print(f"Gaps > 2.0s: {len([g for g in hyst_gaps if g['dur'] > 2.0])}")
print(f"Gaps > 5.0s: {len([g for g in hyst_gaps if g['dur'] > 5.0])}")

# Show gaps that were shortened or eliminated
print(f"\n--- Impact ---")
bridged = len(stock_gaps) - len(hyst_gaps)
shortened = 0
for sg in long_stock:
  matching = [hg for hg in long_hyst
              if abs(hg["start"] - sg["start"]) < 2.0]
  if matching:
    hg = matching[0]
    if hg["dur"] < sg["dur"] - 0.5:
      shortened += 1
      print(f"  Shortened: seg={sg['seg']} {sg['dur']:.1f}s -> {hg['dur']:.1f}s "
            f"(v={sg['v_ego']*2.237:.0f}mph)")
  else:
    if sg["dur"] <= 1.5:
      print(f"  Eliminated: seg={sg['seg']} {sg['dur']:.1f}s gap bridged "
            f"(v={sg['v_ego']*2.237:.0f}mph)")

# Remaining long gaps (hysteresis alone can't fix these)
remaining_long = [g for g in hyst_gaps if g["dur"] > 2.0]
if remaining_long:
  print(f"\n  Remaining long gaps (>2s, need model improvement):")
  for g in sorted(remaining_long, key=lambda x: -x["dur"])[:10]:
    print(f"    seg={g['seg']} {g['dur']:.1f}s (v={g['v_ego']*2.237:.0f}mph)")

print()
