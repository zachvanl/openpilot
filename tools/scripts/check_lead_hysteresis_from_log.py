#!/usr/bin/env python3
"""
Infer whether LeadHysteresis was active during a drive.

Stock radard (vision-only): leadOne.status True only when model prob > 0.5,
so modelProb should always be > 0.5 when status is True.

With hysteresis hold: we republish from get_RadarState_from_vision while
0.3 <= prob < 0.5, so status True with modelProb < 0.5 (and radar False for
radarless) is a strong fingerprint.

Also report lead gap counts vs typical pre-hysteresis drives.
"""
import sys
import math
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
    return float("nan")

route = sys.argv[1]
seg = int(sys.argv[2]) if len(sys.argv) > 2 else 0

seg_id = f"{route}/{seg}"
lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)

latest_prob = float("nan")
held_fingerprint = 0  # status True, prob < 0.5, radar False
status_true_high_prob = 0
status_true_radar = 0
total_radar = 0

prev_status = None
gaps = []
gap_start = None

for msg in lr:
  typ = msg.which()
  if typ == "modelV2":
    lv = capget(msg.modelV2, "leadsV3")
    if lv and len(lv) > 0:
      latest_prob = f(lv[0], "prob")

  elif typ == "radarState":
    total_radar += 1
    rs = msg.radarState
    l0 = capget(rs, "leadOne")
    status = bool(l0 and capget(l0, "status", False))
    mp = f(l0, "modelProb") if l0 else float("nan")
    radar = bool(capget(l0, "radar", False)) if l0 else False

    if status and not math.isnan(mp):
      if mp < 0.5 and not radar:
        held_fingerprint += 1
      if mp >= 0.5:
        status_true_high_prob += 1
      if radar:
        status_true_radar += 1

    if prev_status is True and not status:
      gap_start = msg.logMonoTime / 1e9
    elif prev_status is False and status and gap_start is not None:
      gaps.append(msg.logMonoTime / 1e9 - gap_start)
      gap_start = None
    prev_status = status

print("=" * 70)
print(f"LEAD HYSTERESIS INFERENCE: {seg_id}")
print("=" * 70)
print(f"\nradarState frames: {total_radar}")
print(f"lead status=True with modelProb >= 0.5: {status_true_high_prob}")
print(f"lead status=True with modelProb < 0.5 and radar=False: {held_fingerprint}")
print(f"lead status=True with radar=True (fusion car): {status_true_radar}")

if held_fingerprint > 10:
  print("\n** Verdict: LeadHysteresis was almost certainly ON **")
  print("   (Stock vision-only cannot show status=True with modelProb<0.5)")
elif held_fingerprint > 0:
  print("\n** Verdict: LeadHysteresis likely ON (few hold frames) **")
else:
  print("\n** Verdict: LeadHysteresis was likely OFF (no hold fingerprint) **")
  print("   If the car is radarless and you never see prob<0.5 with a lead,")
  print("   hysteresis was not holding through low-confidence dips.")

long_gaps = [g for g in gaps if g > 0.5]
print(f"\nLead gaps (status True->False->True): {len(gaps)} total, {len(long_gaps)} > 0.5s")
if gaps:
  print(f"  Max gap: {max(gaps):.1f}s")

print()
