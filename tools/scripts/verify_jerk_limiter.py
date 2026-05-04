#!/usr/bin/env python3
"""
Verify the jerk limiter + TTC bypass simulation against real drive data.
Runs two parallel simulations:
  1. Limiter-only (no TTC bypass) — shows max smoothing
  2. Limiter + TTC bypass — matches actual deployed code
Compares both to raw (no limiter) baseline.
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

OUTPUT_DECEL_JERK_LIMIT = -3.0
OUTPUT_DECEL_EMERGENCY_DIST = 4.0
OUTPUT_DECEL_TTC_BYPASS = 4.0
GENTLE_DECEL_NO_LEAD_DIST = 40.0
DT = 0.05

prev_a_target = None
sim_no_ttc = None     # limiter only (no TTC bypass)
sim_with_ttc = None   # limiter + TTC bypass (deployed code)
sds_latest = {}
cs_latest = {}
radar_latest = {}
engaged_since = 0

raw_jumps = []
smooth_no_ttc_jumps = []
smooth_ttc_jumps = []
raw_a_series = []
smooth_no_ttc_series = []
smooth_ttc_series = []
times = []

ttc_bypass_events = []

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
        radar_latest = {"dRel": f(l0, "dRel"), "vLead": f(l0, "vLead"), "status": True}
      else:
        radar_latest = {"status": False}

    elif typ in ("longitudinalPlan", "longitudinalPlanSP"):
      if typ == "longitudinalPlanSP":
        continue
      lp = msg.longitudinalPlan
      a_target = f(lp, "aTarget")
      should_stop = bool(capget(lp, "shouldStop", False))

      active = sds_latest.get("active", False)
      if not active or cs_latest.get("standstill", True):
        engaged_since = 0
        prev_a_target = None
        sim_no_ttc = None
        sim_with_ttc = None
        continue

      engaged_since += 1
      if engaged_since < 50:
        prev_a_target = a_target
        sim_no_ttc = a_target
        sim_with_ttc = a_target
        continue

      v_ego = cs_latest.get("v_ego", 0)

      if prev_a_target is not None and sim_no_ttc is not None:
        raw_jump = a_target - prev_a_target
        raw_jumps.append(raw_jump)

        d_rel = radar_latest.get("dRel", GENTLE_DECEL_NO_LEAD_DIST) if radar_latest.get("status") else GENTLE_DECEL_NO_LEAD_DIST
        lead_status = radar_latest.get("status", False)

        # --- Simulation 1: Limiter only (no TTC bypass) ---
        emergency_dist = lead_status and d_rel < OUTPUT_DECEL_EMERGENCY_DIST
        if a_target < sim_no_ttc and not should_stop and not emergency_dist:
          min_a = sim_no_ttc + OUTPUT_DECEL_JERK_LIMIT * DT
          out_no_ttc = max(a_target, min_a)
        else:
          out_no_ttc = a_target

        # --- Simulation 2: Limiter + TTC bypass (deployed code) ---
        emergency_full = lead_status and d_rel < OUTPUT_DECEL_EMERGENCY_DIST
        if not emergency_full and lead_status:
          v_lead = max(radar_latest.get("vLead", 0), 0.0)
          closing_speed = max(v_ego - v_lead, 0.1)
          ttc = d_rel / closing_speed
          if ttc < OUTPUT_DECEL_TTC_BYPASS:
            emergency_full = True

        if a_target < sim_with_ttc and not should_stop and not emergency_full:
          min_a = sim_with_ttc + OUTPUT_DECEL_JERK_LIMIT * DT
          out_ttc = max(a_target, min_a)
        else:
          out_ttc = a_target

        # Track TTC bypass activations
        if emergency_full and not emergency_dist and a_target < sim_with_ttc:
          v_lead_ev = max(radar_latest.get("vLead", 0), 0.0)
          closing_ev = max(v_ego - v_lead_ev, 0.1)
          ttc_ev = d_rel / closing_ev
          ttc_bypass_events.append({
            "t": t, "seg": seg, "v_ego": v_ego * 2.237, "d_rel": d_rel,
            "v_lead": v_lead_ev * 2.237, "ttc": ttc_ev,
            "a_target": a_target, "sim_prev": sim_with_ttc,
            "delta_held_back": sim_no_ttc + OUTPUT_DECEL_JERK_LIMIT * DT - a_target,
          })

        smooth_no_ttc_jumps.append(out_no_ttc - sim_no_ttc)
        smooth_ttc_jumps.append(out_ttc - sim_with_ttc)

        raw_a_series.append(a_target)
        smooth_no_ttc_series.append(out_no_ttc)
        smooth_ttc_series.append(out_ttc)
        times.append(t)

        sim_no_ttc = out_no_ttc
        sim_with_ttc = out_ttc
      else:
        sim_no_ttc = a_target
        sim_with_ttc = a_target

      prev_a_target = a_target

raw_jumps = np.array(raw_jumps)
smooth_no_ttc_jumps = np.array(smooth_no_ttc_jumps)
smooth_ttc_jumps = np.array(smooth_ttc_jumps)

print("=" * 80)
print(f"JERK LIMITER + TTC BYPASS SIMULATION: {route}")
print("=" * 80)

print(f"\nTotal engaged plan cycles analyzed: {len(raw_jumps)}")

decel_mask = raw_jumps < -0.1
decel_raw = raw_jumps[decel_mask]
decel_no_ttc = smooth_no_ttc_jumps[decel_mask]
decel_ttc = smooth_ttc_jumps[decel_mask]

print(f"Deceleration events (raw jump < -0.1 m/s2): {len(decel_raw)}")

if len(decel_raw) > 0:
  harsh_raw = int(np.sum(decel_raw < -0.3))
  harsh_no_ttc = int(np.sum(decel_no_ttc < -0.3))
  harsh_ttc = int(np.sum(decel_ttc < -0.3))

  print(f"\n{'':>30} {'Raw':>12} {'Limiter Only':>14} {'Limiter+TTC':>14}")
  print(f"  {'-'*72}")
  print(f"  {'Mean decel jump':>28} {np.mean(decel_raw):>+11.3f} {np.mean(decel_no_ttc):>+13.3f} {np.mean(decel_ttc):>+13.3f}")
  print(f"  {'P95 decel jump':>28} {np.percentile(decel_raw, 5):>+11.3f} {np.percentile(decel_no_ttc, 5):>+13.3f} {np.percentile(decel_ttc, 5):>+13.3f}")
  print(f"  {'Worst decel jump':>28} {np.min(decel_raw):>+11.3f} {np.min(decel_no_ttc):>+13.3f} {np.min(decel_ttc):>+13.3f}")
  print(f"  {'Harsh jumps (>0.3)':>28} {harsh_raw:>11} {harsh_no_ttc:>14} {harsh_ttc:>14}")

  if len(raw_a_series) > 0:
    raw_arr = np.array(raw_a_series)
    no_ttc_arr = np.array(smooth_no_ttc_series)
    ttc_arr = np.array(smooth_ttc_series)
    raw_jerk = np.diff(raw_arr) / DT
    no_ttc_jerk = np.diff(no_ttc_arr) / DT
    ttc_jerk = np.diff(ttc_arr) / DT

    decel_jerk_raw = raw_jerk[raw_jerk < -1.0]
    decel_jerk_no_ttc = no_ttc_jerk[no_ttc_jerk < -1.0]
    decel_jerk_ttc = ttc_jerk[ttc_jerk < -1.0]

    print(f"\n  {'Decel jerk events (<-1)':>28} {len(decel_jerk_raw):>11} {len(decel_jerk_no_ttc):>14} {len(decel_jerk_ttc):>14}")
    if len(decel_jerk_raw) > 0:
      print(f"  {'P95 decel jerk (m/s3)':>28} {np.percentile(decel_jerk_raw, 5):>+11.1f} ", end="")
    else:
      print(f"  {'P95 decel jerk (m/s3)':>28} {'N/A':>11} ", end="")
    if len(decel_jerk_no_ttc) > 0:
      print(f"{np.percentile(decel_jerk_no_ttc, 5):>+13.1f} ", end="")
    else:
      print(f"{'N/A':>14} ", end="")
    if len(decel_jerk_ttc) > 0:
      print(f"{np.percentile(decel_jerk_ttc, 5):>+13.1f}")
    else:
      print(f"{'N/A':>14}")

# TTC bypass events
print(f"\n{'='*80}")
print(f"TTC BYPASS ACTIVATIONS")
print(f"{'='*80}")
print(f"\nTotal TTC bypass events: {len(ttc_bypass_events)}")

if ttc_bypass_events:
  print(f"\n  {'seg':>4} {'v_ego':>8} {'d_rel':>7} {'v_lead':>8} {'TTC':>6} {'aTarget':>9} {'prev_out':>10} {'held_back':>11}")
  print(f"  {'-'*70}")
  for ev in ttc_bypass_events[:25]:
    print(f"  {ev['seg']:>4} {ev['v_ego']:>7.0f}mph {ev['d_rel']:>6.0f}m {ev['v_lead']:>7.0f}mph "
          f"{ev['ttc']:>5.1f}s {ev['a_target']:>+8.2f} {ev['sim_prev']:>+9.2f} "
          f"{ev.get('delta_held_back', 0):>+10.2f}")
  if len(ttc_bypass_events) > 25:
    print(f"  ... and {len(ttc_bypass_events) - 25} more")

  # Summarize: what would have happened WITHOUT the bypass?
  print(f"\n  Impact summary:")
  total_decel_held = sum(max(0, ev["sim_prev"] - ev["a_target"]) for ev in ttc_bypass_events)
  print(f"    Total decel authority preserved by bypass: {total_decel_held:.1f} m/s2 across {len(ttc_bypass_events)} events")
  ttcs = [ev["ttc"] for ev in ttc_bypass_events]
  print(f"    TTC range: {min(ttcs):.1f}s - {max(ttcs):.1f}s")
  speeds = [ev["v_ego"] for ev in ttc_bypass_events]
  print(f"    Speed range: {min(speeds):.0f} - {max(speeds):.0f} mph")

# Comfort vs safety summary
print(f"\n{'='*80}")
print(f"SUMMARY: COMFORT vs SAFETY")
print(f"{'='*80}")

if len(decel_raw) > 0:
  comfort_smoothed = harsh_raw - harsh_ttc
  print(f"\n  Comfort (harsh jumps smoothed):        {comfort_smoothed} / {harsh_raw} eliminated")
  print(f"  Safety (TTC bypass activations):        {len(ttc_bypass_events)} events got full braking")
  if ttc_bypass_events:
    print(f"  Safety (lowest TTC during bypass):      {min(ttcs):.1f}s")
  print(f"  Safety (shouldStop bypass):             always active (not rate-limited)")

print()
