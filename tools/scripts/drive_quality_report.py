#!/usr/bin/env python3
"""
Comprehensive drive quality diagnostics.
Evaluates both lateral and longitudinal control quality against
community benchmarks (ISO standards, academic research, openpilot community).

Works with qlogs. Usage:
  python drive_quality_report.py <route_prefix> [segment_start] [segment_end]
"""
import sys
import math
import json
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

# ─── Accumulators ───
speed_groups = {
  "crawl":    (0, 5,   "0-11 mph"),
  "slow":     (5, 15,  "11-34 mph"),
  "medium":   (15, 25, "34-56 mph"),
  "fast":     (25, 35, "56-78 mph"),
  "veryfast": (35, 99, "78+ mph"),
}

lat_stats = {g: {"error": [], "saturated": 0, "total": 0, "des_curv": [], "act_curv": [],
                 "lat_accel": [], "steer_torque": []} for g in speed_groups}

long_stats = {
  "a_target": [], "a_ego": [], "a_error": [],
  "jerk": [], "v_ego": [],
  "source_flips": 0, "source_timeline": [],
  "brake_events": [],
  "gas_interventions": 0, "brake_interventions": 0,
  "engaged_samples": 0, "total_samples": 0,
  "dRel": [], "following_gap_error": [],
}

prev_source = None
prev_a_target = None
prev_t = None
engaged_since = 0
debug_counts = defaultdict(int)

for seg in range(seg_start, seg_end):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  cs_latest = {}
  ctrl_latest = {}
  radar_latest = {}
  sds_latest = {}

  for msg in lr:
    typ = msg.which()
    debug_counts[typ] += 1
    t = msg.logMonoTime / 1e9

    if typ == "carState":
      cs = msg.carState
      v_ego = f(cs, "vEgo")
      a_ego = f(cs, "aEgo")
      gas = bool(capget(cs, "gasPressed", False))
      brake = bool(capget(cs, "brakePressed", False))
      standstill = bool(capget(cs, "standstill", False))
      steer_override = bool(capget(cs, "steeringPressed", False))
      steer_torque = f(cs, "steeringTorque", 0.0)
      cs_latest = {"v_ego": v_ego, "a_ego": a_ego, "gas": gas, "brake": brake,
                   "standstill": standstill, "steer_override": steer_override,
                   "steer_torque": steer_torque, "t": t}

    elif typ == "selfdriveState":
      sds = msg.selfdriveState
      active = bool(capget(sds, "active", False))
      enabled = bool(capget(sds, "enabled", False))
      sds_latest = {"active": active or enabled, "t": t}

    elif typ == "controlsState":
      cst = msg.controlsState
      active_cs = bool(capget(cst, "active", False))
      des_curv = f(cst, "desiredCurvature")
      act_curv = f(cst, "curvature")
      lat_state = capget(cst, "lateralControlState")
      saturated = False
      if lat_state:
        ts = capget(lat_state, "torqueState")
        if ts:
          saturated = bool(capget(ts, "saturated", False))
      ctrl_latest = {"active_cs": active_cs, "des_curv": des_curv, "act_curv": act_curv,
                     "saturated": saturated, "t": t}

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      if l0 and capget(l0, "status", False):
        radar_latest = {"dRel": f(l0, "dRel"), "vLead": f(l0, "vLead"), "status": True}
      else:
        radar_latest = {"status": False}

    elif typ in ("longitudinalPlan", "longitudinalPlanSP"):
      if typ == "longitudinalPlanSP":
        lp = msg.longitudinalPlanSP
      else:
        lp = msg.longitudinalPlan
      a_target = f(lp, "aTarget")
      source = str(capget(lp, "longitudinalPlanSource", ""))

      long_stats["total_samples"] += 1

      if not cs_latest or (not ctrl_latest and not sds_latest):
        continue

      v_ego = cs_latest.get("v_ego", 0)
      a_ego = cs_latest.get("a_ego", 0)
      active = sds_latest.get("active", False) or ctrl_latest.get("active_cs", False)

      if not active or cs_latest.get("standstill", True):
        engaged_since = 0
        prev_source = None
        prev_a_target = None
        continue

      engaged_since += 1
      long_stats["engaged_samples"] += 1
      if "first_engaged_t" not in long_stats:
        long_stats["first_engaged_t"] = t
      long_stats["last_engaged_t"] = t

      if engaged_since < 100:
        continue

      if cs_latest.get("gas", False):
        long_stats["gas_interventions"] += 1
      if cs_latest.get("brake", False):
        long_stats["brake_interventions"] += 1

      if not math.isnan(a_target) and not math.isnan(a_ego) and v_ego > 1.0:
        long_stats["a_target"].append(a_target)
        long_stats["a_ego"].append(a_ego)
        long_stats["a_error"].append(abs(a_target - a_ego))
        long_stats["v_ego"].append(v_ego)

        if prev_a_target is not None and prev_t is not None:
          dt = t - prev_t
          if 0.01 < dt < 1.0:
            jerk = (a_target - prev_a_target) / dt
            long_stats["jerk"].append(jerk)

        if a_target < -0.8:
          d = radar_latest.get("dRel", float("nan")) if radar_latest.get("status") else float("nan")
          long_stats["brake_events"].append({"a": a_target, "v": v_ego, "d": d})

      if prev_source and source != prev_source:
        long_stats["source_flips"] += 1
      long_stats["source_timeline"].append(source)
      prev_source = source
      prev_a_target = a_target
      prev_t = t

      if radar_latest.get("status") and v_ego > 5.0:
        d = radar_latest["dRel"]
        if not math.isnan(d):
          long_stats["dRel"].append(d)
          t_follow = 1.45
          bonus = float(np.interp(v_ego, [20.0, 28.0], [0.0, 0.30]))
          t_eff = t_follow + bonus
          v_lead = radar_latest.get("vLead", v_ego)
          d_safe = (v_ego**2) / (2 * 2.5) + t_eff * v_ego + 6.0 - (v_lead**2) / (2 * 2.5)
          if d_safe > 5:
            gap_err = (d - d_safe) / d_safe
            long_stats["following_gap_error"].append(gap_err)

      # Lateral stats
      if not cs_latest.get("steer_override", True) and v_ego > 1.0:
        des_curv = ctrl_latest.get("des_curv", float("nan"))
        act_curv = ctrl_latest.get("act_curv", float("nan"))
        if not math.isnan(des_curv) and not math.isnan(act_curv):
          for g, (lo, hi, _) in speed_groups.items():
            if lo <= v_ego < hi:
              curv_err = abs(des_curv - act_curv)
              lat_a = abs(des_curv) * v_ego**2
              lat_stats[g]["error"].append(curv_err)
              lat_stats[g]["des_curv"].append(abs(des_curv))
              lat_stats[g]["act_curv"].append(abs(act_curv))
              lat_stats[g]["lat_accel"].append(lat_a)
              lat_stats[g]["total"] += 1
              if ctrl_latest.get("saturated", False):
                lat_stats[g]["saturated"] += 1
              break

# ═══════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════

print("=" * 80)
print(f"DRIVE QUALITY REPORT: {route}")
print(f"Segments: {seg_start}-{seg_end}")
print(f"Total samples: {long_stats['total_samples']}, Engaged: {long_stats['engaged_samples']}")
top_types = sorted(debug_counts.items(), key=lambda x: -x[1])[:15]
print(f"Message types: {', '.join(f'{k}:{v}' for k,v in top_types)}")
print("=" * 80)

# ─── LATERAL ───
print("\n" + "=" * 80)
print("LATERAL CONTROL QUALITY")
print("=" * 80)
print(f"{'Group':<10} {'Samples':>8} {'Curv Err':>10} {'Lat Accel':>12} {'P95 LatA':>10} {'Sat%':>6}")
print("-" * 60)

for g in ["crawl", "slow", "medium", "fast", "veryfast"]:
  lo, hi, label = speed_groups[g]
  s = lat_stats[g]
  n = s["total"]
  if n < 10:
    print(f"{g:<10} {n:>8} {'(insufficient data)':>30}")
    continue
  errs = np.array(s["error"])
  lat_a = np.array(s["lat_accel"])
  sat_pct = s["saturated"] / n * 100
  print(f"{g:<10} {n:>8} {np.mean(errs):>9.5f} {np.mean(lat_a):>10.3f}m/s2 {np.percentile(lat_a, 95):>8.2f} {sat_pct:>5.1f}%")

# Benchmarks
print("\nBenchmarks:")
print("  Curv error: < 0.002 good, < 0.005 acceptable, > 0.005 needs work")
print("  Lat accel (highway): < 2.0 good, 2.0-3.0 acceptable, > 3.0 needs work")
print("  Saturation: < 1% good, < 5% acceptable, > 10% needs work")

# ─── LONGITUDINAL ───
print("\n" + "=" * 80)
print("LONGITUDINAL CONTROL QUALITY")
print("=" * 80)

if long_stats["a_error"]:
  a_err = np.array(long_stats["a_error"])
  a_targets = np.array(long_stats["a_target"])
  a_egos = np.array(long_stats["a_ego"])
  jerks = np.array(long_stats["jerk"]) if long_stats["jerk"] else np.array([0])

  print(f"\naTarget vs aEgo tracking:")
  print(f"  Mean absolute error: {np.mean(a_err):.3f} m/s2  {'GOOD' if np.mean(a_err) < 0.1 else 'ACCEPTABLE' if np.mean(a_err) < 0.3 else 'NEEDS WORK'}")
  print(f"  Median error: {np.median(a_err):.3f} m/s2")
  print(f"  P95 error: {np.percentile(a_err, 95):.3f} m/s2")
  print(f"  Max error: {np.max(a_err):.3f} m/s2")

  print(f"\nAcceleration distribution:")
  print(f"  aTarget range: [{np.min(a_targets):+.2f}, {np.max(a_targets):+.2f}] m/s2")
  print(f"  aEgo range:    [{np.min(a_egos):+.2f}, {np.max(a_egos):+.2f}] m/s2")
  print(f"  Median aTarget: {np.median(a_targets):+.3f} m/s2")

  print(f"\nJerk profile (rate of aTarget change):")
  print(f"  Mean |jerk|: {np.mean(np.abs(jerks)):.2f} m/s3  {'GOOD' if np.mean(np.abs(jerks)) < 2.0 else 'ACCEPTABLE' if np.mean(np.abs(jerks)) < 5.0 else 'NEEDS WORK'}")
  print(f"  P95 |jerk|: {np.percentile(np.abs(jerks), 95):.2f} m/s3")
  print(f"  P99 |jerk|: {np.percentile(np.abs(jerks), 99):.2f} m/s3")
  print(f"  Max |jerk|: {np.max(np.abs(jerks)):.2f} m/s3")
  harsh_jerk = np.sum(np.abs(jerks) > 5.0)
  print(f"  Harsh jerk events (>5 m/s3): {harsh_jerk} ({harsh_jerk/len(jerks)*100:.1f}%)")

  print(f"\nDeceleration harshness:")
  decel = a_targets[a_targets < -0.1]
  if len(decel) > 0:
    bins = [(-0.5, "gentle"), (-1.0, "moderate"), (-1.5, "firm"), (-2.0, "hard"), (-99, "emergency")]
    print(f"  Total decel samples: {len(decel)}")
    for threshold, label in bins:
      count = np.sum(decel < threshold) if threshold > -99 else 0
      pct = np.sum(decel < threshold) / len(decel) * 100 if threshold > -99 else 0
      if threshold > -99:
        print(f"  Beyond {threshold:+.1f} ({label}): {int(np.sum(decel < threshold)):>5} ({pct:.1f}%)")

  print(f"\nSource stability:")
  if long_stats["source_timeline"]:
    total_source = len(long_stats["source_timeline"])
    from collections import Counter
    src_counts = Counter(long_stats["source_timeline"])
    drive_minutes = long_stats["engaged_samples"] / 20.0 / 60.0  # approximate from engaged count at ~20Hz qlog
    if long_stats.get("first_engaged_t") and long_stats.get("last_engaged_t"):
      drive_minutes = (long_stats["last_engaged_t"] - long_stats["first_engaged_t"]) / 60.0
    flips_per_min = long_stats["source_flips"] / max(drive_minutes, 0.1)
    print(f"  Drive time (engaged): {drive_minutes:.1f} min")
    print(f"  Total source flips: {long_stats['source_flips']}")
    print(f"  Flips per minute: {flips_per_min:.1f}  {'GOOD' if flips_per_min < 5 else 'ACCEPTABLE' if flips_per_min < 15 else 'NEEDS WORK'}")
    print(f"  Source distribution: {dict(src_counts.most_common(5))}")

  print(f"\nFollowing distance quality:")
  if long_stats["following_gap_error"]:
    gap_err = np.array(long_stats["following_gap_error"])
    print(f"  Mean gap error vs d_safe: {np.mean(gap_err)*100:+.1f}%  {'GOOD' if abs(np.mean(gap_err)) < 0.1 else 'ACCEPTABLE' if abs(np.mean(gap_err)) < 0.2 else 'NEEDS WORK'}")
    print(f"  Median gap error: {np.median(gap_err)*100:+.1f}%")
    print(f"  P10/P90: {np.percentile(gap_err, 10)*100:+.1f}% / {np.percentile(gap_err, 90)*100:+.1f}%")

  print(f"\nDriver interventions:")
  print(f"  Gas presses while engaged: {long_stats['gas_interventions']}")
  print(f"  Brake presses while engaged: {long_stats['brake_interventions']}")

# ─── OVERALL GRADE ───
print("\n" + "=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)

issues = []
improvements = []

if long_stats["a_error"]:
  mean_err = np.mean(np.array(long_stats["a_error"]))
  if mean_err > 0.3:
    issues.append(f"aTarget/aEgo tracking error is high ({mean_err:.2f} m/s2)")
  elif mean_err > 0.1:
    improvements.append(f"aTarget/aEgo tracking could be tighter ({mean_err:.2f} m/s2)")

if long_stats["jerk"]:
  p95_jerk = np.percentile(np.abs(np.array(long_stats["jerk"])), 95)
  if p95_jerk > 5.0:
    issues.append(f"P95 jerk is harsh ({p95_jerk:.1f} m/s3)")
  elif p95_jerk > 2.0:
    improvements.append(f"P95 jerk is noticeable ({p95_jerk:.1f} m/s3)")

if long_stats["source_timeline"]:
  if flips_per_min > 15:
    issues.append(f"Source flipping is excessive ({flips_per_min:.0f}/min)")
  elif flips_per_min > 5:
    improvements.append(f"Source flipping is moderate ({flips_per_min:.0f}/min)")

for g in ["medium", "fast", "veryfast"]:
  s = lat_stats[g]
  if s["total"] > 10:
    sat_pct = s["saturated"] / s["total"] * 100
    if sat_pct > 10:
      issues.append(f"Lateral saturation high in {g} ({sat_pct:.0f}%)")
    elif sat_pct > 5:
      improvements.append(f"Lateral saturation moderate in {g} ({sat_pct:.0f}%)")

    p95_la = np.percentile(np.array(s["lat_accel"]), 95)
    if p95_la > 3.0:
      issues.append(f"P95 lateral accel high in {g} ({p95_la:.1f} m/s2)")
    elif p95_la > 2.0:
      improvements.append(f"P95 lateral accel moderate in {g} ({p95_la:.1f} m/s2)")

if issues:
  print("\nISSUES (should fix):")
  for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")
else:
  print("\nNo critical issues found.")

if improvements:
  print("\nIMPROVEMENTS (nice to have):")
  for i, imp in enumerate(improvements, 1):
    print(f"  {i}. {imp}")
else:
  print("\nNo suggested improvements.")

print()
