#!/usr/bin/env python3
"""Deep analysis of braking events and curve tracking."""
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
segments = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else list(range(30))

all_braking = []
all_curve = []
all_accel_timeline = []

for seg in segments:
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  events = list(lr)
  cs_data = {}
  lp_data = {}
  radar_data = {}
  ctrl_data = {}

  for msg in events:
    typ = msg.which()
    t = msg.logMonoTime / 1e9

    if typ == "carState":
      cs = msg.carState
      cs_data[t] = {
        "vEgo": f(cs, "vEgo"),
        "aEgo": f(cs, "aEgo"),
        "brakePressed": bool(capget(cs, "brakePressed", False)),
        "gasPressed": bool(capget(cs, "gasPressed", False)),
      }

    elif typ == "longitudinalPlan":
      lp = msg.longitudinalPlan
      src = str(capget(lp, "longitudinalPlanSource", ""))
      lp_data[t] = {
        "aTarget": f(lp, "aTarget"),
        "source": src,
        "hasLead": bool(capget(lp, "hasLead", False)),
        "vTarget": f(lp, "vTarget") if hasattr(lp, "vTarget") else float("nan"),
      }

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      l1 = capget(rs, "leadTwo")
      entry = {}
      if l0 and capget(l0, "status", False):
        entry["l0_dRel"] = f(l0, "dRel")
        entry["l0_vLead"] = f(l0, "vLead")
        entry["l0_aLeadK"] = f(l0, "aLeadK")
        entry["l0_status"] = True
      else:
        entry["l0_status"] = False
      if l1 and capget(l1, "status", False):
        entry["l1_dRel"] = f(l1, "dRel")
        entry["l1_vLead"] = f(l1, "vLead")
        entry["l1_status"] = True
      else:
        entry["l1_status"] = False
      radar_data[t] = entry

    elif typ == "controlsState":
      dc = f(msg.controlsState, "desiredCurvature")
      ac = f(msg.controlsState, "curvature")
      ctrl_data[t] = {
        "desCurv": dc,
        "actCurv": ac,
      }

  lp_times = sorted(lp_data.keys())
  radar_times = sorted(radar_data.keys())
  cs_times = sorted(cs_data.keys())
  ctrl_times = sorted(ctrl_data.keys())

  def nearest(times_dict, t, max_dt=0.15):
    times = sorted(times_dict.keys())
    if not times:
      return None
    idx = np.searchsorted(times, t)
    best = None
    for i in [idx-1, idx]:
      if 0 <= i < len(times) and abs(times[i] - t) < max_dt:
        if best is None or abs(times[i] - t) < abs(best - t):
          best = times[i]
    return times_dict[best] if best is not None else None

  for t in lp_times:
    lp = lp_data[t]
    cs = nearest(cs_data, t)
    rd = nearest(radar_data, t)
    ct = nearest(ctrl_data, t)

    if cs is None:
      continue

    v_ego = cs["vEgo"]
    a_ego = cs["aEgo"]
    a_target = lp["aTarget"]

    d_rel = rd.get("l0_dRel", float("nan")) if rd and rd.get("l0_status") else float("nan")
    v_lead = rd.get("l0_vLead", float("nan")) if rd and rd.get("l0_status") else float("nan")
    a_lead = rd.get("l0_aLeadK", float("nan")) if rd and rd.get("l0_status") else float("nan")

    all_accel_timeline.append({
      "t": t, "seg": seg, "v_ego": v_ego, "a_ego": a_ego,
      "a_target": a_target, "source": lp["source"],
      "d_rel": d_rel, "v_lead": v_lead, "a_lead": a_lead,
    })

    if a_target < -0.8 and v_ego > 2.0:
      closing = (v_ego - v_lead) if not math.isnan(v_lead) else float("nan")
      all_braking.append({
        "t": t, "seg": seg, "v_ego": v_ego, "v_mph": v_ego/0.44704,
        "a_ego": a_ego, "a_target": a_target, "source": lp["source"],
        "d_rel": d_rel, "v_lead": v_lead, "a_lead": a_lead,
        "closing": closing,
      })

    if ct:
      des = ct["desCurv"]
      act = ct["actCurv"]
      if not math.isnan(des) and not math.isnan(act) and abs(des) > 0.002 and v_ego > 3.0:
        err = des - act
        all_curve.append({
          "t": t, "seg": seg, "v_ego": v_ego, "v_mph": v_ego/0.44704,
          "desCurv": des, "actCurv": act, "err": err,
          "lat_accel": des * v_ego**2,
        })

print("=" * 100)
print("BRAKING EVENTS (aTarget < -0.8 m/s^2, v > 2 m/s)")
print("=" * 100)
print(f"{'seg':>4} {'t':>8} {'v_mph':>7} {'aTarget':>8} {'aEgo':>6} {'source':>8} {'dRel':>6} {'vLead':>6} {'aLead':>6} {'closing':>7}")
print("-" * 100)
for b in all_braking:
  print(f"{b['seg']:4d} {b['t']:8.1f} {b['v_mph']:6.1f}mph {b['a_target']:+7.2f} {b['a_ego']:+5.2f} {b['source']:>8} "
        f"{b['d_rel']:5.1f}m {b['v_lead']:5.1f}ms {b['a_lead']:+5.2f} {b['closing']:+6.1f}ms")

print()
print("=" * 100)
print("BRAKING JERK ANALYSIS (rate of change of aTarget)")
print("=" * 100)

for seg in segments:
  seg_points = [p for p in all_accel_timeline if p["seg"] == seg and p["v_ego"] > 2.0]
  if len(seg_points) < 2:
    continue
  jerks = []
  for i in range(1, len(seg_points)):
    dt = seg_points[i]["t"] - seg_points[i-1]["t"]
    if 0.01 < dt < 0.5:
      da = seg_points[i]["a_target"] - seg_points[i-1]["a_target"]
      jerk = da / dt
      if abs(jerk) > 2.0:
        jerks.append({
          "t": seg_points[i]["t"],
          "jerk": jerk,
          "a_from": seg_points[i-1]["a_target"],
          "a_to": seg_points[i]["a_target"],
          "v_ego": seg_points[i]["v_ego"],
          "d_rel": seg_points[i]["d_rel"],
          "source": seg_points[i]["source"],
        })
  if jerks:
    harsh = [j for j in jerks if abs(j["jerk"]) > 5.0]
    print(f"  Seg {seg}: {len(jerks)} high-jerk moments (|jerk|>2), {len(harsh)} harsh (|jerk|>5)")
    for j in sorted(jerks, key=lambda x: abs(x["jerk"]), reverse=True)[:5]:
      print(f"    t={j['t']:.1f} jerk={j['jerk']:+.1f}m/s^3 aTarget: {j['a_from']:+.2f}->{j['a_to']:+.2f} "
            f"v={j['v_ego']/0.44704:.0f}mph d={j['d_rel']:.0f}m src={j['source']}")

print()
print("=" * 100)
print("CURVE TRACKING - ALL MOMENTS (|desCurv| > 0.002, v > 3 m/s)")
print("=" * 100)
print(f"{'seg':>4} {'t':>8} {'v_mph':>7} {'desCurv':>9} {'actCurv':>9} {'err':>9} {'latAccel':>9}")
print("-" * 90)
for c in all_curve:
  flag = ""
  if abs(c["err"]) > 0.003:
    flag = " << BIG ERROR"
  elif abs(c["err"]) > 0.002:
    flag = " < moderate"
  print(f"{c['seg']:4d} {c['t']:8.1f} {c['v_mph']:6.1f}mph {c['desCurv']:+8.5f} {c['actCurv']:+8.5f} "
        f"{c['err']:+8.5f} {c['lat_accel']:+8.2f}m/s2{flag}")

print()
print("=" * 100)
print("OVERALL BRAKING ROUGHNESS - aTarget distribution while decelerating")
print("=" * 100)
decel_targets = [p["a_target"] for p in all_accel_timeline if p["a_target"] < -0.1 and p["v_ego"] > 2.0]
if decel_targets:
  arr = np.array(decel_targets)
  print(f"  Total deceleration samples: {len(arr)}")
  print(f"  Median: {np.median(arr):+.2f} m/s^2")
  print(f"  P10: {np.percentile(arr, 10):+.2f} m/s^2")
  print(f"  P25: {np.percentile(arr, 25):+.2f} m/s^2")
  print(f"  P75: {np.percentile(arr, 75):+.2f} m/s^2")
  print(f"  P90: {np.percentile(arr, 90):+.2f} m/s^2")
  print(f"  Min (hardest): {np.min(arr):+.2f} m/s^2")

  bins = [(-0.5, -0.1), (-1.0, -0.5), (-1.5, -1.0), (-2.0, -1.5), (-3.5, -2.0)]
  for lo, hi in bins:
    count = np.sum((arr >= lo) & (arr < hi))
    pct = count / len(arr) * 100
    label = "gentle" if hi >= -0.5 else "moderate" if hi >= -1.0 else "firm" if hi >= -1.5 else "hard" if hi >= -2.0 else "emergency"
    print(f"  {lo:+.1f} to {hi:+.1f}: {count:5d} ({pct:5.1f}%) [{label}]")
