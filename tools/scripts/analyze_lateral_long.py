#!/usr/bin/env python3
"""Analyze both lateral (curve oversteer) and longitudinal (freeway oscillation) from qlogs."""
import math
import sys
import numpy as np
from collections import Counter

from openpilot.common.constants import CV
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

def b(obj, name, default=False):
  try:
    return bool(getattr(obj, name))
  except Exception:
    return default

def fmt_mps(v):
  if v is None or math.isnan(v):
    return "nan"
  return f"{v:.1f}m/s/{v / CV.MPH_TO_MS:.1f}mph"


def analyze(route):
  print(f"\n{'='*80}")
  print(f"=== {route} ===")
  print(f"{'='*80}")
  lr = LogReader(route, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)

  car_states = []
  radar_states = []
  plan_rows = []
  controls_states = []
  live_params = []
  live_torque = []
  model_states = []
  lat_plan_states = []
  start_t = None
  counts = Counter()
  car_params_info = {}

  for msg in lr:
    typ = msg.which()
    counts[typ] += 1
    t = msg.logMonoTime / 1e9
    if start_t is None:
      start_t = t
    rt = t - start_t

    if typ == "carParams":
      cp = msg.carParams
      car_params_info = {
        "fingerprint": str(capget(cp, "carFingerprint", "")),
        "steerRatio": f(cp, "steerRatio"),
        "wheelbase": f(cp, "wheelbase"),
        "steerActuatorDelay": f(cp, "steerActuatorDelay"),
        "opLong": b(cp, "openpilotLongitudinalControl"),
      }
      lt = capget(cp, "lateralTuning")
      if lt:
        which = str(lt.which()) if lt else "unknown"
        car_params_info["latTuningType"] = which
        if which == "torque":
          torque = lt.torque
          car_params_info["latAccelFactor"] = f(torque, "latAccelFactor")
          car_params_info["friction"] = f(torque, "friction")
        elif which == "pid":
          pid = lt.pid
          car_params_info["kf"] = f(pid, "kf")

    elif typ == "carState":
      cs = msg.carState
      car_states.append({
        "t": rt,
        "v_ego": f(cs, "vEgo"),
        "a_ego": f(cs, "aEgo"),
        "steer_angle": f(cs, "steeringAngleDeg"),
        "steer_torque": f(cs, "steeringTorque"),
        "standstill": b(cs, "standstill"),
        "v_cruise": f(cs, "vCruise") * CV.KPH_TO_MS,
      })

    elif typ == "radarState":
      lead = msg.radarState.leadOne
      radar_states.append({
        "t": rt, "lead_status": b(lead, "status"),
        "d_rel": f(lead, "dRel"), "v_rel": f(lead, "vRel"),
        "v_lead": f(lead, "vLead"),
      })

    elif typ == "controlsState":
      cst = msg.controlsState
      controls_states.append({
        "t": rt,
        "desired_curvature": f(cst, "desiredCurvature"),
        "curvature": f(cst, "curvature"),
        "steer_limited": b(cst, "steerLimitedBySafety") if hasattr(cst, "steerLimitedBySafety") else False,
      })

    elif typ == "liveParameters":
      lp = msg.liveParameters
      live_params.append({
        "t": rt,
        "steerRatio": f(lp, "steerRatio"),
        "angleOffsetDeg": f(lp, "angleOffsetDeg"),
        "angleOffsetAverageDeg": f(lp, "angleOffsetAverageDeg"),
        "roll": f(lp, "roll"),
      })

    elif typ == "liveTorqueParameters":
      ltp = msg.liveTorqueParameters
      live_torque.append({
        "t": rt,
        "latAccelFactorFiltered": f(ltp, "latAccelFactorFiltered"),
        "frictionCoefficientFiltered": f(ltp, "frictionCoefficientFiltered"),
        "useParams": b(ltp, "useParams"),
      })

    elif typ == "modelV2":
      m = msg.modelV2
      action = capget(m, "action")
      pos_y = list(capget(m, "position", {}).y) if capget(m, "position") else []
      model_states.append({
        "t": rt,
        "model_curvature": f(action, "desiredCurvature") if action else float("nan"),
        "model_accel": f(action, "desiredAcceleration") if action else float("nan"),
        "pos_y_2s": float(pos_y[10]) if len(pos_y) > 10 else float("nan"),
        "pos_y_4s": float(pos_y[20]) if len(pos_y) > 20 else float("nan"),
      })

    elif typ == "longitudinalPlan":
      lp = msg.longitudinalPlan
      speeds = capget(lp, "speeds", [])
      plan_rows.append({
        "t": rt, "a_target": f(lp, "aTarget"),
        "speed0": float(speeds[0]) if len(speeds) else float("nan"),
        "has_lead": b(lp, "hasLead"),
        "source": str(capget(lp, "longitudinalPlanSource", "")),
      })

  if not car_states:
    print("No carState messages found")
    return

  print(f"\ncounts: {', '.join(f'{k}={v}' for k, v in counts.most_common(15))}")

  # carParams
  if car_params_info:
    print(f"\n--- CarParams ---")
    for k, v in car_params_info.items():
      print(f"  {k}: {v}")

  # liveParameters summary
  if live_params:
    srs = [lp["steerRatio"] for lp in live_params if not math.isnan(lp["steerRatio"])]
    aos = [lp["angleOffsetDeg"] for lp in live_params if not math.isnan(lp["angleOffsetDeg"])]
    aoas = [lp["angleOffsetAverageDeg"] for lp in live_params if not math.isnan(lp["angleOffsetAverageDeg"])]
    rolls = [lp["roll"] for lp in live_params if not math.isnan(lp["roll"])]
    print(f"\n--- liveParameters ---")
    if srs: print(f"  steerRatio: min={min(srs):.2f} med={sorted(srs)[len(srs)//2]:.2f} max={max(srs):.2f}")
    if aos: print(f"  angleOffsetDeg: min={min(aos):.2f} med={sorted(aos)[len(aos)//2]:.2f} max={max(aos):.2f}")
    if aoas: print(f"  angleOffsetAverageDeg: min={min(aoas):.2f} med={sorted(aoas)[len(aoas)//2]:.2f} max={max(aoas):.2f}")
    if rolls: print(f"  roll: min={min(rolls):.4f} med={sorted(rolls)[len(rolls)//2]:.4f} max={max(rolls):.4f}")

  # liveTorqueParameters summary
  if live_torque:
    lafs = [lt["latAccelFactorFiltered"] for lt in live_torque if not math.isnan(lt["latAccelFactorFiltered"])]
    frcs = [lt["frictionCoefficientFiltered"] for lt in live_torque if not math.isnan(lt["frictionCoefficientFiltered"])]
    ups = [lt["useParams"] for lt in live_torque]
    print(f"\n--- liveTorqueParameters ---")
    if lafs: print(f"  latAccelFactor: min={min(lafs):.3f} med={sorted(lafs)[len(lafs)//2]:.3f} max={max(lafs):.3f}")
    if frcs: print(f"  friction: min={min(frcs):.3f} med={sorted(frcs)[len(frcs)//2]:.3f} max={max(frcs):.3f}")
    print(f"  useParams: {Counter(ups)}")

  # ============================================================
  # LATERAL: curve analysis
  # ============================================================
  def nearest(samples, t):
    if not samples: return None
    return min(samples, key=lambda s: abs(s["t"] - t))

  print(f"\n{'='*60}")
  print("LATERAL ANALYSIS — curve sections")
  print(f"{'='*60}")

  # Find sections where |desired_curvature| is elevated (turning)
  if controls_states and car_states:
    # Print timeline of high-curvature moments
    print("\n--- High-curvature moments (|desired_curv| > 0.003, every ~3s) ---")
    last_printed = -10.0
    for cs_row in controls_states:
      dc = cs_row["desired_curvature"]
      ac = cs_row["curvature"]
      if math.isnan(dc) or abs(dc) < 0.003:
        continue
      if cs_row["t"] - last_printed < 3.0:
        continue
      last_printed = cs_row["t"]
      car = nearest(car_states, cs_row["t"])
      mdl = nearest(model_states, cs_row["t"])
      lp = nearest(live_params, cs_row["t"])
      v = car["v_ego"] if car else float("nan")
      steer = car["steer_angle"] if car else float("nan")
      model_curv = mdl["model_curvature"] if mdl else float("nan")
      pos_y_2s = mdl["pos_y_2s"] if mdl else float("nan")
      pos_y_4s = mdl["pos_y_4s"] if mdl else float("nan")
      sr = lp["steerRatio"] if lp else float("nan")
      ao = lp["angleOffsetDeg"] if lp else float("nan")
      lat_accel = dc * v**2 if not math.isnan(v) else float("nan")
      print(
        f"  t={cs_row['t']:.1f}s vEgo={fmt_mps(v)} "
        f"desCurv={dc:+.5f} actCurv={ac:+.5f} err={dc-ac:+.5f} "
        f"latAccel={lat_accel:+.2f}m/s² steerAngle={steer:+.1f}° "
        f"modelCurv={model_curv:+.5f} posY2s={pos_y_2s:+.2f}m posY4s={pos_y_4s:+.2f}m "
        f"learnedSR={sr:.2f} AO={ao:+.2f}°"
      )

    # Curvature error stats on curves
    curv_errs = []
    for cs_row in controls_states:
      dc = cs_row["desired_curvature"]
      ac = cs_row["curvature"]
      if math.isnan(dc) or math.isnan(ac) or abs(dc) < 0.002:
        continue
      curv_errs.append(dc - ac)
    if curv_errs:
      arr = np.array(curv_errs)
      print(f"\n  Curvature tracking error (desired - actual) on curves:")
      print(f"    mean={np.mean(arr):+.5f} std={np.std(arr):.5f} min={np.min(arr):+.5f} max={np.max(arr):+.5f}")
      print(f"    If mean is systematically positive -> car is under-steering (model asks more than car delivers)")
      print(f"    If mean is systematically negative -> car is over-steering")

  # ============================================================
  # LONGITUDINAL: freeway oscillation
  # ============================================================
  print(f"\n{'='*60}")
  print("LONGITUDINAL ANALYSIS — freeway oscillation")
  print(f"{'='*60}")

  if plan_rows and car_states:
    freeway_lead = []
    for p in plan_rows:
      car = nearest(car_states, p["t"])
      rad = nearest(radar_states, p["t"])
      if car is None or rad is None: continue
      if not car["v_ego"] > 22.0: continue  # > ~50 mph
      if car["standstill"]: continue
      if not rad["lead_status"]: continue
      row = dict(p)
      row.update(car)
      row.update(rad)
      row["below_cruise"] = row["v_cruise"] - row["v_ego"]
      row["closing_speed"] = row["v_ego"] - row["v_lead"]
      freeway_lead.append(row)

    if freeway_lead:
      d_rels = [r["d_rel"] for r in freeway_lead]
      a_targets = [r["a_target"] for r in freeway_lead]
      v_egos = [r["v_ego"] for r in freeway_lead]
      source_counts = Counter(r["source"] for r in freeway_lead)
      print(f"\n  Freeway lead-following samples: {len(freeway_lead)}")
      print(f"  Sources: {dict(source_counts)}")
      print(f"  dRel: min={min(d_rels):.1f} med={sorted(d_rels)[len(d_rels)//2]:.1f} max={max(d_rels):.1f}")
      print(f"  aTarget: min={min(a_targets):.2f} med={sorted(a_targets)[len(a_targets)//2]:.2f} max={max(a_targets):.2f}")
      print(f"  vEgo: min={fmt_mps(min(v_egos))} max={fmt_mps(max(v_egos))}")

      # aTarget sign flips
      sign_flips = 0
      for i in range(1, len(freeway_lead)):
        pa = freeway_lead[i-1]["a_target"]
        ca = freeway_lead[i]["a_target"]
        if (pa > 0.05 and ca < -0.05) or (pa < -0.05 and ca > 0.05):
          sign_flips += 1
      print(f"  aTarget sign flips: {sign_flips}")

      # Source flips
      src_flips = 0
      for i in range(1, len(freeway_lead)):
        if freeway_lead[i]["source"] != freeway_lead[i-1]["source"]:
          src_flips += 1
      print(f"  source flips: {src_flips}")

      # dRel range (oscillation amplitude)
      # Window analysis: max - min dRel in 30-second windows
      if len(freeway_lead) > 10:
        print(f"\n--- Freeway lead timeline (every ~3s) ---")
        last_printed = -10.0
        for r in freeway_lead:
          if r["t"] - last_printed < 3.0: continue
          last_printed = r["t"]
          print(
            f"  t={r['t']:.1f}s vEgo={fmt_mps(r['v_ego'])} vCruise={fmt_mps(r['v_cruise'])} "
            f"aTarget={r['a_target']:+.2f} aEgo={r['a_ego']:+.2f} "
            f"source={r['source']} dRel={r['d_rel']:.1f}m "
            f"vLead={fmt_mps(r['v_lead'])} closing={r['closing_speed']:.1f} "
            f"belowCruise={r['below_cruise']:.1f}"
          )
    else:
      print("  No freeway lead-following samples found")
  else:
    print("  Missing longitudinalPlan or carState data")


if __name__ == "__main__":
  for route_name in sys.argv[1:]:
    analyze(route_name)
