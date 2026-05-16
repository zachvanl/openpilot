#!/usr/bin/env python3
"""Frame-by-frame longitudinal analysis for process-replay vs original route."""
import argparse
import math
import sys
from bisect import bisect_left

import numpy as np
from openpilot.tools.lib.logreader import LogReader, ReadMode

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CITY_CLOSING_MIN_SPEED,
  CITY_JERK_BYPASS_TTC,
  CITY_FOLLOW_CAP_MAX_D_REL,
  CITY_FOLLOW_CAP_MAX_V_EGO,
  CITY_FOLLOW_CAP_MAX_V_LEAD,
  CITY_STOP_MAX_D_REL,
  CITY_STOP_MAX_V_EGO,
  CITY_STOP_MAX_V_LEAD,
  OUTPUT_DECEL_EMERGENCY_DIST,
  OUTPUT_DECEL_TTC_BYPASS,
  cap_v_cruise_for_slow_lead,
  city_closing_brake_active,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import GENTLE_FAR_LEAD_SPEED_MAX

ACCEL_EMERGENCY = -3.4


def capget(o, n, d=None):
  try:
    return getattr(o, n)
  except Exception:
    return d


def f(o, n, d=float("nan")):
  try:
    return float(getattr(o, n))
  except Exception:
    return d


def load_frames(lr_iter, t_lo: float, t_hi: float):
  """Build per-plan frames with nearest carState / radar / model."""
  car = []  # (t, v_ego)
  radar = []  # (t, status, d, vl, model_prob)
  model = []  # (t, a_e2e, stop_e2e, hard_brake)
  sp = []  # (t, v_target, source)
  plans = []

  for msg in lr_iter:
    t = msg.logMonoTime / 1e9
    w = msg.which()
    if w == "carState":
      car.append((t, f(msg.carState, "vEgo")))
    elif w == "radarState":
      l0 = capget(msg.radarState, "leadOne")
      if l0:
        radar.append((t, bool(capget(l0, "status")), f(l0, "dRel"), f(l0, "vLead"),
                      f(l0, "modelProb", 0.0)))
    elif w == "modelV2":
      act = capget(msg.modelV2, "action")
      meta = capget(msg.modelV2, "meta")
      model.append((t, f(act, "desiredAcceleration"), bool(capget(act, "shouldStop", False)),
                    bool(capget(meta, "hardBrakePredicted", False))))
    elif w == "longitudinalPlanSP":
      sp.append((t, f(msg.longitudinalPlanSP, "vTarget"), str(capget(msg.longitudinalPlanSP, "longitudinalPlanSource", ""))))
    elif w == "longitudinalPlan":
      if t_lo <= t <= t_hi:
        lp = msg.longitudinalPlan
        plans.append({
          "t": t,
          "a": f(lp, "aTarget"),
          "src": str(capget(lp, "longitudinalPlanSource", "")),
          "stop": bool(capget(lp, "shouldStop", False)),
          "has_lead": bool(capget(lp, "hasLead", False)),
          "fcw": bool(capget(lp, "fcw", False)),
        })

  def nearest(series, t, max_dt=0.12, model_series=False):
    if model_series:
      max_dt = 0.35  # qlog modelV2 is sparse vs plan
    if not series:
      return None
    ts = [s[0] for s in series]
    i = bisect_left(ts, t)
    best = None
    for j in (i - 1, i):
      if 0 <= j < len(series) and (best is None or abs(ts[j] - t) < abs(best[0] - t)):
        best = series[j]
    if best is None or abs(best[0] - t) > max_dt:
      return None
    return best

  frames = []
  for p in plans:
    t = p["t"]
    c = nearest(car, t)
    r = nearest(radar, t)
    m = nearest(model, t, model_series=True)
    s = nearest(sp, t)
    v_ego = c[1] if c else float("nan")
    if r:
      status, d_rel, v_lead, mprob = r[1], r[2], r[3], r[4]
    else:
      status, d_rel, v_lead, mprob = False, float("nan"), float("nan"), float("nan")

    class Lead:
      status = False
      dRel = 0.0
      vLead = 0.0

    lead = Lead()
    lead.status = status
    lead.dRel = d_rel if status else 999.0
    lead.vLead = v_lead if status else 0.0

    v_cruise_sp = s[1] if s else float("nan")
    v_cruise_cap = cap_v_cruise_for_slow_lead(v_cruise_sp, lead if status else None, v_ego) if s and status else v_cruise_sp

    closing = max(v_ego - max(v_lead, 0.0), 0.0) if status and not math.isnan(v_ego) else float("nan")
    ttc = (d_rel / closing) if status and closing > 0.1 else float("inf")

    emerg_dist = status and d_rel < OUTPUT_DECEL_EMERGENCY_DIST
    ttc_lim = CITY_JERK_BYPASS_TTC if v_ego < GENTLE_FAR_LEAD_SPEED_MAX else OUTPUT_DECEL_TTC_BYPASS
    emerg_ttc = status and ttc < ttc_lim
    city_close = city_closing_brake_active(lead if status else None, v_ego) if status else False

    a_e2e = m[1] if m else float("nan")
    stop_e2e = m[2] if m else False
    hard_brake = m[3] if m else False

    frames.append({
      **p,
      "v_ego": v_ego,
      "v_mph": v_ego / 0.44704 if not math.isnan(v_ego) else float("nan"),
      "lead": status,
      "d": d_rel if status else float("nan"),
      "vl": v_lead if status else float("nan"),
      "vl_mph": (v_lead / 0.44704) if status else float("nan"),
      "mprob": mprob if status else float("nan"),
      "closing": closing,
      "ttc": ttc,
      "v_cruise_sp": v_cruise_sp,
      "v_cruise_cap": v_cruise_cap,
      "a_e2e": a_e2e,
      "stop_e2e": stop_e2e,
      "hard_brake": hard_brake,
      "emerg_dist": emerg_dist,
      "emerg_ttc": emerg_ttc,
      "city_close": city_close,
      "emerg_any": emerg_dist or emerg_ttc,
    })
  return frames


def print_table(label: str, frames: list, sample_dt: float = 0.25):
  print(f"\n{'='*100}")
  print(label)
  if not frames:
    print("  (no frames)")
    return

  hdr = (
    "     t  mph  lead   d(m) vl(mph) close  ttc  vCrSP vCrCap  aE2e   aTgt  src      "
    "stop e2eStp fcw city emerg"
  )
  print(hdr)
  print("-" * len(hdr))

  last_t = -1e9
  for fr in frames:
    if fr["t"] - last_t < sample_dt and fr["a"] > ACCEL_EMERGENCY and not fr["emerg_any"]:
      continue
    last_t = fr["t"]
    mark = " ***" if fr["a"] <= ACCEL_EMERGENCY else (" !!" if fr["emerg_any"] and fr["a"] < -1.0 else "")
    print(
      f"{fr['t']:7.2f} {fr['v_mph']:4.0f} "
      f"{'Y' if fr['lead'] else 'n':4} "
      f"{fr['d']:6.0f} {fr['vl_mph']:5.1f} "
      f"{fr['closing']:5.1f} {fr['ttc']:5.1f} "
      f"{fr['v_cruise_sp']:5.1f} {fr['v_cruise_cap']:6.1f} "
      f"{fr['a_e2e']:+5.2f} {fr['a']:+6.2f} {fr['src'][:8]:8} "
      f"{'S' if fr['stop'] else '.'}{'S' if fr['stop_e2e'] else '.'} "
      f"{'F' if fr['fcw'] else '.'} "
      f"{'C' if fr['city_close'] else '.'} "
      f"{'D' if fr['emerg_dist'] else '.'}{'T' if fr['emerg_ttc'] else '.'}"
      f"{mark}"
    )


def print_transitions(label: str, frames: list):
  print(f"\n--- transitions ({label}) ---")
  prev = None
  for fr in frames:
    if prev is None:
      prev = fr
      continue
    notes = []
    if fr["lead"] != prev["lead"]:
      notes.append(f"lead {'ON' if fr['lead'] else 'OFF'}")
    if fr["a"] <= ACCEL_EMERGENCY and prev["a"] > ACCEL_EMERGENCY:
      notes.append("HIT -3.5")
    if fr["emerg_any"] and not prev["emerg_any"]:
      notes.append("emergency bypass ON")
    if fr["city_close"] and not prev["city_close"]:
      notes.append("city_close ON")
    if fr["src"] != prev["src"]:
      notes.append(f"src {prev['src']}->{fr['src']}")
    if abs(fr["a"] - prev["a"]) > 0.4:
      notes.append(f"a {prev['a']:+.2f}->{fr['a']:+.2f}")
    if notes:
      print(f"  t={fr['t']:.2f}  " + "; ".join(notes))
    prev = fr


def align_compare(orig_frames, new_frames, max_dt=0.12):
  print(f"\n{'='*100}")
  print("ALIGNED OLD vs NEW (plan times from original)")
  hdr = "     t  d  vl  ttc  aE2e  aOLD  aNEW  dNEW  srcOLD->NEW  flags"
  print(hdr)
  print("-" * len(hdr))
  for o in orig_frames:
    best = min(new_frames, key=lambda n: abs(n["t"] - o["t"]), default=None)
    if best is None or abs(best["t"] - o["t"]) > max_dt:
      continue
    flags = []
    if o["emerg_any"]:
      flags.append("emerg")
    if o["city_close"]:
      flags.append("city")
    if best["a"] <= ACCEL_EMERGENCY:
      flags.append("NEW-35")
    print(
      f"{o['t']:7.2f} {o['d']:4.0f} {o['vl_mph']:4.1f} {o['ttc']:4.1f} "
      f"{o['a_e2e']:+5.2f} {o['a']:+6.2f} {best['a']:+6.2f} {best['d']:4.0f} "
      f"{o['src'][:6]}->{best['src'][:6]}  {','.join(flags)}"
    )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("orig_route_seg", help="e.g. 987638facc544f63/00000080--0684ca20c8/7/q")
  ap.add_argument("regen_rlog")
  ap.add_argument("--t-lo", type=float, default=445.0)
  ap.add_argument("--t-hi", type=float, default=453.0)
  ap.add_argument("--sample-dt", type=float, default=0.25)
  ap.add_argument("--regen-t-lo", type=float, default=None, help="Regen log time window (defaults to --t-lo)")
  ap.add_argument("--regen-t-hi", type=float, default=None, help="Regen log time window (defaults to --t-hi)")
  args = ap.parse_args()

  regen_lo = args.t_lo if args.regen_t_lo is None else args.regen_t_lo
  regen_hi = args.t_hi if args.regen_t_hi is None else args.regen_t_hi

  orig_lr = LogReader(args.orig_route_seg, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  new_lr = LogReader(args.regen_rlog, sort_by_time=True, only_union_types=True)

  orig_frames = load_frames(orig_lr, args.t_lo, args.t_hi)
  new_frames = load_frames(new_lr, regen_lo, regen_hi)

  print(f"Window {args.t_lo:.1f}s - {args.t_hi:.1f}s")
  print(f"Original plans: {len(orig_frames)}  Regen plans: {len(new_frames)}")

  print_table("ORIGINAL ROUTE (qlog)", orig_frames, args.sample_dt)
  print_transitions("original", orig_frames)
  print_table("REGEN (plannerd+radard)", new_frames, args.sample_dt)
  print_transitions("regen", new_frames)
  align_compare(orig_frames, new_frames)

  # Summary stats
  for label, frames in [("orig", orig_frames), ("regen", new_frames)]:
    if not frames:
      continue
    n35 = sum(1 for fr in frames if fr["a"] <= ACCEL_EMERGENCY)
    n_emerg = sum(1 for fr in frames if fr["emerg_any"])
    n_city = sum(1 for fr in frames if fr["city_close"])
    n_lead = sum(1 for fr in frames if fr["lead"])
    print(f"\n{label}: lead_frames={n_lead}/{len(frames)} city_close={n_city} "
          f"emerg_bypass={n_emerg} a<=-3.4={n35}")


if __name__ == "__main__":
  main()
