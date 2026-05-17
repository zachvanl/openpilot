#!/usr/bin/env python3
"""Compare qlog vs regen for city follow: late-brake assist vs stopping-lead gentle follow."""
import argparse
import sys
from bisect import bisect_left

import numpy as np
from openpilot.tools.lib.logreader import LogReader, ReadMode

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CITY_BRAKE_ASSIST_MAX_A_EGO,
  CITY_LEAD_STOPPED_SPEED,
  needs_city_late_brake_assist,
  city_lead_stopping,
)


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


def load_aligned(orig_path: str, regen_path: str, t_lo: float, t_hi: float):
  car = []
  radar = []
  plans_old = []
  plans_new = []
  t_origin = None

  for msg in LogReader(orig_path, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True):
    t = msg.logMonoTime / 1e9
    if t_origin is None:
      t_origin = t
    t = t - t_origin
    w = msg.which()
    if w == "carState":
      car.append((t, f(msg.carState, "vEgo"), f(msg.carState, "aEgo")))
    elif w == "radarState":
      l0 = capget(msg.radarState, "leadOne")
      if l0:
        radar.append((t, bool(capget(l0, "status")), f(l0, "dRel"), f(l0, "vLead")))
    elif w == "longitudinalPlan" and t_lo <= t <= t_hi:
      lp = msg.longitudinalPlan
      plans_old.append((t, f(lp, "aTarget"), bool(capget(lp, "shouldStop", False))))

  t_origin_new = None
  for msg in LogReader(regen_path, sort_by_time=True, only_union_types=True):
    t = msg.logMonoTime / 1e9
    if t_origin_new is None:
      t_origin_new = t
    t = t - t_origin_new
    if msg.which() == "longitudinalPlan" and t_lo <= t <= t_hi:
      lp = msg.longitudinalPlan
      plans_new.append((t, f(lp, "aTarget"), bool(capget(lp, "shouldStop", False))))

  def nearest(series, t, max_dt=0.12):
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

  rows = []
  for t, a_old, stop_old in plans_old:
    c = nearest(car, t)
    r = nearest(radar, t)
    n = nearest(plans_new, t, max_dt=0.15)
    if c is None or n is None:
      continue
    v_ego, a_ego = c[1], c[2]
    if r and r[1]:
      status, d_rel, v_lead = r[1], r[2], max(r[3], 0.0)
    else:
      status, d_rel, v_lead = False, float("nan"), float("nan")

    class Lead:
      status = False
      dRel = 0.0
      vLead = 0.0

    lead = Lead()
    lead.status = status
    lead.dRel = d_rel if status else 999.0
    lead.vLead = v_lead if status else 0.0

    assist = needs_city_late_brake_assist(lead, v_ego, a_ego, a_old) if status else False
    stopping = city_lead_stopping(v_lead) if status else False
    gentle = a_ego <= CITY_BRAKE_ASSIST_MAX_A_EGO

    rows.append({
      "t": t,
      "a_old": a_old,
      "a_new": n[1],
      "stop_old": stop_old,
      "stop_new": n[2],
      "d": d_rel if status else float("nan"),
      "vl": v_lead if status else float("nan"),
      "a_ego": a_ego,
      "stopping": stopping,
      "gentle": gentle,
      "assist": assist,
    })
  return rows


def summarize(label: str, rows: list):
  if not rows:
    print(f"\n{label}: no aligned samples")
    return

  da = np.array([r["a_new"] - r["a_old"] for r in rows])
  stopping = [r for r in rows if r["stopping"]]
  gentle_stopping = [r for r in rows if r["stopping"] and r["gentle"]]
  far_stop_old = sum(1 for r in rows if r["stop_old"] and r["d"] > 15)
  far_stop_new = sum(1 for r in rows if r["stop_new"] and r["d"] > 15)
  harsh_new = sum(1 for r in gentle_stopping if r["a_new"] <= -2.5)
  harsh_old = sum(1 for r in gentle_stopping if r["a_old"] <= -2.5)

  print(f"\n{'='*72}")
  print(label)
  print(f"  samples: {len(rows)}  delta aTarget mean={da.mean():+.3f} min={da.min():+.3f} max={da.max():+.3f}")
  print(f"  shouldStop with d>15m: old={far_stop_old} new={far_stop_new}")
  if gentle_stopping:
    print(f"  stopping lead + gentle aEgo: n={len(gentle_stopping)}")
    print(f"    a<=-2.5: old={harsh_old} new={harsh_new}")
    print(f"    mean aTarget: old={np.mean([r['a_old'] for r in gentle_stopping]):+.2f} "
          f"new={np.mean([r['a_new'] for r in gentle_stopping]):+.2f}")
  em_old = sum(1 for r in rows if r["a_old"] <= -3.4)
  em_new = sum(1 for r in rows if r["a_new"] <= -3.4)
  print(f"  emergency a<=-3.4: old={em_old} new={em_new}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("orig_qlog")
  p.add_argument("regen_rlog")
  p.add_argument("t_lo", type=float)
  p.add_argument("t_hi", type=float)
  args = p.parse_args()
  rows = load_aligned(args.orig_qlog, args.regen_rlog, args.t_lo, args.t_hi)
  summarize(f"{args.orig_qlog} [{args.t_lo}-{args.t_hi}s]", rows)
  return 0 if rows else 1


if __name__ == "__main__":
  sys.exit(main())
