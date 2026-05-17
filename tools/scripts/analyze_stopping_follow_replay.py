#!/usr/bin/env python3
"""Find stopping-lead follow windows and compare qlog vs regen aTarget."""
import argparse
import sys
from bisect import bisect_left

import numpy as np
from openpilot.tools.lib.logreader import LogReader, ReadMode

from openpilot.selfdrive.controls.lib.longitudinal_planner import CITY_LEAD_STOPPED_SPEED


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


def load_series(path: str, qlog: bool):
  car, radar, plans = [], [], []
  t0 = None
  kwargs = {"sort_by_time": True}
  if qlog:
    kwargs["default_mode"] = ReadMode.QLOG
    kwargs["only_union_types"] = True
  for msg in LogReader(path, **kwargs):
    t = msg.logMonoTime / 1e9
    if t0 is None:
      t0 = t
    t -= t0
    w = msg.which()
    if w == "carState":
      car.append((t, f(msg.carState, "vEgo"), f(msg.carState, "aEgo")))
    elif w == "radarState":
      l0 = capget(msg.radarState, "leadOne")
      if l0:
        radar.append((t, bool(capget(l0, "status")), f(l0, "dRel"), f(l0, "vLead")))
    elif w == "longitudinalPlan":
      lp = msg.longitudinalPlan
      plans.append((t, f(lp, "aTarget"), bool(capget(lp, "shouldStop", False))))

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
  for t, a, stop in plans:
    c = nearest(car, t)
    r = nearest(radar, t)
    if c is None:
      continue
    v_ego, a_ego = c[1], c[2]
    if r and r[1]:
      d_rel, v_lead = r[2], max(r[3], 0.0)
      lead = True
    else:
      d_rel, v_lead, lead = float("nan"), float("nan"), False
    rows.append({
      "t": t, "a": a, "stop": stop, "v_ego": v_ego, "a_ego": a_ego,
      "lead": lead, "d": d_rel, "vl": v_lead,
      "stopping": lead and v_lead < CITY_LEAD_STOPPED_SPEED,
    })
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument("qlog")
  p.add_argument("regen_rlog")
  p.add_argument("--min-d", type=float, default=8.0)
  p.add_argument("--max-d", type=float, default=45.0)
  args = p.parse_args()

  old = load_series(args.qlog, True)
  new = load_series(args.regen_rlog, False)

  # align new to old by time
  diffs = []
  for o in old:
    if not o["stopping"] or not (args.min_d <= o["d"] <= args.max_d):
      continue
    if o["a_ego"] > -0.8:
      continue
    best = min(new, key=lambda n: abs(n["t"] - o["t"]), default=None)
    if best is None or abs(best["t"] - o["t"]) > 0.15:
      continue
    da = best["a"] - o["a"]
    diffs.append({**o, "a_new": best["a"], "da": da, "stop_new": best["stop"]})

  print(f"stopping-lead gentle-follow windows (d={args.min_d}-{args.max_d}m, aEgo<=-0.8): {len(diffs)}")
  if not diffs:
    return 1

  da = np.array([d["da"] for d in diffs])
  harsh_old = sum(1 for d in diffs if d["a"] <= -2.5)
  harsh_new = sum(1 for d in diffs if d["a_new"] <= -2.5)
  far_stop_old = sum(1 for d in diffs if d["stop"] and d["d"] > 15)
  far_stop_new = sum(1 for d in diffs if d["stop_new"] and d["d"] > 15)

  print(f"  delta aTarget: mean={da.mean():+.3f} min={da.min():+.3f} max={da.max():+.3f}")
  print(f"  a<=-2.5: old={harsh_old} new={harsh_new}")
  print(f"  shouldStop d>15m: old={far_stop_old} new={far_stop_new}")

  print("\n  Largest softening (new - old):")
  for d in sorted(diffs, key=lambda x: x["da"], reverse=True)[:12]:
    print(f"    t={d['t']:6.1f} d={d['d']:5.1f} vl={d['vl']:.1f} aEgo={d['a_ego']:+.2f} "
          f"old={d['a']:+.2f} new={d['a_new']:+.2f} d={d['da']:+.2f} stop {d['stop']}->{d['stop_new']}")

  ok = da.mean() > 0.05 and harsh_new <= harsh_old
  print(f"\n  PASS softening: {ok}")
  return 0 if ok else 2


if __name__ == "__main__":
  sys.exit(main())
