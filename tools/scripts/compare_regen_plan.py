#!/usr/bin/env python3
"""Compare original route longitudinalPlan vs process-replay regen output."""
import sys
from collections import defaultdict

import numpy as np
from openpilot.tools.lib.logreader import LogReader, ReadMode


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


def load_plan_series(lr_iter):
  series = []
  radar = {}
  for msg in lr_iter:
    t = msg.logMonoTime / 1e9
    if msg.which() == "radarState":
      l0 = capget(msg.radarState, "leadOne")
      if l0 and capget(l0, "status"):
        radar[t] = (f(l0, "dRel"), f(l0, "vLead"))
    elif msg.which() == "longitudinalPlan":
      lp = msg.longitudinalPlan
      d, vl = radar.get(t, (float("nan"), float("nan")))
      series.append({
        "t": t,
        "a_old": f(lp, "aTarget"),
        "src": str(capget(lp, "longitudinalPlanSource", "")),
        "d": d,
        "vl": vl,
        "stop": bool(capget(lp, "shouldStop", False)),
      })
  return series


def compare(orig_route_seg: str, regen_rlog: str, t_lo: float = 0, t_hi: float = 99999):
  orig = load_plan_series(LogReader(orig_route_seg, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True))
  new = load_plan_series(LogReader(regen_rlog, sort_by_time=True, only_union_types=True))

  # align by nearest time (regen may shift mono time)
  diffs = []
  emergency_old = emergency_new = 0
  for o in orig:
    if not (t_lo <= o["t"] <= t_hi):
      continue
    if o["a_old"] <= -3.4:
      emergency_old += 1
    # find closest new sample within 0.1s
    best = min(new, key=lambda n: abs(n["t"] - o["t"]), default=None)
    if best is None or abs(best["t"] - o["t"]) > 0.15:
      continue
    da = best["a_old"] - o["a_old"]
    if best["a_old"] <= -3.4:
      emergency_new += 1
    diffs.append((o["t"], o["a_old"], best["a_old"], da, o["d"], o["src"]))

  print(f"\n{'='*80}")
  print(f"COMPARE {orig_route_seg}")
  print(f"  regen log: {regen_rlog}")
  print(f"  window: {t_lo:.1f}s - {t_hi:.1f}s  samples: {len(diffs)}")
  print(f"  emergency clips (a<=-3.4): old={emergency_old} new={emergency_new}")

  if not diffs:
    print("  No aligned samples.")
    return

  da = np.array([d[3] for d in diffs])
  print(f"  delta aTarget (new-old): mean={da.mean():+.3f} min={da.min():+.3f} max={da.max():+.3f}")

  # show largest improvements (more negative = stronger brake earlier)
  print("\n  Top 12 earlier/softer changes (new - old, most negative first):")
  for row in sorted(diffs, key=lambda x: x[3])[:12]:
    t, a0, a1, d, dist, src = row
    dist_s = f"{dist:.0f}m" if not np.isnan(dist) else "no lead"
    print(f"    t={t:7.1f} old={a0:+.2f} new={a1:+.2f} d={d:+.2f} lead={dist_s} src={src}")

  print("\n  Samples still at emergency -3.5 (new):")
  for row in [d for d in diffs if d[2] <= -3.4]:
    t, a0, a1, d, dist, src = row
    print(f"    t={t:7.1f} old={a0:+.2f} new={a1:+.2f} d={d:+.2f}")


if __name__ == "__main__":
  orig = sys.argv[1]  # e.g. 987638facc544f63/00000080--0684ca20c8/7/q
  regen = sys.argv[2]
  t_lo = float(sys.argv[3]) if len(sys.argv) > 3 else 430
  t_hi = float(sys.argv[4]) if len(sys.argv) > 4 else 455
  compare(orig, regen, t_lo, t_hi)
