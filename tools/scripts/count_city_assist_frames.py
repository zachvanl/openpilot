#!/usr/bin/env python3
"""Count how often city assist would fire on a qlog (old vs new gate)."""
import sys
from bisect import bisect_left

import numpy as np
from openpilot.tools.lib.logreader import LogReader, ReadMode

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  city_closing_brake_active,
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


class Lead:
  status = False
  dRel = 0.0
  vLead = 0.0


def main():
  path = sys.argv[1]
  t_lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0
  t_hi = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
  car, radar, plans = [], [], []
  t0 = None
  for msg in LogReader(path, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True):
    t = msg.logMonoTime / 1e9
    if t0 is None:
      t0 = t
    t -= t0
    if msg.which() == "carState":
      car.append((t, f(msg.carState, "vEgo"), f(msg.carState, "aEgo")))
    elif msg.which() == "radarState":
      l0 = capget(msg.radarState, "leadOne")
      if l0:
        radar.append((t, bool(capget(l0, "status")), f(l0, "dRel"), f(l0, "vLead")))
    elif msg.which() == "longitudinalPlan":
      plans.append((t, f(msg.longitudinalPlan, "aTarget")))

  def nearest(series, t):
    if not series:
      return None
    ts = [s[0] for s in series]
    i = bisect_left(ts, t)
    best = None
    for j in (i - 1, i):
      if 0 <= j < len(series) and (best is None or abs(ts[j] - t) < abs(best[0] - t)):
        best = series[j]
    return best if best and abs(best[0] - t) <= 0.12 else None

  old_close = new_assist = stopping = 0
  for t, a_plan in plans:
    if not (t_lo <= t <= t_hi):
      continue
    c = nearest(car, t)
    r = nearest(radar, t)
    if c is None or r is None or not r[1]:
      continue
    v_ego, a_ego = c[1], c[2]
    lead = Lead()
    lead.status = True
    lead.dRel = r[2]
    lead.vLead = max(r[3], 0.0)
    if city_lead_stopping(lead.vLead):
      stopping += 1
    if city_closing_brake_active(lead, v_ego):
      old_close += 1
    if needs_city_late_brake_assist(lead, v_ego, a_ego, a_plan):
      new_assist += 1

  print(f"{path} [{t_lo}-{t_hi}s]")
  print(f"  stopping lead frames: {stopping}")
  print(f"  city_closing_brake_active (old path): {old_close}")
  print(f"  needs_city_late_brake_assist (new path): {new_assist}")


if __name__ == "__main__":
  main()
