#!/usr/bin/env python3
"""Scan all available segments of a route for a high-level summary."""
import sys
import math
import numpy as np
from collections import Counter
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
print(f"Scanning route: {route}")
print(f"{'seg':>4} {'dur':>5} {'v_min':>6} {'v_max':>7} {'v_med':>7} {'lead%':>6} {'src_counts':>40} {'a_min':>6} {'a_max':>6} {'a_flips':>7} {'curv_max':>8}")
print("-" * 120)

for seg in range(30):
  seg_id = f"{route}/{seg}"
  try:
    lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  except Exception:
    continue

  v_egos = []
  a_targets = []
  sources = []
  leads = 0
  total = 0
  max_curv = 0.0
  start_t = None

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9
    if start_t is None:
      start_t = t

    if typ == "carState":
      cs = msg.carState
      v = f(cs, "vEgo")
      if not math.isnan(v):
        v_egos.append(v)

    elif typ == "longitudinalPlan":
      lp = msg.longitudinalPlan
      a = f(lp, "aTarget")
      src = str(capget(lp, "longitudinalPlanSource", ""))
      has_lead = bool(capget(lp, "hasLead", False))
      if not math.isnan(a):
        a_targets.append(a)
        sources.append(src)
      if has_lead:
        leads += 1
      total += 1

    elif typ == "controlsState":
      dc = f(msg.controlsState, "desiredCurvature")
      if not math.isnan(dc):
        max_curv = max(max_curv, abs(dc))

  if not v_egos:
    continue

  dur = (t - start_t) if start_t else 0
  v_arr = np.array(v_egos)
  v_min = np.min(v_arr)
  v_max = np.max(v_arr)
  v_med = np.median(v_arr)
  lead_pct = (leads / total * 100) if total else 0

  a_flips = 0
  if len(a_targets) > 1:
    for i in range(1, len(a_targets)):
      if (a_targets[i-1] > 0.05 and a_targets[i] < -0.05) or \
         (a_targets[i-1] < -0.05 and a_targets[i] > 0.05):
        a_flips += 1

  src_counts = Counter(sources)
  src_str = " ".join(f"{k}={v}" for k, v in src_counts.most_common(4))

  a_min = min(a_targets) if a_targets else float("nan")
  a_max = max(a_targets) if a_targets else float("nan")

  mph_med = v_med / 0.44704
  mph_max = v_max / 0.44704

  print(f"{seg:4d} {dur:5.0f}s {v_min:5.1f}ms {v_max:6.1f}ms/{mph_max:4.0f}mph {v_med:5.1f}ms/{mph_med:4.0f}mph "
        f"{lead_pct:5.0f}% {src_str:>40} {a_min:+5.2f} {a_max:+5.2f} {a_flips:7d} {max_curv:8.5f}")
