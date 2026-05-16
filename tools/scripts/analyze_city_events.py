#!/usr/bin/env python3
"""
Find city braking / FCW / red-light issues in route qlogs.

Detects:
  - FCW (longitudinalPlan.fcw)
  - Late braking while engaged (low TTC, weak aTarget)
  - Emergency clip (-3.5 aTarget) events
  - No-lead cruise at city speed (possible red-light / vision gap)
  - model shouldStop vs plan shouldStop at low speed
"""
import json
import math
import sys
from collections import Counter, defaultdict

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


MS_TO_MPH = 2.23693629
CITY_SPEED_MAX = 50.0 / MS_TO_MPH  # ~22.35 m/s


def analyze_route(route: str, seg_end: int = 40) -> dict:
  events = {
    "fcw": [],
    "emergency_decel": [],
    "late_brake_engaged": [],
    "no_lead_city_cruise": [],
    "stoplight_miss": [],
  }
  stats = Counter()

  latest = {}
  engaged_since = 0

  for seg in range(seg_end):
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
        latest["v_ego"] = f(cs, "vEgo")
        latest["a_ego"] = f(cs, "aEgo")
        latest["brake"] = bool(capget(cs, "brakePressed", False))

      elif typ == "selfdriveState":
        sds = msg.selfdriveState
        latest["op_active"] = bool(capget(sds, "active", False) or capget(sds, "enabled", False))
        latest["experimental"] = bool(capget(sds, "experimentalMode", False))

      elif typ == "radarState":
        rs = msg.radarState
        l0 = capget(rs, "leadOne")
        if l0 and capget(l0, "status", False):
          latest["lead"] = True
          latest["d_rel"] = f(l0, "dRel")
          latest["v_lead"] = f(l0, "vLead")
          latest["model_prob"] = f(capget(l0, "modelProb", 0.0), 0.0)
        else:
          latest["lead"] = False

      elif typ == "modelV2":
        md = msg.modelV2
        action = capget(md, "action")
        latest["e2e_a"] = f(action, "desiredAcceleration") if action else float("nan")
        latest["model_should_stop"] = bool(capget(action, "shouldStop", False)) if action else False

      elif typ == "longitudinalPlanSP":
        lpsp = msg.longitudinalPlanSP
        dec = capget(lpsp, "dec")
        latest["dec_state"] = str(capget(dec, "state", "")) if dec else ""
        latest["sp_source"] = str(capget(lpsp, "longitudinalPlanSource", ""))

      elif typ == "longitudinalPlan":
        lp = msg.longitudinalPlan
        a_target = f(lp, "aTarget")
        source = str(capget(lp, "longitudinalPlanSource", ""))
        should_stop = bool(capget(lp, "shouldStop", False))
        fcw = bool(capget(lp, "fcw", False))

        op_active = latest.get("op_active", False)
        if op_active:
          engaged_since += 1
        else:
          engaged_since = 0

        v_ego = latest.get("v_ego", 0.0)
        if math.isnan(v_ego):
          continue

        row = {
          "t": t,
          "seg": seg,
          "v_ego_mph": v_ego * MS_TO_MPH,
          "a_target": a_target,
          "a_ego": latest.get("a_ego", float("nan")),
          "source": source,
          "should_stop_plan": should_stop,
          "model_should_stop": latest.get("model_should_stop", False),
          "fcw": fcw,
          "op_active": op_active,
          "brake": latest.get("brake", False),
          "lead": latest.get("lead", False),
          "d_rel": latest.get("d_rel", float("nan")),
          "v_lead_mph": latest.get("v_lead", float("nan")) * MS_TO_MPH if latest.get("lead") else float("nan"),
          "e2e_a": latest.get("e2e_a", float("nan")),
          "dec_state": latest.get("dec_state", ""),
          "sp_source": latest.get("sp_source", ""),
          "experimental": latest.get("experimental", False),
        }

        stats["samples"] += 1
        if v_ego < CITY_SPEED_MAX:
          stats["city_samples"] += 1

        if fcw:
          events["fcw"].append(row)
          stats["fcw"] += 1

        if a_target <= -3.4:
          events["emergency_decel"].append(row)
          stats["emergency_decel"] += 1

        if latest.get("lead") and op_active and engaged_since > 20:
          d = latest["d_rel"]
          v_lead = latest["v_lead"]
          closing = max(v_ego - v_lead, 0.0)
          ttc = d / max(closing, 0.5)
          if d < 70 and v_lead < 3.0 and v_ego > 5.0 and ttc < 4.0 and a_target > -1.0:
            row["ttc"] = ttc
            events["late_brake_engaged"].append(row)
            stats["late_brake_engaged"] += 1

        if (op_active and engaged_since > 20 and v_ego < CITY_SPEED_MAX and v_ego > 8.0 and
            not latest.get("lead", False) and a_target > -0.4 and not should_stop):
          events["no_lead_city_cruise"].append(row)
          stats["no_lead_city_cruise"] += 1

        if (v_ego < 12.0 and v_ego > 2.0 and op_active and
            not should_stop and a_target > -0.8 and
            not latest.get("model_should_stop", False) and
            latest.get("experimental", False)):
          events["stoplight_miss"].append(row)
          stats["stoplight_miss"] += 1

  return {"route": route, "stats": dict(stats), "events": events}


def merge_windows(events: list[dict], gap_s: float = 2.0) -> list[dict]:
  if not events:
    return []
  events = sorted(events, key=lambda r: (r["seg"], r["t"]))
  windows = []
  cur = {"start": events[0], "end": events[0], "count": 1}
  for ev in events[1:]:
    if ev["seg"] == cur["end"]["seg"] and ev["t"] - cur["end"]["t"] <= gap_s:
      cur["end"] = ev
      cur["count"] += 1
    else:
      windows.append(cur)
      cur = {"start": ev, "end": ev, "count": 1}
  windows.append(cur)
  return windows


def print_report(result: dict) -> None:
  route = result["route"]
  stats = result["stats"]
  print("=" * 90)
  print(f"CITY EVENT ANALYSIS: {route}")
  print(f"  samples={stats.get('samples', 0)} city={stats.get('city_samples', 0)}")
  print("=" * 90)

  for key in ("fcw", "emergency_decel", "late_brake_engaged", "no_lead_city_cruise", "stoplight_miss"):
    evs = result["events"][key]
    windows = merge_windows(evs)
    print(f"\n--- {key}: {len(evs)} samples, {len(windows)} windows ---")
    for i, w in enumerate(windows[:12]):
      s, e = w["start"], w["end"]
      lead_s = f"{s['d_rel']:.0f}m" if s.get("lead") and not math.isnan(s.get("d_rel", float("nan"))) else "no lead"
      print(f"  [{i+1}] seg {s['seg']} t={s['t']:.1f}-{e['t']:.1f}s "
            f"v={s['v_ego_mph']:.0f}->{e['v_ego_mph']:.0f}mph "
            f"aT={s['a_target']:+.2f}->{e['a_target']:+.2f} "
            f"src={s['source']} active={s['op_active']} "
            f"lead={lead_s} planStop={s['should_stop_plan']} modelStop={s['model_should_stop']} "
            f"fcw={s['fcw']}")


def main() -> None:
  routes = sys.argv[1:] or [
    "987638facc544f63/00000080--0684ca20c8",
    "987638facc544f63/0000007a--e54be11ada",
    "987638facc544f63/00000071--f2a9720142",
    "987638facc544f63/0000006e--860c70648a",
  ]
  out_path = "tools/scripts/route_city_events.json"
  all_results = []
  for route in routes:
    result = analyze_route(route)
    print_report(result)
    all_results.append(result)

  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=float)
  print(f"\nWrote {out_path}")


if __name__ == "__main__":
  main()
