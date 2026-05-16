#!/usr/bin/env python3
"""
Extract regression scenarios from route qlogs into route_scenarios.json.

Usage:
  .venv/Scripts/python.exe tools/scripts/extract_route_scenarios.py [route ...]

Requires comma API access (same as LogReader / connect).
"""
import json
import math
import sys
from pathlib import Path

from openpilot.tools.lib.logreader import LogReader, ReadMode

# Reuse detectors from analyze_city_events
from analyze_city_events import analyze_route, merge_windows, MS_TO_MPH  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "tools" / "scripts" / "route_scenarios.json"

DEFAULT_ROUTES = [
  {"id": "987638facc544f63/00000080--0684ca20c8", "label": "city_sccv_primary", "scc_v": True},
  {"id": "987638facc544f63/0000007a--e54be11ada", "label": "city_sccv_secondary", "scc_v": True},
  {"id": "987638facc544f63/00000071--f2a9720142", "label": "city_pre_sccv", "scc_v": False},
  {"id": "987638facc544f63/0000006e--860c70648a", "label": "city_pre_sccv_2", "scc_v": False},
]

WINDOW_SPECS = [
  ("emergency_decel", "late_emergency_brake", 15.0, 3.0),
  ("stoplight_miss", "red_light_no_stop", 12.0, 2.0),
  ("no_lead_city_cruise", "no_lead_before_brake", 10.0, 2.0),
  ("fcw", "fcw_alert", 8.0, 2.0),
]


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


def extract_timeline(route: str, seg: int, t_center: float, before_s: float, after_s: float) -> list[dict]:
  seg_id = f"{route}/{seg}"
  lr = LogReader(seg_id, default_mode=ReadMode.QLOG, sort_by_time=True, only_union_types=True)
  t0 = t_center - before_s
  t1 = t_center + after_s
  latest = {}
  rows = []

  for msg in lr:
    typ = msg.which()
    t = msg.logMonoTime / 1e9
    if t < t0:
      continue
    if t > t1:
      break

    if typ == "carState":
      cs = msg.carState
      latest["v_ego"] = f(cs, "vEgo")
      latest["brake"] = bool(capget(cs, "brakePressed", False))

    elif typ == "selfdriveState":
      sds = msg.selfdriveState
      latest["op_active"] = bool(capget(sds, "active", False) or capget(sds, "enabled", False))

    elif typ == "radarState":
      rs = msg.radarState
      l0 = capget(rs, "leadOne")
      if l0 and capget(l0, "status", False):
        latest["lead"] = True
        latest["d_rel"] = f(l0, "dRel")
        latest["v_lead"] = f(l0, "vLead")
      else:
        latest["lead"] = False

    elif typ == "modelV2":
      md = msg.modelV2
      action = capget(md, "action")
      latest["e2e_a"] = f(action, "desiredAcceleration") if action else float("nan")
      latest["model_should_stop"] = bool(capget(action, "shouldStop", False)) if action else False
      latest["hard_brake"] = bool(capget(capget(md, "meta"), "hardBrakePredicted", False))

    elif typ == "longitudinalPlan":
      lp = msg.longitudinalPlan
      rows.append({
        "t_offset": round(t - t_center, 2),
        "v_ego_mph": round(latest.get("v_ego", 0) * MS_TO_MPH, 1),
        "a_target": round(f(lp, "aTarget"), 2),
        "source": str(capget(lp, "longitudinalPlanSource", "")),
        "should_stop": bool(capget(lp, "shouldStop", False)),
        "fcw": bool(capget(lp, "fcw", False)),
        "op_active": latest.get("op_active", False),
        "brake": latest.get("brake", False),
        "lead": latest.get("lead", False),
        "d_rel": round(latest["d_rel"], 1) if latest.get("lead") else None,
        "v_lead_mph": round(latest["v_lead"] * MS_TO_MPH, 1) if latest.get("lead") else None,
        "model_should_stop": latest.get("model_should_stop", False),
        "e2e_a": round(latest.get("e2e_a", float("nan")), 2) if not math.isnan(latest.get("e2e_a", float("nan"))) else None,
        "hard_brake": latest.get("hard_brake", False),
      })

  return rows


def build_scenarios(route_meta: dict, analysis: dict) -> list[dict]:
  scenarios = []
  route = route_meta["id"]
  for event_key, scenario_type, before_s, after_s in WINDOW_SPECS:
    windows = merge_windows(analysis["events"][event_key])
    for idx, w in enumerate(windows[:5]):
      s = w["start"]
      center_t = (s["t"] + w["end"]["t"]) / 2.0
      timeline = extract_timeline(route, s["seg"], center_t, before_s, after_s)
      if len(timeline) < 3:
        continue
      scenarios.append({
        "name": f"{route_meta['label']}_{scenario_type}_{s['seg']}_{int(s['t'])}",
        "route": route,
        "segment": s["seg"],
        "t_center": round(center_t, 1),
        "type": scenario_type,
        "event_key": event_key,
        "scc_v": route_meta.get("scc_v"),
        "summary": {
          "v_ego_mph_start": s["v_ego_mph"],
          "v_ego_mph_end": w["end"]["v_ego_mph"],
          "a_target_start": s["a_target"],
          "a_target_end": w["end"]["a_target"],
          "source": s["source"],
          "op_active": s["op_active"],
          "model_should_stop": s.get("model_should_stop"),
          "plan_should_stop": s.get("should_stop_plan"),
        },
        "timeline": timeline,
      })
  return scenarios


def main() -> None:
  route_metas = DEFAULT_ROUTES
  if len(sys.argv) > 1:
    route_metas = [{"id": r, "label": r.split("/")[-1][:16], "scc_v": None} for r in sys.argv[1:]]

  registry = {"routes": route_metas, "scenarios": []}
  for meta in route_metas:
    print(f"Extracting {meta['id']}...")
    analysis = analyze_route(meta["id"])
    registry["scenarios"].extend(build_scenarios(meta, analysis))

  OUT_PATH.write_text(json.dumps(registry, indent=2))
  print(f"Wrote {len(registry['scenarios'])} scenarios to {OUT_PATH}")


if __name__ == "__main__":
  main()
