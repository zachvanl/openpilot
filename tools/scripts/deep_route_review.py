#!/usr/bin/env python3
"""Deep cross-route review for city braking / e2e / lead / SCC issues."""
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np
from openpilot.tools.lib.logreader import LogReader, ReadMode

ROUTES = [
  ("00000080--0684ca20c8", "city_sccv", True),
  ("0000007a--e54be11ada", "city_sccv_2", True),
  ("00000071--f2a9720142", "pre_sccv", False),
  ("0000006e--860c70648a", "pre_sccv_2", False),
]
BASE = "987638facc544f63"
MS_TO_MPH = 2.23693629
CITY_V = 50.0 / MS_TO_MPH


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


def read_init(route_prefix):
  try:
    lr = LogReader(f"{BASE}/{route_prefix}/0", default_mode=ReadMode.QLOG, only_union_types=True)
  except Exception:
    return {}
  for msg in lr:
    if msg.which() == "initData":
      init = msg.initData
      keys = [
        "GentleLeadBraking", "GentleLeadBrakingFarLead", "GentleLeadBrakingLevel",
        "LeadHysteresis", "ExperimentalMode", "AlphaLongitudinalEnabled",
        "DynamicExperimentalControl", "SmartCruiseControlVision", "SmartCruiseControlMap",
        "LongitudinalTFollowAggressive", "LongitudinalTFollowStandard", "LongitudinalTFollowRelaxed",
      ]
      params = {}
      for e in init.params.entries:
        try:
          params[e.key] = e.value.decode("utf-8", errors="replace")
        except Exception:
          params[e.key] = str(e.value)
      out = {k: params.get(k, "?") for k in keys}
      def _s(x):
        if not x:
          return "?"
        return x.decode(errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)
      out["gitBranch"] = _s(init.gitBranch)
      out["version"] = _s(init.version)
      return out
  return {}


def analyze_route(route_prefix, label, scc_v_label):
  stats = Counter()
  gaps_no_lead = []  # (seg, t, v_mph, duration)
  emergency = []
  hard_brake = []
  plan_fcw = []
  pos_accel_close = []
  lead_drop_frames = 0
  lead_active_frames = 0
  has_lead_plan_mismatch = 0
  scc_active_frames = 0
  dec_states = Counter()
  sp_sources = Counter()
  sources_city = Counter()
  sources_hwy = Counter()
  disengages = []
  driver_brakes_while_engaged = []

  prev_lead = None
  no_lead_start = None
  prev_active = False
  engaged = False

  for seg in range(40):
    try:
      lr = LogReader(f"{BASE}/{route_prefix}/{seg}", default_mode=ReadMode.QLOG,
                     sort_by_time=True, only_union_types=True)
    except Exception:
      break

    latest = {}
    for msg in lr:
      typ = msg.which()
      t = msg.logMonoTime / 1e9
      stats["msgs"] += 1

      if typ == "initData":
        continue

      if typ == "carState":
        cs = msg.carState
        latest["v"] = f(cs, "vEgo")
        latest["brake"] = bool(capget(cs, "brakePressed"))
        latest["gas"] = bool(capget(cs, "gasPressed"))

      elif typ == "selfdriveState":
        sds = msg.selfdriveState
        active = bool(capget(sds, "active") or capget(sds, "enabled"))
        if prev_active and not active:
          disengages.append((seg, t, latest.get("v", 0) * MS_TO_MPH))
        prev_active = active
        engaged = active
        latest["exp"] = bool(capget(sds, "experimentalMode"))

      elif typ == "modelV2":
        md = msg.modelV2
        action = capget(md, "action")
        latest["e2e_a"] = f(action, "desiredAcceleration") if action else float("nan")
        latest["model_stop"] = bool(capget(action, "shouldStop", False)) if action else False
        meta = capget(md, "meta")
        if meta and bool(capget(meta, "hardBrakePredicted", False)):
          hard_brake.append((seg, t, latest.get("v", 0) * MS_TO_MPH, latest.get("e2e_a", float("nan"))))
        if len(md.leadsV3) > 0:
          latest["model_prob"] = float(md.leadsV3[0].prob)
          latest["model_x"] = float(md.leadsV3[0].x[0]) if len(md.leadsV3[0].x) else float("nan")
        else:
          latest["model_prob"] = float("nan")
          latest["model_x"] = float("nan")

      elif typ == "radarState":
        rs = msg.radarState
        l0 = capget(rs, "leadOne")
        lead_on = bool(l0 and capget(l0, "status"))
        if lead_on:
          lead_active_frames += 1
          latest["d"] = f(l0, "dRel")
          latest["vl"] = f(l0, "vLead")
          latest["mprob"] = float(capget(l0, "modelProb", 0) or 0)
          if prev_lead is False and no_lead_start is not None:
            gaps_no_lead.append((seg, no_lead_start, t - no_lead_start, latest.get("v", 0) * MS_TO_MPH))
            no_lead_start = None
          prev_lead = True
        else:
          lead_drop_frames += 1
          if prev_lead is True or prev_lead is None:
            if no_lead_start is None:
              no_lead_start = t
          prev_lead = False
          latest.pop("d", None)

      elif typ == "longitudinalPlanSP":
        lpsp = msg.longitudinalPlanSP
        sp = str(capget(lpsp, "longitudinalPlanSource", ""))
        sp_sources[sp] += 1
        scc = capget(lpsp, "smartCruiseControl")
        if scc:
          vis = capget(scc, "vision")
          if vis and str(capget(vis, "state", "")) not in ("disabled", ""):
            scc_active_frames += 1
        dec = capget(lpsp, "dec")
        if dec:
          dec_states[str(capget(dec, "state", ""))] += 1

      elif typ == "longitudinalPlan":
        lp = msg.longitudinalPlan
        a = f(lp, "aTarget")
        src = str(capget(lp, "longitudinalPlanSource", ""))
        has_lead = bool(capget(lp, "hasLead", False))
        fcw = bool(capget(lp, "fcw", False))
        ss = bool(capget(lp, "shouldStop", False))
        v = latest.get("v", 0)
        stats["plan"] += 1

        if v < CITY_V:
          sources_city[src] += 1
        else:
          sources_hwy[src] += 1

        if has_lead and not latest.get("d"):
          has_lead_plan_mismatch += 1
        if fcw and engaged:
          plan_fcw.append((seg, t, v * MS_TO_MPH, a, src))
        if a <= -3.4 and engaged:
          emergency.append((seg, t, v * MS_TO_MPH, a, src, latest.get("d"), latest.get("vl")))

        if engaged and v > 5 and latest.get("brake"):
          driver_brakes_while_engaged.append((seg, t, v * MS_TO_MPH, a, src))

        if engaged and latest.get("d") is not None:
          d = latest["d"]
          vl = latest.get("vl", 0)
          if v < CITY_V and v > 8 and vl < 5 and (v - vl) > 2 and a > 0.2:
            pos_accel_close.append((seg, t, v * MS_TO_MPH, a, src, d, vl * MS_TO_MPH, ss, latest.get("model_stop")))

  long_gaps = [g for g in gaps_no_lead if g[2] > 1.0 and g[3] > 25]
  return {
    "label": label,
    "scc_v_label": scc_v_label,
    "init": read_init(route_prefix),
    "stats": dict(stats),
    "lead_active_pct": 100 * lead_active_frames / max(lead_active_frames + lead_drop_frames, 1),
    "long_no_lead_gaps": sorted(long_gaps, key=lambda x: -x[2])[:12],
    "emergency": emergency[:15],
    "hard_brake": hard_brake[:12],
    "plan_fcw": plan_fcw[:12],
    "pos_accel_close": sorted(pos_accel_close, key=lambda x: -x[3])[:15],
    "has_lead_mismatch": has_lead_plan_mismatch,
    "scc_active_frames": scc_active_frames,
    "dec_states": dict(dec_states),
    "sp_sources_top": sp_sources.most_common(6),
    "sources_city": sources_city.most_common(5),
    "sources_hwy": sources_hwy.most_common(5),
    "disengages": disengages[:12],
    "driver_brakes_engaged": len(driver_brakes_while_engaged),
  }


def main():
  routes = ROUTES
  if len(sys.argv) > 1:
    routes = [(r, r, True) for r in sys.argv[1:]]

  reports = []
  for rp, label, scc in routes:
    print(f"\nAnalyzing {rp}...")
    reports.append(analyze_route(rp, label, scc))

  out_path = "tools/scripts/deep_route_review.json"
  with open(out_path, "w") as fp:
    json.dump(reports, fp, indent=2, default=str)

  for r in reports:
    print("\n" + "=" * 88)
    print(f"ROUTE {r['label']} ({BASE}/{r['label']})  SCC-V drive: {r['scc_v_label']}")
    print("=" * 88)
    init = r["init"]
    if init:
      print("Params:", {k: init[k] for k in init if k not in ("gitBranch", "version")})
      print(f"Build: {init.get('version','?')} branch {init.get('gitBranch','?')}")
    print(f"Lead active {r['lead_active_pct']:.1f}% of radar frames | hasLead w/o radar lead: {r['has_lead_mismatch']}")
    print(f"SCC-V active frames (vision state not disabled): {r['scc_active_frames']}")
    print(f"DEC states: {r['dec_states']} | SP sources: {r['sp_sources_top']}")
    print(f"City sources: {r['sources_city']} | Hwy sources: {r['sources_hwy']}")
    print(f"Disengagements: {len(r['disengages'])} | Driver brake while engaged samples: {r['driver_brakes_engaged']}")
    print(f"Emergency -3.5 clips: {len(r['emergency'])} | model hardBrake: {len(r['hard_brake'])} | plan fcw: {len(r['plan_fcw'])}")

    if r["long_no_lead_gaps"]:
      print("\nLong no-lead gaps (>1s, >25mph):")
      for seg, t0, dur, vm in r["long_no_lead_gaps"][:8]:
        print(f"  seg={seg} t~{t0:.0f}s duration={dur:.1f}s v~{vm:.0f}mph")

    if r["emergency"]:
      print("\nEmergency aTarget samples:")
      for row in r["emergency"][:6]:
        print(f"  seg={row[0]} t={row[1]:.1f} v={row[2]:.0f}mph a={row[3]:+.2f} src={row[4]} d={row[5]} vl={row[6]}")

    if r["hard_brake"]:
      print("\nmodelV2.meta.hardBrakePredicted:")
      for row in r["hard_brake"][:6]:
        print(f"  seg={row[0]} t={row[1]:.1f} v={row[2]:.0f}mph e2e_a={row[3]:+.2f}")

    if r["plan_fcw"]:
      print("\nlongitudinalPlan.fcw:")
      for row in r["plan_fcw"][:6]:
        print(f"  seg={row[0]} t={row[1]:.1f} v={row[2]:.0f}mph a={row[3]:+.2f} src={row[4]}")

    if r["pos_accel_close"]:
      print("\nPositive aTarget closing on slow lead (city):")
      for row in r["pos_accel_close"][:8]:
        print(f"  seg={row[0]} t={row[1]:.1f} v={row[2]:.0f}mph a={row[3]:+.2f} src={row[4]} "
              f"d={row[5]:.0f}m vl={row[6]:.0f}mph stop={row[7]}/{row[8]}")

    if r["disengages"]:
      print("\nDisengagements:")
      for seg, t, v in r["disengages"][:6]:
        print(f"  seg={seg} t={t:.1f} v={v:.0f}mph")

  print(f"\nWrote {out_path}")


if __name__ == "__main__":
  main()
