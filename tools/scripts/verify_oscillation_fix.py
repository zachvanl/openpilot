#!/usr/bin/env python3
"""Verify the cruise speed matching fix against real oscillation data from route 25/5."""
import numpy as np

FREEWAY_CRUISE_MATCH_DIST_BP = [60.0, 120.0]
FREEWAY_CRUISE_MATCH_BUFFER_V = [2.0, 8.0]
FREEWAY_CRUISE_MATCH_MIN_VLEAD = 10.0
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
T_FOLLOW_BASE = 1.45
FREEWAY_FOLLOW_BONUS = 0.30

def safe_dist(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def stopped_eq(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

# Real data from route 987638facc544f63/00000025--1c4fd18d6f/5
data = [
  # (t, v_ego, v_lead, d_rel, source_old)
  (301.0, 27.5, 26.7, 54.5, "cruise"),
  (304.5, 28.4, 28.1, 59.8, "cruise"),
  (307.5, 28.9, 28.3, 67.0, "cruise"),
  (311.0, 29.3, 28.8, 77.2, "cruise"),
  (314.5, 29.7, 28.6, 80.8, "cruise"),
  (318.0, 29.9, 29.0, 67.6, "cruise"),
  (321.5, 29.4, 27.9, 54.1, "lead0"),
  (325.0, 28.5, 27.2, 55.0, "cruise"),
  (328.5, 28.7, 27.9, 64.4, "cruise"),
  (332.0, 29.0, 27.1, 62.4, "cruise"),
  (335.5, 27.4, 23.4, 56.1, "lead1"),
  (339.0, 26.3, 24.6, 21.3, "lead1"),
  (342.0, 25.6, 24.5, 11.7, "lead0"),
]

v_cruise_raw = 35.0  # approximate user cruise speed (~78 mph)
t_follow = T_FOLLOW_BASE + FREEWAY_FOLLOW_BONUS

print("Freeway cruise speed matching simulation")
print("=" * 100)
print(f"{'t':>6} {'vEgo':>6} {'vLead':>6} {'dRel':>6} {'vCr_old':>8} {'vCr_new':>8} {'src_old':>8} {'cap_buf':>8} {'lead_obs':>9} {'cruise_obs_old':>14} {'cruise_obs_new':>14}")
print("-" * 100)

for t, v_ego, v_lead, d_rel, src_old in data:
  lead_obs = d_rel + stopped_eq(v_lead)
  cruise_obs_old = safe_dist(v_ego, t_follow)  # without speed match
  
  # Apply cruise speed matching
  v_cruise_new = v_cruise_raw
  if v_lead > FREEWAY_CRUISE_MATCH_MIN_VLEAD and d_rel < FREEWAY_CRUISE_MATCH_DIST_BP[1]:
    cap_buffer = float(np.interp(d_rel, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
    v_cruise_new = min(v_cruise_raw, v_lead + cap_buffer)
  else:
    cap_buffer = float("nan")
  
  cruise_obs_new = safe_dist(v_ego, t_follow)  # t=0 obstacle uses v_ego not v_cruise
  
  # Which source would be selected? (simplified: just check t=0)
  src_new = "lead" if lead_obs < cruise_obs_new else "cruise"
  
  print(f"{t:6.1f} {v_ego:6.1f} {v_lead:6.1f} {d_rel:6.1f} {v_cruise_raw:8.1f} {v_cruise_new:8.1f} "
        f"{src_old:>8} {cap_buffer:8.1f} {lead_obs:9.1f} {cruise_obs_old:14.1f} {cruise_obs_new:14.1f}")

print()
print("Key insight: the v_cruise cap reduces cruise_obstacle at future timesteps (not t=0),")
print("because v_cruise_clipped at later timesteps is capped. This keeps the lead obstacle")
print("as the binding constraint for more of the horizon, preventing aggressive acceleration.")
print()

# Show the effect on acceleration intent
print("Effect on MPC acceleration target:")
print("-" * 60)
for t, v_ego, v_lead, d_rel, src_old in data:
  v_cruise_capped = v_cruise_raw
  if v_lead > FREEWAY_CRUISE_MATCH_MIN_VLEAD and d_rel < FREEWAY_CRUISE_MATCH_DIST_BP[1]:
    cap_buffer = float(np.interp(d_rel, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
    v_cruise_capped = min(v_cruise_raw, v_lead + cap_buffer)
  
  gap_old = v_cruise_raw - v_ego
  gap_new = v_cruise_capped - v_ego
  
  print(f"  t={t:.0f}s: dRel={d_rel:.0f}m  "
        f"OLD gap-to-cruise={gap_old:+.1f}m/s  "
        f"NEW gap-to-cruise={gap_new:+.1f}m/s  "
        f"(reduced by {gap_old - gap_new:.1f}m/s)")
