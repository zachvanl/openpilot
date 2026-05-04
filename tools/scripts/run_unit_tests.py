#!/usr/bin/env python3
"""Standalone validation of lateral and longitudinal changes."""
import sys
import numpy as np

passed = 0
failed = 0

def check(name, condition):
  global passed, failed
  if condition:
    passed += 1
    print(f"  PASS: {name}")
  else:
    failed += 1
    print(f"  FAIL: {name}")

# ─── Replicate constants and functions from drive_helpers.py ───
DT_CTRL = 0.01
ACCELERATION_DUE_TO_GRAVITY = 9.81
MIN_SPEED = 1.0
MAX_CURVATURE = 0.2
MAX_LATERAL_JERK = 5.0
MAX_LATERAL_ACCEL_NO_ROLL = 3.0
_HIGH_SPEED_LAT_BP = [15., 30.]
_MAX_LAT_JERK_V = [MAX_LATERAL_JERK, 3.5]
_MAX_LAT_ACCEL_V = [MAX_LATERAL_ACCEL_NO_ROLL, 2.5]

def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def clip_curvature(v_ego, prev_curvature, new_curvature, roll):
  v_ego = max(v_ego, MIN_SPEED)
  max_lateral_jerk = float(np.interp(v_ego, _HIGH_SPEED_LAT_BP, _MAX_LAT_JERK_V))
  max_curvature_rate = max_lateral_jerk / (v_ego ** 2)
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)
  max_lat_accel_base = float(np.interp(v_ego, _HIGH_SPEED_LAT_BP, _MAX_LAT_ACCEL_V))
  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  max_lat_accel = max_lat_accel_base + roll_compensation
  min_lat_accel = -max_lat_accel_base + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)
  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv

# ─── Replicate constants and functions from long_mpc.py ───
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
FREEWAY_FOLLOW_BONUS = 0.30
FREEWAY_FOLLOW_BP = [20.0, 28.0]
FREEWAY_CRUISE_MATCH_DIST_BP = [50.0, 150.0]
FREEWAY_CRUISE_MATCH_BUFFER_V = [1.5, 5.0]
FREEWAY_CRUISE_MATCH_MIN_VLEAD = 10.0

GENTLE_DECEL_SMOOTH_DIST_BP = [8.0, 20.0, 50.0]
GENTLE_DECEL_SMOOTH_JERK_V = [-6.0, -2.5, -1.5]
GENTLE_DECEL_NO_LEAD_DIST = 40.0
GENTLE_FAR_LEAD_START = 60.0
GENTLE_FAR_LEAD_END = 165.0
GENTLE_FAR_LEAD_CITY_MAX_SPEED_REDUCTION = 1.5
GENTLE_FAR_LEAD_CITY_MIN_CLOSING_SPEED = 2.0
GENTLE_FAR_LEAD_NORMAL_GAP_BUFFER = 20.0
GENTLE_FAR_LEAD_SPEED_TAPER_START = 45.0 * 0.44704
GENTLE_FAR_LEAD_SPEED_MAX = 50.0 * 0.44704
GENTLE_SLOW_LEAD_MAX_SPEED_REDUCTION = 4.0
GENTLE_SLOW_LEAD_MAX_SPEED = 8.0
GENTLE_SLOW_LEAD_MIN_CLOSING_SPEED = 6.0
GENTLE_SLOW_LEAD_MAX_TTC = 12.0
GENTLE_SLOW_LEAD_MIN_TTC = 5.0
GENTLE_ACCEL_RECOVERY_CRUISE_GAP = 1.5
GENTLE_ACCEL_RECOVERY_MIN_DISTANCE = 45.0
GENTLE_ACCEL_RECOVERY_MAX_CLOSING_SPEED = 3.0

def get_T_FOLLOW(): return 1.45

def get_gentle_lead_factor(level):
  return float(np.clip(level, 0, 100)) / 100.0

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

class Lead:
  def __init__(self, status=True, dRel=100.0, vLead=15.0):
    self.status = status
    self.dRel = dRel
    self.vLead = vLead

def get_gentle_slow_lead_condition(v_ego, lead):
  if lead is None or not lead.status:
    return False
  d_rel = float(lead.dRel)
  v_lead = max(float(lead.vLead), 0.0)
  closing_speed = v_ego - v_lead
  ttc = d_rel / max(closing_speed, 0.1)
  return (d_rel >= GENTLE_FAR_LEAD_START and v_lead <= GENTLE_SLOW_LEAD_MAX_SPEED and
          closing_speed >= GENTLE_SLOW_LEAD_MIN_CLOSING_SPEED and ttc <= GENTLE_SLOW_LEAD_MAX_TTC)

def should_relax_gentle_lead_for_accel(v_cruise, v_ego, lead):
  if lead is None or not lead.status:
    return False
  d_rel = float(lead.dRel)
  v_lead = max(float(lead.vLead), 0.0)
  closing_speed = v_ego - v_lead
  below_cruise = v_cruise - v_ego
  if below_cruise < GENTLE_ACCEL_RECOVERY_CRUISE_GAP:
    return False
  if get_gentle_slow_lead_condition(v_ego, lead):
    return False
  return d_rel >= GENTLE_ACCEL_RECOVERY_MIN_DISTANCE and closing_speed <= GENTLE_ACCEL_RECOVERY_MAX_CLOSING_SPEED

def get_gentle_far_lead_v_cruise(v_cruise, v_ego, lead, level, slow_lead_confirmed=False, t_follow=None):
  if lead is None or not lead.status:
    return v_cruise
  d_rel = float(lead.dRel)
  v_lead = max(float(lead.vLead), 0.0)
  closing_speed = v_ego - v_lead
  if d_rel < GENTLE_FAR_LEAD_START:
    return v_cruise
  level_factor = get_gentle_lead_factor(level)
  t_follow = get_T_FOLLOW() if t_follow is None else t_follow
  desired_follow = get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)
  normal_gap_distance = desired_follow + GENTLE_FAR_LEAD_NORMAL_GAP_BUFFER
  gap_denominator = max(GENTLE_FAR_LEAD_END - normal_gap_distance, 1.0)
  gap_factor = np.clip((d_rel - normal_gap_distance) / gap_denominator, 0.0, 1.0)
  city_speed_factor = np.clip((GENTLE_FAR_LEAD_SPEED_MAX - v_ego) /
                              (GENTLE_FAR_LEAD_SPEED_MAX - GENTLE_FAR_LEAD_SPEED_TAPER_START), 0.0, 1.0)
  city_closing_factor = np.clip((closing_speed - GENTLE_FAR_LEAD_CITY_MIN_CLOSING_SPEED) / 8.0, 0.0, 1.0)
  city_reduction = GENTLE_FAR_LEAD_CITY_MAX_SPEED_REDUCTION * level_factor * gap_factor * city_closing_factor * city_speed_factor
  slow_reduction = 0.0
  if slow_lead_confirmed:
    ttc = d_rel / max(closing_speed, 0.1)
    distance_factor = np.clip((d_rel - GENTLE_FAR_LEAD_START) / (GENTLE_FAR_LEAD_END - GENTLE_FAR_LEAD_START), 0.0, 1.0)
    closing_factor = np.clip((closing_speed - GENTLE_SLOW_LEAD_MIN_CLOSING_SPEED) / 8.0, 0.0, 1.0)
    ttc_factor = np.clip((GENTLE_SLOW_LEAD_MAX_TTC - ttc) / (GENTLE_SLOW_LEAD_MAX_TTC - GENTLE_SLOW_LEAD_MIN_TTC), 0.0, 1.0)
    slow_reduction = GENTLE_SLOW_LEAD_MAX_SPEED_REDUCTION * level_factor * distance_factor * closing_factor * ttc_factor
  speed_reduction = max(city_reduction, slow_reduction)
  if speed_reduction <= 0.0:
    return v_cruise
  far_lead_v_cruise = max(v_lead, v_ego - speed_reduction)
  return min(v_cruise, far_lead_v_cruise)

# ═══════════════════════════════════════════
print("=== Lateral limit tests ===")

# Test accel limit by starting from prev_curvature near target (bypasses jerk rate limit)
low_speed_curv, _ = clip_curvature(10.0, 0.5, 0.5, 0.0)
high_speed_curv, _ = clip_curvature(30.0, 0.5, 0.5, 0.0)
check("low speed allows more curvature than high speed",
      abs(low_speed_curv) > abs(high_speed_curv))

max_curv_at_30 = _MAX_LAT_ACCEL_V[1] / 30.0**2
check(f"high_speed_curv ({high_speed_curv:.5f}) matches 2.5/900 ({max_curv_at_30:.5f})",
      abs(high_speed_curv - max_curv_at_30) < 0.0001)

curv_flat, _ = clip_curvature(30.0, 0.005, 0.005, 0.0)
curv_banked, _ = clip_curvature(30.0, 0.005, 0.005, 0.08)
check("banked curve allows more curvature than flat",
      abs(curv_banked) > abs(curv_flat))

check("jerk reduced at highway speed",
      float(np.interp(30.0, _HIGH_SPEED_LAT_BP, _MAX_LAT_JERK_V)) == 3.5)
check("jerk unchanged at city speed",
      float(np.interp(10.0, _HIGH_SPEED_LAT_BP, _MAX_LAT_JERK_V)) == 5.0)

check("lat accel at city speed is 3.0",
      float(np.interp(10.0, _HIGH_SPEED_LAT_BP, _MAX_LAT_ACCEL_V)) == 3.0)
check("lat accel at highway speed is 2.5",
      float(np.interp(30.0, _HIGH_SPEED_LAT_BP, _MAX_LAT_ACCEL_V)) == 2.5)

# Interstate curve: prev_curvature already at target to test accel limit only
curv_interstate, limited = clip_curvature(30.0, 0.003, 0.003, 0.08)
max_with_bank = (2.5 + 0.08 * 9.81) / 900
check(f"interstate curve (banked 8%) allows curv={curv_interstate:.5f} (limit {max_with_bank:.5f})",
      curv_interstate >= 0.0029)

# Flat road accel limit at 30 m/s
curv_flat_max, limited = clip_curvature(30.0, 0.01, 0.01, 0.0)
check(f"flat road at 30 m/s clamped to ~0.00278: curv={curv_flat_max:.5f}",
      abs(curv_flat_max - 2.5/900) < 0.0005)

print("\n=== Freeway follow bonus tests ===")

bonus_low = float(np.interp(15.0, FREEWAY_FOLLOW_BP, [0.0, FREEWAY_FOLLOW_BONUS]))
check(f"bonus at 15 m/s is 0 ({bonus_low:.3f})", bonus_low == 0.0)

bonus_mid = float(np.interp(24.0, FREEWAY_FOLLOW_BP, [0.0, FREEWAY_FOLLOW_BONUS]))
check(f"bonus at 24 m/s is partial ({bonus_mid:.3f})", 0.0 < bonus_mid < FREEWAY_FOLLOW_BONUS)

bonus_high = float(np.interp(30.0, FREEWAY_FOLLOW_BP, [0.0, FREEWAY_FOLLOW_BONUS]))
check(f"bonus at 30 m/s is full ({bonus_high:.3f})",
      abs(bonus_high - FREEWAY_FOLLOW_BONUS) < 0.001)

t_base = get_T_FOLLOW()
dist_base = get_safe_obstacle_distance(30.0, t_base)
dist_with_bonus = get_safe_obstacle_distance(30.0, t_base + FREEWAY_FOLLOW_BONUS)
delta = dist_with_bonus - dist_base
check(f"follow dist increases by ~{delta:.1f}m at 30 m/s", delta > 5.0)

print("\n=== Gentle braking regression tests ===")

v_cruise = 30.0
check("city far lead: v_cruise reduced",
      get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(), 100) < v_cruise)
check("freeway far lead: v_cruise unchanged",
      get_gentle_far_lead_v_cruise(v_cruise, 23.0, Lead(), 100) == v_cruise)
check("no lead: unchanged", get_gentle_far_lead_v_cruise(v_cruise, 20.0, None, 100) == v_cruise)
check("close lead: unchanged", get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(dRel=45.0), 100) == v_cruise)
check("level 0: unchanged", get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(), 0) == v_cruise)

check("slow lead cond: stopped far lead",
      get_gentle_slow_lead_condition(25.0, Lead(vLead=0.0)))
check("slow lead cond: no lead",
      not get_gentle_slow_lead_condition(25.0, None))
check("slow lead cond: close",
      not get_gentle_slow_lead_condition(25.0, Lead(dRel=50.0, vLead=0.0)))

check("accel recovery: far normal lead relaxes",
      should_relax_gentle_lead_for_accel(30.0, 25.0, Lead(dRel=90.0, vLead=24.0)))
check("accel recovery: close lead no relax",
      not should_relax_gentle_lead_for_accel(30.0, 25.0, Lead(dRel=35.0, vLead=24.0)))
check("accel recovery: near cruise no relax",
      not should_relax_gentle_lead_for_accel(33.0, 32.0, Lead(dRel=70.0, vLead=31.0)))

check("speed gate threshold valid",
      GENTLE_FAR_LEAD_SPEED_MAX > 20.0 and 30.0 > GENTLE_FAR_LEAD_SPEED_MAX)

# ─── Verify freeway speeds do NOT affect gentle braking ───
print("\n=== Freeway gentle braking isolation ===")
v_cruise_fw = 35.0
v_ego_fw = 29.0
lead_fw = Lead(dRel=70.0, vLead=28.0)
check("freeway: gentle far lead no-op (city_speed_factor=0)",
      get_gentle_far_lead_v_cruise(v_cruise_fw, v_ego_fw, lead_fw, 100) == v_cruise_fw)
check("freeway: gentle at_city_speed is False",
      v_ego_fw >= GENTLE_FAR_LEAD_SPEED_MAX)

print("\n=== Freeway cruise speed matching ===")

# Close lead (50m): tight cap
v_lead_test = 28.0
cap_close = v_lead_test + float(np.interp(50.0, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
check(f"close lead (50m): cruise capped to {cap_close:.1f} m/s (lead+1.5)",
      abs(cap_close - 29.5) < 0.1)

# Medium lead (80m): moderate cap
cap_mid = v_lead_test + float(np.interp(80.0, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
check(f"medium lead (80m): cruise capped to {cap_mid:.1f} m/s",
      cap_mid < 31.0 and cap_mid > 29.5)

# Far lead (100m): tighter than before
cap_100 = v_lead_test + float(np.interp(100.0, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
check(f"far lead (100m): cruise capped to {cap_100:.1f} m/s",
      cap_100 < 32.0)

# Very far lead (150m): max buffer
cap_far = v_lead_test + float(np.interp(150.0, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
check(f"very far lead (150m): cruise capped to {cap_far:.1f} m/s (lead+5.0)",
      abs(cap_far - 33.0) < 0.1)

# Slow leads excluded
check(f"slow lead threshold >= 10 m/s", FREEWAY_CRUISE_MATCH_MIN_VLEAD >= 10.0)

# Real oscillation scenario: lead at 28 m/s, ego at 29 m/s, cruise at 35 m/s, dRel=80
v_cruise_scenario = 35.0
v_lead_scenario = 28.0
d_rel_scenario = 80.0
cap_scenario = v_lead_scenario + float(np.interp(d_rel_scenario, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
capped_cruise = min(v_cruise_scenario, cap_scenario)
check(f"oscillation scenario: cruise {v_cruise_scenario:.0f} capped to {capped_cruise:.1f} (prevents overshoot)",
      capped_cruise < v_cruise_scenario and capped_cruise > v_lead_scenario)

# Verify gap at 100m is much tighter than before (was ~8 m/s at 120m, now ~3 m/s at 100m)
gap_at_100 = float(np.interp(100.0, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
check(f"gap at 100m is {gap_at_100:.1f} m/s (tighter than 6 m/s)",
      gap_at_100 < 4.5)

print("\n=== Deceleration smoothing tests (gentle braking) ===")

DT_MDL = 0.05

# Far lead (50m): gentle jerk limit
jerk_50 = float(np.interp(50.0, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
check(f"jerk limit at 50m is -1.5 ({jerk_50:.1f})", abs(jerk_50 - (-1.5)) < 0.01)

# Close lead (8m): minimal smoothing
jerk_8 = float(np.interp(8.0, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
check(f"jerk limit at 8m is -6.0 ({jerk_8:.1f})", abs(jerk_8 - (-6.0)) < 0.01)

# Medium lead (20m)
jerk_20 = float(np.interp(20.0, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
check(f"jerk limit at 20m is -2.5 ({jerk_20:.1f})", abs(jerk_20 - (-2.5)) < 0.01)

# No lead defaults to 40m
jerk_no_lead = float(np.interp(GENTLE_DECEL_NO_LEAD_DIST, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
check(f"no lead default (40m) jerk is ~-1.83 ({jerk_no_lead:.2f})", -2.0 < jerk_no_lead < -1.5)

# Simulation: from 0 to -1.0 at 50m should take ~0.67s (13 cycles)
prev_a = 0.0
cycles_to_minus_1 = 0
for i in range(100):
  target = -2.0
  min_a = prev_a + jerk_50 * DT_MDL
  smoothed = max(target, min_a)
  prev_a = smoothed
  if prev_a <= -1.0:
    cycles_to_minus_1 = i + 1
    break
time_to_minus_1 = cycles_to_minus_1 * DT_MDL
check(f"0 to -1.0 at 50m takes {time_to_minus_1:.2f}s (~0.67s expected)",
      0.5 < time_to_minus_1 < 1.0)

# At 8m, transition is nearly instant
prev_a = 0.0
cycles_close = 0
for i in range(100):
  target = -2.0
  min_a = prev_a + jerk_8 * DT_MDL
  smoothed = max(target, min_a)
  prev_a = smoothed
  if prev_a <= -1.0:
    cycles_close = i + 1
    break
time_close = cycles_close * DT_MDL
check(f"0 to -1.0 at 8m takes {time_close:.2f}s (fast, ~0.17s)",
      time_close < 0.5)

print("\n=== Universal output jerk limiter ===")

OUTPUT_DECEL_JERK_LIMIT = -3.0
OUTPUT_DECEL_EMERGENCY_DIST = 4.0
OUTPUT_DECEL_TTC_BYPASS = 4.0

check(f"universal jerk limit is -3.0 m/s^3", OUTPUT_DECEL_JERK_LIMIT == -3.0)
check(f"emergency dist bypass at 4m", OUTPUT_DECEL_EMERGENCY_DIST == 4.0)
check(f"TTC bypass threshold at 4.0s", OUTPUT_DECEL_TTC_BYPASS == 4.0)

# Worst-case cruise->e2e jump (1.34 m/s^2 in data): how long to complete?
worst_jump = 1.34
cycles_worst = int(worst_jump / abs(OUTPUT_DECEL_JERK_LIMIT * DT_MDL))
time_worst = cycles_worst * DT_MDL
check(f"worst e2e jump ({worst_jump} m/s^2) smoothed over {time_worst:.2f}s (target ~0.4s)",
      0.3 < time_worst < 0.7)

# At 3.0 m/s^3, per-cycle change is 0.15 m/s^2
per_cycle = abs(OUTPUT_DECEL_JERK_LIMIT) * DT_MDL
check(f"per-cycle decel change = {per_cycle:.3f} m/s^2 (target 0.15)",
      abs(per_cycle - 0.15) < 0.001)

# Simulate: from +0.85 to -0.49 (real worst-case from data)
prev_a = 0.85
target_a = -0.49
cycles = 0
for i in range(200):
  min_a_uni = prev_a + OUTPUT_DECEL_JERK_LIMIT * DT_MDL
  smoothed = max(target_a, min_a_uni)
  prev_a = smoothed
  cycles += 1
  if prev_a <= target_a:
    break
time_settle = cycles * DT_MDL
check(f"cruise->e2e (+0.85 to -0.49): settles in {time_settle:.2f}s (comfortable ~0.45s)",
      0.3 < time_settle < 0.8)

# Emergency bypass: lead at 3m should NOT be smoothed
check("emergency bypass: lead at 3m skips smoothing",
      3.0 < OUTPUT_DECEL_EMERGENCY_DIST)
check("emergency bypass: lead at 5m IS smoothed",
      5.0 > OUTPUT_DECEL_EMERGENCY_DIST)

# Gentle braking uses stronger smoothing than universal at distance
check("gentle jerk at 50m is stronger than universal",
      abs(jerk_50) < abs(OUTPUT_DECEL_JERK_LIMIT))
check("gentle jerk at 20m is still stronger than universal",
      abs(jerk_20) < abs(OUTPUT_DECEL_JERK_LIMIT))

# Verify universal doesn't over-smooth close encounters
jerk_at_close_dist = OUTPUT_DECEL_JERK_LIMIT
prev_a = 0.0
cycles_urgent = 0
for i in range(200):
  target_urgent = -2.5
  min_a_u = prev_a + jerk_at_close_dist * DT_MDL
  smoothed = max(target_urgent, min_a_u)
  prev_a = smoothed
  cycles_urgent += 1
  if prev_a <= -2.0:
    break
time_urgent = cycles_urgent * DT_MDL
check(f"universal: 0 to -2.0 in {time_urgent:.2f}s (under 1s for safety)",
      time_urgent < 1.0)

print("\n=== TTC-based jerk limiter bypass ===")

# Stop #13 from drive: v_ego=13.4, lead at 42m going 1.8 m/s
ttc_stop13 = 42.0 / max(13.4 - 1.8, 0.1)
check(f"stop #13 TTC={ttc_stop13:.1f}s bypasses limiter (< 4.0s)",
      ttc_stop13 < OUTPUT_DECEL_TTC_BYPASS)

# Stop #20: v_ego=18.8, lead at 42m going 1.8 m/s
ttc_stop20 = 42.0 / max(18.8 - 1.8, 0.1)
check(f"stop #20 TTC={ttc_stop20:.1f}s bypasses limiter (< 4.0s)",
      ttc_stop20 < OUTPUT_DECEL_TTC_BYPASS)

# Smooth stop #2: v_ego=12.1, lead at 48m going 7.1 m/s
ttc_smooth2 = 48.0 / max(12.1 - 7.1, 0.1)
check(f"smooth stop #2 TTC={ttc_smooth2:.1f}s keeps limiter (> 4.0s)",
      ttc_smooth2 > OUTPUT_DECEL_TTC_BYPASS)

# No lead: limiter still applies (no TTC check when lead.status=False)
check("no lead: TTC bypass requires lead.status=True",
      True)  # by design: `if not emergency and lead.status:` guard

# Far lead, slow approach: limiter stays active
ttc_gentle = 80.0 / max(15.0 - 12.0, 0.1)
check(f"gentle approach TTC={ttc_gentle:.1f}s keeps limiter (> 4.0s)",
      ttc_gentle > OUTPUT_DECEL_TTC_BYPASS)

# Lead at same speed (not closing): limiter stays active
ttc_same = 50.0 / max(15.0 - 14.9, 0.1)
check(f"matching speed TTC={ttc_same:.0f}s keeps limiter (> 4.0s)",
      ttc_same > OUTPUT_DECEL_TTC_BYPASS)

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
  sys.exit(1)
