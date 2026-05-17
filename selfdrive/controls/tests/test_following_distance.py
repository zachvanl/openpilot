import pytest
import itertools
import numpy as np
from openpilot.common.parameterized import parameterized_class

from cereal import log

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  GENTLE_DECEL_SMOOTH_DIST_BP,
  GENTLE_DECEL_SMOOTH_JERK_V,
  GENTLE_DECEL_NO_LEAD_DIST,
  OUTPUT_DECEL_JERK_LIMIT,
  OUTPUT_DECEL_EMERGENCY_DIST,
  OUTPUT_DECEL_TTC_BYPASS,
  CITY_DECEL_TTC_BYPASS,
  CITY_FAR_DECEL_TTC,
  CITY_JERK_BYPASS_TTC,
  CITY_COMFORT_DECEL_CAP,
  CITY_CLOSE_DECEL_CAP,
  city_closing_brake_active,
  city_closing_decel_jerk,
  city_comfort_decel_limit,
  city_far_decel_floor,
  city_imminent_collision,
  city_urgent_mpc_disable,
  cap_v_cruise_for_slow_lead,
  OUTPUT_DECEL_EMERGENCY_DIST,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  FREEWAY_CRUISE_MATCH_BUFFER_V,
  FREEWAY_CRUISE_MATCH_DIST_BP,
  FREEWAY_CRUISE_MATCH_MIN_VLEAD,
  FREEWAY_FOLLOW_BONUS,
  FREEWAY_FOLLOW_BP,
  GENTLE_FAR_LEAD_SPEED_MAX,
  get_gentle_far_lead_v_cruise,
  get_gentle_slow_lead_condition,
  get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
  should_relax_gentle_lead_for_accel,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


def desired_follow_distance(v_ego, v_lead, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  t_follow += float(np.interp(v_ego, FREEWAY_FOLLOW_BP, [0.0, FREEWAY_FOLLOW_BONUS]))
  return get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)

def run_following_distance_simulation(v_lead, t_end=100.0, e2e=False, personality=0):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_lead),
    lead_relevancy=True,
    initial_distance_lead=100,
    speed_lead_values=[v_lead],
    breakpoints=[0.],
    e2e=e2e,
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1,2] - output[-1,1]

class Lead:
  def __init__(self, status=True, dRel=100.0, vLead=15.0):
    self.status = status
    self.dRel = dRel
    self.vLead = vLead


def test_gentle_far_lead_speed_gate():
  v_cruise = 30.0
  assert get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(), 100) < v_cruise
  assert get_gentle_far_lead_v_cruise(v_cruise, 23.0, Lead(), 100) == v_cruise


def test_gentle_far_lead_noop_conditions():
  v_cruise = 30.0
  assert get_gentle_far_lead_v_cruise(v_cruise, 20.0, None, 100) == v_cruise
  assert get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(status=False), 100) == v_cruise
  assert get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(), 0) == v_cruise
  assert get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(dRel=100.0, vLead=19.0), 100) == v_cruise


def test_gentle_far_lead_does_not_hold_normal_city_gap():
  v_cruise = 30.0
  assert get_gentle_far_lead_v_cruise(v_cruise, 20.0, Lead(dRel=45.0), 100) == v_cruise


def test_gentle_slow_far_lead_needs_confirmation_at_freeway_speed():
  v_cruise = 30.0
  v_ego = 25.0
  assert get_gentle_far_lead_v_cruise(v_cruise, v_ego, Lead(vLead=0.0), 100) == v_cruise
  assert get_gentle_far_lead_v_cruise(v_cruise, v_ego, Lead(vLead=0.0), 100, slow_lead_confirmed=True) < v_cruise


def test_gentle_slow_lead_condition():
  assert get_gentle_slow_lead_condition(25.0, Lead(vLead=0.0))
  assert not get_gentle_slow_lead_condition(25.0, None)
  assert not get_gentle_slow_lead_condition(25.0, Lead(status=False, vLead=0.0))
  assert not get_gentle_slow_lead_condition(25.0, Lead(dRel=50.0, vLead=0.0))
  assert not get_gentle_slow_lead_condition(25.0, Lead(dRel=400.0, vLead=0.0))
  assert not get_gentle_slow_lead_condition(25.0, Lead(dRel=100.0, vLead=20.0))


def test_gentle_accel_recovery_relaxes_for_normal_far_lead():
  assert should_relax_gentle_lead_for_accel(30.0, 25.0, Lead(dRel=90.0, vLead=24.0))


def test_gentle_accel_recovery_keeps_caution_for_close_or_closing_leads():
  assert not should_relax_gentle_lead_for_accel(30.0, 25.0, Lead(dRel=35.0, vLead=24.0))
  assert not should_relax_gentle_lead_for_accel(30.0, 25.0, Lead(dRel=90.0, vLead=18.0))


def test_gentle_accel_recovery_keeps_caution_for_slow_stopped_far_lead():
  assert not should_relax_gentle_lead_for_accel(30.0, 25.0, Lead(dRel=100.0, vLead=0.0))


def test_gentle_accel_recovery_not_used_at_near_cruise_speed():
  assert not should_relax_gentle_lead_for_accel(33.0, 32.0, Lead(dRel=70.0, vLead=31.0))


def test_gentle_weight_speed_gate_threshold():
  assert GENTLE_FAR_LEAD_SPEED_MAX > 20.0
  assert 30.0 > GENTLE_FAR_LEAD_SPEED_MAX


def test_freeway_follow_bonus_zero_at_low_speed():
  bonus = float(np.interp(15.0, FREEWAY_FOLLOW_BP, [0.0, FREEWAY_FOLLOW_BONUS]))
  assert bonus == 0.0


def test_freeway_follow_bonus_full_at_highway_speed():
  bonus = float(np.interp(30.0, FREEWAY_FOLLOW_BP, [0.0, FREEWAY_FOLLOW_BONUS]))
  assert bonus == pytest.approx(FREEWAY_FOLLOW_BONUS)


def test_freeway_follow_bonus_increases_follow_distance():
  t_base = get_T_FOLLOW()
  dist_base = get_safe_obstacle_distance(30.0, t_base)
  dist_with_bonus = get_safe_obstacle_distance(30.0, t_base + FREEWAY_FOLLOW_BONUS)
  assert dist_with_bonus > dist_base


def test_freeway_cruise_match_caps_close_lead():
  v_lead = 28.0
  d_rel = 50.0
  cap = v_lead + float(np.interp(d_rel, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
  assert cap < 30.0  # at min distance, cap is very tight (lead + 1.5)


def test_freeway_cruise_match_tighter_at_medium_distance():
  v_lead = 28.0
  d_rel = 100.0
  cap = v_lead + float(np.interp(d_rel, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
  assert cap < 32.0  # at 100m, cap is moderate (~lead + 3.25)


def test_freeway_cruise_match_ignores_slow_leads():
  assert FREEWAY_CRUISE_MATCH_MIN_VLEAD >= 10.0


def test_decel_smoothing_gentle_at_far_distance():
  jerk = float(np.interp(50.0, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
  assert jerk == pytest.approx(-1.5)


def test_decel_smoothing_fast_at_close_distance():
  jerk = float(np.interp(8.0, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
  assert jerk == pytest.approx(-6.0)


def test_decel_smoothing_no_lead_default():
  jerk = float(np.interp(GENTLE_DECEL_NO_LEAD_DIST, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
  assert -2.0 < jerk < -1.5


def test_universal_jerk_limit_value():
  assert OUTPUT_DECEL_JERK_LIMIT == pytest.approx(-3.0)


def test_universal_jerk_limit_per_cycle():
  DT_MDL = 0.05
  per_cycle = abs(OUTPUT_DECEL_JERK_LIMIT) * DT_MDL
  assert per_cycle == pytest.approx(0.15)


def test_universal_jerk_emergency_bypass():
  assert OUTPUT_DECEL_EMERGENCY_DIST == pytest.approx(4.0)


def test_universal_jerk_gentler_than_gentle_close():
  jerk_gentle_close = float(np.interp(8.0, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
  assert abs(OUTPUT_DECEL_JERK_LIMIT) < abs(jerk_gentle_close)


def test_universal_jerk_allows_strong_decel_under_1s():
  DT_MDL = 0.05
  prev_a = 0.0
  target = -2.0
  cycles = 0
  while prev_a > target and cycles < 200:
    min_a = prev_a + OUTPUT_DECEL_JERK_LIMIT * DT_MDL
    prev_a = max(target, min_a)
    cycles += 1
  assert cycles * DT_MDL < 1.0


def test_ttc_bypass_threshold():
  assert OUTPUT_DECEL_TTC_BYPASS == pytest.approx(4.0)


def test_ttc_bypass_urgent_stop():
  ttc = 42.0 / max(13.4 - 1.8, 0.1)
  assert ttc < OUTPUT_DECEL_TTC_BYPASS


def test_ttc_bypass_preserves_gentle_approach():
  ttc = 48.0 / max(12.1 - 7.1, 0.1)
  assert ttc > OUTPUT_DECEL_TTC_BYPASS


def test_city_ttc_bypass_more_lenient_than_highway():
  assert CITY_DECEL_TTC_BYPASS > OUTPUT_DECEL_TTC_BYPASS


def test_city_jerk_bypass_matches_highway():
  assert CITY_JERK_BYPASS_TTC == OUTPUT_DECEL_TTC_BYPASS


def test_city_far_decel_floor_stronger_when_far():
  assert city_far_decel_floor(30.0) < city_far_decel_floor(90.0)


def test_city_comfort_cap_limits_emergency():
  assert CITY_COMFORT_DECEL_CAP > -3.5
  assert city_comfort_decel_limit(40.0, 6.0, True) == CITY_COMFORT_DECEL_CAP
  assert city_comfort_decel_limit(12.0, 3.0, True) == CITY_COMFORT_DECEL_CAP
  assert city_comfort_decel_limit(8.0, 1.0, True) == ACCEL_MIN


def test_city_comfort_limit_smooth_near_close():
  mid = city_comfort_decel_limit(12.0, 2.0, True)
  assert CITY_COMFORT_DECEL_CAP >= mid >= CITY_CLOSE_DECEL_CAP


def test_city_closing_decel_jerk_moderate_mid_ttc():
  assert city_closing_decel_jerk(6.0, 30.0, 15.0) == -2.0
  assert city_closing_decel_jerk(2.0, 10.0, 15.0) == OUTPUT_DECEL_JERK_LIMIT


def test_city_urgent_mpc_only_when_close():
  lead = _LeadStub(True, 40.0, 1.0)
  assert not city_urgent_mpc_disable(lead, 15.0)
  assert city_urgent_mpc_disable(_LeadStub(True, 10.0, 1.0), 15.0)


def test_city_imminent_collision():
  assert city_imminent_collision(10.0, 5.0, True)
  assert city_imminent_collision(20.0, 2.0, True)
  assert not city_imminent_collision(30.0, 5.0, True)


class _LeadStub:
  def __init__(self, status, d_rel, v_lead):
    self.status = status
    self.dRel = d_rel
    self.vLead = v_lead


def test_city_closing_brake_active_slow_lead():
  lead = _LeadStub(True, 40.0, 1.0)
  assert city_closing_brake_active(lead, 15.0)


def test_city_closing_brake_inactive_highway_speed():
  lead = _LeadStub(True, 40.0, 1.0)
  assert not city_closing_brake_active(lead, 30.0)


def test_cap_v_cruise_for_slow_lead():
  lead = _LeadStub(True, 25.0, 2.0)
  capped = cap_v_cruise_for_slow_lead(20.0, lead, 15.0)
  assert capped < 20.0
  assert capped >= 2.0


def test_clip_curvature_speed_dependent_limits():
  from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature, _MAX_LAT_ACCEL_V
  low_speed_curv, _ = clip_curvature(10.0, 0.5, 0.5, 0.0)
  high_speed_curv, _ = clip_curvature(30.0, 0.5, 0.5, 0.0)
  assert abs(low_speed_curv) > abs(high_speed_curv)
  max_curv_at_30 = _MAX_LAT_ACCEL_V[1] / 30.0**2
  assert high_speed_curv == pytest.approx(max_curv_at_30, abs=0.0001)


def test_clip_curvature_roll_compensation_highway():
  from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
  curv_flat, _ = clip_curvature(30.0, 0.005, 0.005, 0.0)
  curv_banked, _ = clip_curvature(30.0, 0.005, 0.005, 0.08)
  assert abs(curv_banked) > abs(curv_flat)


def test_get_t_follow_param_overrides():
  assert get_T_FOLLOW(log.LongitudinalPersonality.aggressive, (1.0, 1.45, 1.75)) == pytest.approx(1.0)
  assert get_T_FOLLOW(log.LongitudinalPersonality.standard, (1.25, 1.1, 1.75)) == pytest.approx(1.1)
  assert get_T_FOLLOW(log.LongitudinalPersonality.relaxed, (1.25, 1.45, 1.9)) == pytest.approx(1.9)
  assert get_T_FOLLOW(log.LongitudinalPersonality.aggressive, (0.5, 1.45, 1.75)) == pytest.approx(0.80)  # clamped
  assert get_T_FOLLOW(log.LongitudinalPersonality.relaxed, (1.25, 1.45, 3.0)) == pytest.approx(2.20)


@parameterized_class(("e2e", "personality", "speed"), itertools.product(
                      [True, False], # e2e
                      [log.LongitudinalPersonality.relaxed, # personality
                       log.LongitudinalPersonality.standard,
                       log.LongitudinalPersonality.aggressive],
                      [0,10,35])) # speed
class TestFollowingDistance:
  def test_following_distance(self):
    v_lead = float(self.speed)
    simulation_steady_state = run_following_distance_simulation(v_lead, e2e=self.e2e, personality=self.personality)
    correct_steady_state = desired_follow_distance(v_lead, v_lead, get_T_FOLLOW(self.personality))
    err_ratio = 0.2 if self.e2e else 0.1
    abs_err_margin = 0.5 if v_lead > 0.0 else 1.15
    assert simulation_steady_state == pytest.approx(correct_steady_state, abs=err_ratio * correct_steady_state + abs_err_margin)
