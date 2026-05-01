import pytest
import itertools
import numpy as np
from openpilot.common.parameterized import parameterized_class

from cereal import log

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
  d_rel = 60.0
  cap = v_lead + float(np.interp(d_rel, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
  assert cap < 35.0  # cruise at 78 mph would be capped well below


def test_freeway_cruise_match_relaxes_at_distance():
  v_lead = 28.0
  d_rel = 120.0
  cap = v_lead + float(np.interp(d_rel, FREEWAY_CRUISE_MATCH_DIST_BP, FREEWAY_CRUISE_MATCH_BUFFER_V))
  assert cap > 35.0  # at max distance the buffer is large enough to not interfere


def test_freeway_cruise_match_ignores_slow_leads():
  assert FREEWAY_CRUISE_MATCH_MIN_VLEAD >= 10.0


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
