import pytest
import itertools
from openpilot.common.parameterized import parameterized_class

from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_gentle_far_lead_v_cruise,
  get_gentle_slow_lead_condition,
  get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


def desired_follow_distance(v_ego, v_lead, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
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
