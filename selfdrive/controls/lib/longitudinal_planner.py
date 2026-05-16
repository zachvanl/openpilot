#!/usr/bin/env python3
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import should_relax_gentle_lead_for_accel, GENTLE_FAR_LEAD_SPEED_MAX
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
PARAMS_UPDATE_PERIOD = 1.0

GENTLE_DECEL_SMOOTH_DIST_BP = [8.0, 20.0, 50.0]
GENTLE_DECEL_SMOOTH_JERK_V = [-6.0, -2.5, -1.5]
GENTLE_DECEL_NO_LEAD_DIST = 40.0

OUTPUT_DECEL_JERK_LIMIT = -3.0
OUTPUT_DECEL_EMERGENCY_DIST = 4.0
OUTPUT_DECEL_TTC_BYPASS = 4.0
# Jerk-limit bypass uses the same TTC at city and highway; far city braking uses separate floors/caps.
CITY_JERK_BYPASS_TTC = 4.0
CITY_FAR_DECEL_TTC = 8.0
CITY_COMFORT_DECEL_CAP = -2.8
CITY_IMMINENT_TTC = 2.5
CITY_IMMINENT_DIST = 15.0
# Back-compat alias for tests/tools that referenced the old combined constant.
CITY_DECEL_TTC_BYPASS = CITY_FAR_DECEL_TTC

CITY_STOP_MAX_V_EGO = 14.0  # m/s (~31 mph)
CITY_STOP_MAX_V_LEAD = 2.5
CITY_STOP_MAX_D_REL = 50.0
CITY_FOLLOW_CAP_MAX_V_EGO = 22.0  # m/s (~49 mph)
CITY_FOLLOW_CAP_MAX_V_LEAD = 5.0
CITY_FOLLOW_CAP_MAX_D_REL = 100.0
CITY_CLOSING_MIN_SPEED = 2.0  # m/s closing rate to treat as urgent


def city_closing_brake_active(lead, v_ego: float) -> bool:
  if lead is None or not lead.status:
    return False
  v_lead = max(float(lead.vLead), 0.0)
  d_rel = float(lead.dRel)
  closing = v_ego - v_lead
  return (v_ego < GENTLE_FAR_LEAD_SPEED_MAX and v_lead < CITY_FOLLOW_CAP_MAX_V_LEAD and
          d_rel < CITY_FOLLOW_CAP_MAX_D_REL and closing > CITY_CLOSING_MIN_SPEED)


def cap_v_cruise_for_slow_lead(v_cruise: float, lead, v_ego: float) -> float:
  if lead is None or not lead.status or v_ego >= CITY_FOLLOW_CAP_MAX_V_EGO:
    return v_cruise
  v_lead = max(float(lead.vLead), 0.0)
  d_rel = float(lead.dRel)
  if v_lead >= CITY_FOLLOW_CAP_MAX_V_LEAD or d_rel >= CITY_FOLLOW_CAP_MAX_D_REL:
    return v_cruise
  buffer = float(np.interp(d_rel, [8.0, 25.0, 60.0, 100.0], [0.0, 1.5, 5.0, 10.0]))
  return min(v_cruise, max(v_lead + buffer, 0.5))


def city_imminent_collision(d_rel: float, ttc: float, lead_status: bool) -> bool:
  if not lead_status:
    return False
  return d_rel < CITY_IMMINENT_DIST or ttc < CITY_IMMINENT_TTC


def city_far_decel_floor(d_rel: float) -> float:
  return float(np.interp(d_rel, [30.0, 60.0, 100.0], [-2.5, -2.0, -1.0]))

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.params = Params()
    self.frame = -1
    self.gentle_lead_braking = False
    self.gentle_lead_braking_far_lead = True
    self.gentle_lead_braking_level = 50
    self.t_follow_overrides: tuple[float, float, float] | None = None

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

    self.update_gentle_lead_braking_params()
    self.update_t_follow_params()

  def update_gentle_lead_braking_params(self):
    self.gentle_lead_braking = self.params.get_bool("GentleLeadBraking")
    self.gentle_lead_braking_far_lead = self.params.get_bool("GentleLeadBrakingFarLead")
    self.gentle_lead_braking_level = int(np.clip(self.params.get("GentleLeadBrakingLevel", return_default=True), 0, 100))

  def update_t_follow_params(self):
    self.t_follow_overrides = (
      float(self.params.get("LongitudinalTFollowAggressive", return_default=True)),
      float(self.params.get("LongitudinalTFollowStandard", return_default=True)),
      float(self.params.get("LongitudinalTFollowRelaxed", return_default=True)),
    )

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)
    self.frame += 1
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.update_gentle_lead_braking_params()
      self.update_t_follow_params()

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    lead_one = sm['radarState'].leadOne

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired, v_cruise)
    v_cruise = cap_v_cruise_for_slow_lead(v_cruise, lead_one, v_ego)

    if force_slow_decel:
      v_cruise = 0.0

    gentle_lead_enabled = self.CP.openpilotLongitudinalControl and self.gentle_lead_braking and not reset_state
    gentle_at_city_speed = v_ego < GENTLE_FAR_LEAD_SPEED_MAX
    urgent_city_close = city_closing_brake_active(lead_one, v_ego)
    gentle_accel_recovery = gentle_lead_enabled and gentle_at_city_speed and should_relax_gentle_lead_for_accel(v_cruise, v_ego, lead_one)
    gentle_lead_smoothing_enabled = (gentle_lead_enabled and gentle_at_city_speed and not gentle_accel_recovery and
                                     not urgent_city_close)
    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality,
                         gentle_lead_enabled=gentle_lead_smoothing_enabled, gentle_lead_level=self.gentle_lead_braking_level)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality,
                    gentle_lead_enabled=gentle_lead_enabled, gentle_lead_level=self.gentle_lead_braking_level,
                    gentle_far_lead_enabled=self.gentle_lead_braking_far_lead,
                    t_follow_overrides=self.t_follow_overrides)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    if self.is_e2e(sm):
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    lead = lead_one
    if self.is_e2e(sm) and lead.status:
      v_lead = max(float(lead.vLead), 0.0)
      d_rel = float(lead.dRel)
      if v_ego < CITY_STOP_MAX_V_EGO and v_lead < CITY_STOP_MAX_V_LEAD and d_rel < CITY_STOP_MAX_D_REL:
        self.output_should_stop = True
        output_a_target = min(output_a_target, -1.5)
    if self.is_e2e(sm) and (output_should_stop_e2e or (v_ego < 10.0 and output_a_target_e2e < -1.0)):
      self.output_should_stop = True
      output_a_target = min(output_a_target, output_a_target_e2e)

    if self.is_e2e(sm) and lead.status and city_closing_brake_active(lead, v_ego):
      v_lead = max(float(lead.vLead), 0.0)
      d_rel = float(lead.dRel)
      output_a_target = min(output_a_target, city_far_decel_floor(d_rel))
      closing_speed = max(v_ego - v_lead, 0.1)
      ttc = d_rel / closing_speed
      if not city_imminent_collision(d_rel, ttc, lead.status):
        output_a_target = max(output_a_target, CITY_COMFORT_DECEL_CAP)

    if not self.output_should_stop and output_a_target < self.output_a_target:
      d_rel = float(lead.dRel) if lead.status else GENTLE_DECEL_NO_LEAD_DIST
      emergency = lead.status and d_rel < OUTPUT_DECEL_EMERGENCY_DIST

      if not emergency and lead.status:
        v_lead = max(float(lead.vLead), 0.0)
        closing_speed = max(v_ego - v_lead, 0.1)
        ttc = d_rel / closing_speed
        ttc_bypass = CITY_JERK_BYPASS_TTC if v_ego < GENTLE_FAR_LEAD_SPEED_MAX else OUTPUT_DECEL_TTC_BYPASS
        if ttc < ttc_bypass:
          emergency = True
        elif (v_ego < GENTLE_FAR_LEAD_SPEED_MAX and city_closing_brake_active(lead, v_ego) and
              ttc < CITY_FAR_DECEL_TTC):
          emergency = True

      if not emergency:
        base_jerk = OUTPUT_DECEL_JERK_LIMIT
        urgent_city_close = city_closing_brake_active(lead, v_ego)
        if gentle_lead_enabled and d_rel > GENTLE_DECEL_SMOOTH_DIST_BP[0] and not urgent_city_close:
          base_jerk = float(np.interp(d_rel, GENTLE_DECEL_SMOOTH_DIST_BP, GENTLE_DECEL_SMOOTH_JERK_V))
        min_a = self.output_a_target + base_jerk * self.dt
        output_a_target = max(output_a_target, min_a)

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
