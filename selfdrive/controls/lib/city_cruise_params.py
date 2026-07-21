#!/usr/bin/env python3
"""Sunnylink-tunable city cruise assist (approach slowdown, established coast, late brake)."""
from dataclasses import dataclass

import numpy as np

from openpilot.common.constants import CV
from openpilot.common.params import Params

FT_TO_M = 0.3048

# Code defaults (also params_keys.h defaults where applicable).
DEFAULT_APPROACH_MIN_FT = 100.0
DEFAULT_APPROACH_MAX_FT = 150.0
DEFAULT_APPROACH_MIN_CLOSING_MPH = 7.0
DEFAULT_APPROACH_FAR_BUFFER_MPH = 20.0
DEFAULT_ESTABLISHED_FOLLOW_S = 6.0
DEFAULT_COAST_LEAD_ACCEL = -0.9
DEFAULT_HARD_BRAKE_LEAD_ACCEL = -2.0


@dataclass(frozen=True)
class CityCruiseParams:
  enabled: bool
  approach_enabled: bool
  approach_min_dist_m: float
  approach_max_dist_m: float
  approach_min_closing: float
  approach_buffer_v: tuple[float, float, float]
  established_coast_enabled: bool
  established_follow_time: float
  late_brake_assist_enabled: bool
  coast_max_lead_alead: float
  hard_brake_lead_alead: float

  @classmethod
  def defaults(cls) -> "CityCruiseParams":
    return cls.from_values()

  @classmethod
  def from_values(
    cls,
    *,
    enabled: bool = True,
    approach_enabled: bool = True,
    approach_min_ft: float = DEFAULT_APPROACH_MIN_FT,
    approach_max_ft: float = DEFAULT_APPROACH_MAX_FT,
    approach_min_closing_mph: float = DEFAULT_APPROACH_MIN_CLOSING_MPH,
    approach_far_buffer_mph: float = DEFAULT_APPROACH_FAR_BUFFER_MPH,
    established_coast_enabled: bool = True,
    established_follow_s: float = DEFAULT_ESTABLISHED_FOLLOW_S,
    late_brake_assist_enabled: bool = True,
    coast_max_lead_alead: float = DEFAULT_COAST_LEAD_ACCEL,
    hard_brake_lead_alead: float = DEFAULT_HARD_BRAKE_LEAD_ACCEL,
  ) -> "CityCruiseParams":
    min_m = float(approach_min_ft) * FT_TO_M
    max_m = max(float(approach_max_ft) * FT_TO_M, min_m + 3.0)
    far_buf = float(approach_far_buffer_mph) * CV.MPH_TO_MS
    return cls(
      enabled=enabled,
      approach_enabled=approach_enabled,
      approach_min_dist_m=min_m,
      approach_max_dist_m=max_m,
      approach_min_closing=float(approach_min_closing_mph) * CV.MPH_TO_MS,
      approach_buffer_v=(far_buf, far_buf * 0.67, far_buf * 0.39),
      established_coast_enabled=established_coast_enabled,
      established_follow_time=float(np.clip(established_follow_s, 3.0, 15.0)),
      late_brake_assist_enabled=late_brake_assist_enabled,
      coast_max_lead_alead=float(np.clip(coast_max_lead_alead, -2.0, 0.0)),
      hard_brake_lead_alead=float(np.clip(hard_brake_lead_alead, -5.0, -0.5)),
    )

  @classmethod
  def from_params(cls, params: Params | None = None) -> "CityCruiseParams":
    p = params or Params()
    return cls.from_values(
      enabled=p.get_bool("CityCruiseAssist"),
      approach_enabled=p.get_bool("CityApproachEarlySlow"),
      approach_min_ft=float(p.get("CityApproachMinDistFt", return_default=True)),
      approach_max_ft=float(p.get("CityApproachMaxDistFt", return_default=True)),
      approach_min_closing_mph=float(p.get("CityApproachMinClosingMph", return_default=True)),
      approach_far_buffer_mph=float(p.get("CityApproachFarBufferMph", return_default=True)),
      established_coast_enabled=p.get_bool("CityEstablishedFollowCoast"),
      established_follow_s=float(p.get("CityEstablishedFollowTimeS", return_default=True)),
      late_brake_assist_enabled=p.get_bool("CityLateBrakeAssist"),
      coast_max_lead_alead=float(p.get("CityCoastMaxLeadAccel", return_default=True)),
      hard_brake_lead_alead=float(p.get("CityHardBrakeLeadAccel", return_default=True)),
    )
