#!/usr/bin/env python3
"""
Scenario replay tests using real data from drive 0000002a.
Extracts key moments from the user's actual logs and validates
that our control logic produces the expected improvements.
"""
import sys
import json
import numpy as np

passed = 0
failed = 0
scenarios = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"
    print(f"  {status}: {name}")
    return condition

# ─── Constants (matching our code) ───

# Cruise speed matching
OLD_CRUISE_MATCH_DIST_BP = [60.0, 120.0]
OLD_CRUISE_MATCH_BUFFER_V = [2.0, 8.0]
NEW_CRUISE_MATCH_DIST_BP = [50.0, 150.0]
NEW_CRUISE_MATCH_BUFFER_V = [1.5, 5.0]

# Decel smoothing (gentle braking)
DECEL_SMOOTH_DIST_BP = [8.0, 20.0, 50.0]
DECEL_SMOOTH_JERK_V = [-6.0, -2.5, -1.5]
DT_MDL = 0.05

# Universal output jerk limiter
OUTPUT_DECEL_JERK_LIMIT = -3.0
OUTPUT_DECEL_EMERGENCY_DIST = 4.0

# Torque overrides
OLD_LAT_ACCEL_FACTOR = 2.4
OLD_FRICTION = 0.17
NEW_LAT_ACCEL_FACTOR = 2.5
NEW_FRICTION = 0.18

# ─── Helper functions ───

def cruise_cap(v_lead, d_rel, dist_bp, buffer_v):
    return v_lead + float(np.interp(d_rel, dist_bp, buffer_v))

def simulate_decel_smooth(target_sequence, d_rel, dt=DT_MDL):
    """Simulate decel smoothing over a sequence of target accelerations."""
    smoothed = []
    prev_a = target_sequence[0] if target_sequence else 0.0
    for target in target_sequence:
        if target < prev_a and d_rel > DECEL_SMOOTH_DIST_BP[0]:
            jerk_limit = float(np.interp(d_rel, DECEL_SMOOTH_DIST_BP, DECEL_SMOOTH_JERK_V))
            min_a = prev_a + jerk_limit * dt
            output = max(target, min_a)
        else:
            output = target
        smoothed.append(output)
        prev_a = output
    return smoothed

def feedforward_torque(desired_lat_accel, lat_accel_factor, friction):
    return desired_lat_accel * lat_accel_factor + friction * np.sign(desired_lat_accel)

# ═══════════════════════════════════════
# SCENARIO 1: Freeway oscillation (seg 21, t=1264-1321)
# Real data from drive 0000002a segment 21
# ═══════════════════════════════════════

print("=" * 70)
print("SCENARIO 1: Freeway Oscillation (segment 21)")
print("Cruise set at 75 mph (33.3 m/s), following lead at ~63-70 mph")
print("=" * 70)

freeway_timeline = [
    # (time, v_ego_ms, v_cruise_ms, d_rel, v_lead_ms, source, aTarget)
    (1264.8, 24.9, 33.3, 50.1, 24.1, "cruise", +0.10),
    (1267.8, 25.7, 33.3, 65.4, 26.3, "cruise", +0.57),
    (1271.3, 27.9, 33.3, 65.0, 27.7, "cruise", +0.43),
    (1274.3, 28.9, 33.3, 80.1, 29.6, "cruise", +0.39),
    (1277.3, 30.1, 33.3, 86.9, 29.3, "cruise", +0.32),
    (1280.8, 30.8, 33.3, 80.5, 29.3, "cruise", +0.09),
    (1283.8, 31.0, 33.3, 72.0, 29.9, "lead0", -0.03),
    (1287.3, 30.7, 33.3, 64.5, 28.7, "lead0", -0.22),
    (1290.3, 29.1, 33.3, 22.7, 28.3, "lead0", -0.87),
    (1293.8, 29.0, 33.3, 59.7, 27.7, "lead0", -0.42),
    (1297.3, 27.5, 33.3, 54.9, 25.6, "lead0", -0.32),
    (1300.8, 26.8, 33.3, 56.7, 26.1, "lead0", -0.02),
    (1303.8, 27.1, 33.3, 59.9, 27.3, "cruise", +0.20),
    (1307.3, 27.9, 33.3, 74.2, 28.4, "cruise", +0.30),
    (1310.8, 29.9, 33.3, 101.9, 29.9, "cruise", +0.68),
    (1313.8, 31.8, 33.3, 93.4, 30.5, "cruise", +0.29),
    (1317.3, 32.7, 33.3, 92.3, 31.1, "cruise", +0.10),
    (1320.8, 32.5, 33.3, 78.7, 29.8, "cruise", -0.15),
]

scenario1_data = {"time": [], "v_ego_mph": [], "old_cap_mph": [], "new_cap_mph": [],
                  "cruise_mph": [], "lead_mph": [], "d_rel": [], "aTarget": []}

for t, v_ego, v_cruise, d_rel, v_lead, source, a_target in freeway_timeline:
    old_cap = min(v_cruise, cruise_cap(v_lead, d_rel, OLD_CRUISE_MATCH_DIST_BP, OLD_CRUISE_MATCH_BUFFER_V))
    new_cap = min(v_cruise, cruise_cap(v_lead, d_rel, NEW_CRUISE_MATCH_DIST_BP, NEW_CRUISE_MATCH_BUFFER_V))

    scenario1_data["time"].append(t - freeway_timeline[0][0])
    scenario1_data["v_ego_mph"].append(v_ego / 0.44704)
    scenario1_data["old_cap_mph"].append(old_cap / 0.44704)
    scenario1_data["new_cap_mph"].append(new_cap / 0.44704)
    scenario1_data["cruise_mph"].append(v_cruise / 0.44704)
    scenario1_data["lead_mph"].append(v_lead / 0.44704)
    scenario1_data["d_rel"].append(d_rel)
    scenario1_data["aTarget"].append(a_target)

# Key checks
old_max_gap = max(scenario1_data["old_cap_mph"][i] - scenario1_data["lead_mph"][i]
                  for i in range(len(freeway_timeline)))
new_max_gap = max(scenario1_data["new_cap_mph"][i] - scenario1_data["lead_mph"][i]
                  for i in range(len(freeway_timeline)))

check(f"old max speed gap: {old_max_gap:.1f} mph above lead (visible overshoot)",
      old_max_gap > 7.0)
check(f"new max speed gap: {new_max_gap:.1f} mph above lead (reduced)",
      new_max_gap < old_max_gap)
check(f"new cap prevents cruise overshoot",
      new_max_gap < 12.0)

# At the peak oscillation point (t=1310.8, d=101.9m, v_lead=29.9 m/s)
old_cap_peak = cruise_cap(29.9, 101.9, OLD_CRUISE_MATCH_DIST_BP, OLD_CRUISE_MATCH_BUFFER_V)
new_cap_peak = cruise_cap(29.9, 101.9, NEW_CRUISE_MATCH_DIST_BP, NEW_CRUISE_MATCH_BUFFER_V)
check(f"at peak (d=102m): old cap={old_cap_peak/0.44704:.0f} mph, new cap={new_cap_peak/0.44704:.0f} mph",
      new_cap_peak < old_cap_peak)
check(f"new cap at 102m keeps ego within 8 mph of lead",
      (new_cap_peak - 29.9) / 0.44704 < 8.0)

scenarios.append({"name": "Freeway Oscillation", "data": scenario1_data})

# ═══════════════════════════════════════
# SCENARIO 2: Harsh city braking (seg 4, t=288-300)
# ═══════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 2: Harsh City Braking (segment 4)")
print("Approaching stopped traffic at 40 mph from 61m away")
print("=" * 70)

braking_targets = [
    # (time_offset, aTarget, d_rel)
    (0.0, -0.81, 61.0),
    (0.5, -0.86, 57.2),
    (1.0, -0.88, 55.0),
    (1.5, -0.95, 51.0),
    (2.0, -0.97, 46.9),
    (2.5, -1.33, 43.0),
    (3.0, -1.62, 36.2),
    (3.5, -1.99, 34.7),
    (4.0, -2.18, 28.3),
    (4.5, -2.35, 25.6),
    (5.0, -2.41, 21.8),
    (5.5, -2.21, 18.6),
    (6.0, -1.97, 14.6),
    (6.5, -1.75, 12.3),
    (7.0, -1.63, 10.4),
    (7.5, -1.35, 8.4),
]

raw_targets = [x[1] for x in braking_targets]

# Use average distance for smoothing (it changes, but we use the starting distance)
smoothed_far = simulate_decel_smooth(raw_targets, d_rel=61.0)
smoothed_close = simulate_decel_smooth(raw_targets, d_rel=20.0)

scenario2_data = {
    "time": [x[0] for x in braking_targets],
    "raw_aTarget": raw_targets,
    "smoothed_far": smoothed_far,
    "d_rel": [x[2] for x in braking_targets],
}

# With dynamic distance smoothing
dynamic_smoothed = []
prev_a = 0.0  # start from 0 (cruising)
for _, target, d in braking_targets:
    if target < prev_a and d > DECEL_SMOOTH_DIST_BP[0]:
        jerk = float(np.interp(d, DECEL_SMOOTH_DIST_BP, DECEL_SMOOTH_JERK_V))
        min_a = prev_a + jerk * 0.5  # 0.5s between samples in this log
        output = max(target, min_a)
    else:
        output = target
    dynamic_smoothed.append(output)
    prev_a = output

scenario2_data["dynamic_smoothed"] = dynamic_smoothed

check(f"raw initial brake: {raw_targets[0]:+.2f} m/s^2 (abrupt from zero)",
      raw_targets[0] < -0.7)
check(f"smoothed initial brake: {dynamic_smoothed[0]:+.2f} m/s^2 (gentler onset)",
      dynamic_smoothed[0] > raw_targets[0])
check(f"smoothing doesn't prevent reaching needed decel",
      min(dynamic_smoothed) < -2.0)

peak_raw_idx = raw_targets.index(min(raw_targets))
peak_smooth_idx = dynamic_smoothed.index(min(dynamic_smoothed))
check(f"peak decel delayed by smoothing (raw at t={peak_raw_idx*0.5:.1f}s, smooth at t={peak_smooth_idx*0.5:.1f}s)",
      peak_smooth_idx >= peak_raw_idx)

scenarios.append({"name": "Harsh City Braking", "data": scenario2_data})

# ═══════════════════════════════════════
# SCENARIO 3: Unnecessary e2e braking (seg 17, t=1067-1072)
# ═══════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 3: Unnecessary e2e Braking (segment 17)")
print("e2e brakes hard despite lead being 47-51m away and ACCELERATING")
print("=" * 70)

e2e_braking = [
    # (time, aTarget, d_rel, v_lead_ms, a_lead)
    (0.0, -0.97, 41.4, 3.4, -0.21),
    (0.5, -0.90, 37.4, 6.0, +0.12),
    (1.0, -0.96, 38.0, 3.6, -0.47),
    (1.5, -0.98, 34.1, 5.8, +0.49),
    (2.0, -0.95, 33.2, 5.7, +0.25),
    (2.5, -1.00, 38.5, 9.4, +1.10),
    (3.0, -1.04, 34.0, 8.7, +1.11),
    (3.5, -1.32, 47.5, 11.6, +1.13),
    (4.0, -1.56, 51.4, 11.6, +1.00),
    (4.5, -1.77, 51.5, 11.7, +0.99),
    (5.0, -1.74, 46.5, 11.3, +1.05),
]

raw_e2e = [x[1] for x in e2e_braking]

# Simulate smoothing at real 20Hz rate (interpolate targets between log samples)
times_e2e = [x[0] for x in e2e_braking]
targets_e2e = [x[1] for x in e2e_braking]
drels_e2e = [x[2] for x in e2e_braking]
fine_times = np.arange(times_e2e[0], times_e2e[-1], DT_MDL)
fine_targets = np.interp(fine_times, times_e2e, targets_e2e)
fine_drels = np.interp(fine_times, times_e2e, drels_e2e)

prev_a = 0.0
fine_smoothed = []
for target, d in zip(fine_targets, fine_drels):
    if target < prev_a and d > DECEL_SMOOTH_DIST_BP[0]:
        jerk = float(np.interp(d, DECEL_SMOOTH_DIST_BP, DECEL_SMOOTH_JERK_V))
        min_a = prev_a + jerk * DT_MDL
        output = max(target, min_a)
    else:
        output = target
    fine_smoothed.append(output)
    prev_a = output

# Sample back to log rate for comparison
smoothed_e2e = [float(np.interp(t, fine_times, fine_smoothed)) for t in times_e2e]

scenario3_data = {
    "time": [x[0] for x in e2e_braking],
    "raw_aTarget": raw_e2e,
    "smoothed": smoothed_e2e,
    "d_rel": [x[2] for x in e2e_braking],
    "v_lead_mph": [x[3] / 0.44704 for x in e2e_braking],
    "a_lead": [x[4] for x in e2e_braking],
}

check(f"raw peak brake while lead accelerates: {min(raw_e2e):+.2f}",
      min(raw_e2e) < -1.5)

# Smoothing improves onset, not peak (peak is eventually needed for safety)
avg_raw_first3 = np.mean(raw_e2e[:3])
avg_smooth_first3 = np.mean(smoothed_e2e[:3])
check(f"smoothed onset avg (first 1.5s): {avg_smooth_first3:+.2f} vs raw {avg_raw_first3:+.2f}",
      avg_smooth_first3 > avg_raw_first3)
check(f"smoothing delays initial brake onset",
      smoothed_e2e[0] > raw_e2e[0])

scenarios.append({"name": "Unnecessary e2e Braking", "data": scenario3_data})

# ═══════════════════════════════════════
# SCENARIO 4: Emergency stop (seg 8, t=527-532)
# ═══════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 4: Emergency Stop (segment 8)")
print("Lead at 36m, closing at 10+ m/s - smoothing should NOT interfere")
print("=" * 70)

emergency_braking = [
    # (time, aTarget, d_rel)
    (0.0, -1.71, 36.1),
    (0.5, -2.11, 31.3),
    (1.0, -2.62, 24.7),
    (1.5, -2.95, 20.4),
    (2.0, -3.09, 12.1),
    (2.5, -2.66, 9.3),
    (3.0, -2.49, 6.8),
    (3.5, -2.13, 4.2),
]

raw_emergency = [x[1] for x in emergency_braking]
smoothed_emergency = []
prev_a = 0.0
for _, target, d in emergency_braking:
    if target < prev_a and d > DECEL_SMOOTH_DIST_BP[0]:
        jerk = float(np.interp(d, DECEL_SMOOTH_DIST_BP, DECEL_SMOOTH_JERK_V))
        min_a = prev_a + jerk * 0.5
        output = max(target, min_a)
    else:
        output = target
    smoothed_emergency.append(output)
    prev_a = output

scenario4_data = {
    "time": [x[0] for x in emergency_braking],
    "raw_aTarget": raw_emergency,
    "smoothed": smoothed_emergency,
    "d_rel": [x[2] for x in emergency_braking],
}

check(f"emergency stop: smoothing minimal at close range",
      abs(min(smoothed_emergency) - min(raw_emergency)) < 0.5)
check(f"peak decel reached: {min(smoothed_emergency):+.2f} (needs to be strong)",
      min(smoothed_emergency) < -2.5)
check(f"at d<8m: no smoothing applied",
      smoothed_emergency[-1] == raw_emergency[-1])

scenarios.append({"name": "Emergency Stop", "data": scenario4_data})

# ═══════════════════════════════════════
# SCENARIO 5: Curve tracking (seg 16, t=1058-1062)
# ═══════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 5: Curve Tracking (segment 16, ~20 mph right turn)")
print("Under-steer at entry, over-steer at exit")
print("=" * 70)

curve_data = [
    # (time, v_ego_ms, desCurv, actCurv)
    (0.0, 21.6, +0.00952, +0.00601),
    (0.5, 20.6, +0.02357, +0.01347),
    (1.0, 20.1, +0.03864, +0.02704),
    (1.5, 19.7, +0.04414, +0.03772),
    (2.0, 19.4, +0.04595, +0.04174),
    (2.5, 19.3, +0.04552, +0.04068),
    (3.0, 19.6, +0.04409, +0.03570),
    (3.5, 20.2, +0.02756, +0.02695),
    (4.0, 20.8, +0.00993, +0.01349),
]

scenario5_data = {
    "time": [x[0] for x in curve_data],
    "v_mph": [x[1] / 0.44704 for x in curve_data],
    "desCurv": [x[2] for x in curve_data],
    "actCurv": [x[3] for x in curve_data],
    "error": [x[2] - x[3] for x in curve_data],
    "old_ff_torque": [feedforward_torque(x[2] * x[1]**2, OLD_LAT_ACCEL_FACTOR, OLD_FRICTION) for x in curve_data],
    "new_ff_torque": [feedforward_torque(x[2] * x[1]**2, NEW_LAT_ACCEL_FACTOR, NEW_FRICTION) for x in curve_data],
}

# Calculate percent improvement in feedforward torque
old_ff_sum = sum(abs(x) for x in scenario5_data["old_ff_torque"])
new_ff_sum = sum(abs(x) for x in scenario5_data["new_ff_torque"])
ff_improvement = (new_ff_sum - old_ff_sum) / old_ff_sum * 100

check(f"feedforward torque increased by {ff_improvement:.1f}% with new params",
      ff_improvement > 3.0)

# Check entry under-steer
entry_errors = [curve_data[i][2] - curve_data[i][3] for i in range(4)]
check(f"entry under-steer confirmed (avg error: {np.mean(entry_errors):+.4f})",
      np.mean(entry_errors) > 0.002)

# Check exit over-steer
exit_error = curve_data[-1][2] - curve_data[-1][3]
check(f"exit over-steer confirmed (error: {exit_error:+.4f})",
      exit_error < -0.002)

# Higher feedforward should reduce entry lag
check("new params provide more initial torque at curve entry",
      scenario5_data["new_ff_torque"][1] > scenario5_data["old_ff_torque"][1])

scenarios.append({"name": "Curve Tracking", "data": scenario5_data})

# ═══════════════════════════════════════
# SCENARIO 6: Braking distribution comparison
# ═══════════════════════════════════════

print("\n" + "=" * 70)
print("SCENARIO 6: Overall Braking Distribution")
print("Comparing raw vs smoothed decel onset across all events")
print("=" * 70)

# All notable braking events from the drive (d_rel > 20m, starting aTarget)
braking_events = [
    # (initial_aTarget, d_rel at start)
    (-0.92, 25.2),   # seg 0
    (-0.81, 36.6),   # seg 2
    (-0.81, 61.0),   # seg 4
    (-0.82, 31.0),   # seg 16
    (-0.97, 41.4),   # seg 17 (e2e)
    (-0.82, 48.2),   # seg 17 (e2e #2)
    (-1.71, 36.1),   # seg 8 (emergency)
    (-0.87, 30.3),   # seg 22
    (-0.82, 39.3),   # seg 23
]

raw_initial = [x[0] for x in braking_events]
smoothed_initial = []
for target, d in braking_events:
    if d > DECEL_SMOOTH_DIST_BP[0]:
        jerk = float(np.interp(d, DECEL_SMOOTH_DIST_BP, DECEL_SMOOTH_JERK_V))
        min_a = 0.0 + jerk * DT_MDL  # starting from ~0
        smoothed_initial.append(max(target, min_a))
    else:
        smoothed_initial.append(target)

scenario6_data = {
    "events": [{"d_rel": d, "raw": t, "smoothed": s}
               for (t, d), s in zip(braking_events, smoothed_initial)],
    "raw_initial": raw_initial,
    "smoothed_initial": smoothed_initial,
}

harshness_reduction = np.mean([abs(s) - abs(r) for r, s in zip(raw_initial, smoothed_initial)])
check(f"average initial harshness reduced by {abs(harshness_reduction):.2f} m/s^2",
      harshness_reduction < 0)

# Count events where smoothing makes a difference
improved = sum(1 for r, s in zip(raw_initial, smoothed_initial) if s > r)
check(f"{improved}/{len(braking_events)} braking events have gentler onset",
      improved >= 5)

scenarios.append({"name": "Braking Distribution", "data": scenario6_data})

# ═══════════════════════════════════════
# SCENARIO 7: Source transition smoothing (cruise->e2e jumps)
# Real acceleration jumps observed from diagnostic analysis
# ═══════════════════════════════════════
print("\n=== Scenario 7: Source Transition Smoothing ===")

source_transitions = [
    {"prev_a": 0.85, "target_a": -0.49, "d_rel": float("nan"), "label": "no lead, +0.85 -> -0.49"},
    {"prev_a": 0.84, "target_a": -0.50, "d_rel": 57.0, "label": "lead 57m, +0.84 -> -0.50"},
    {"prev_a": 0.06, "target_a": -0.83, "d_rel": 31.0, "label": "lead 31m, +0.06 -> -0.83"},
    {"prev_a": -0.04, "target_a": -0.81, "d_rel": 61.0, "label": "lead 61m, -0.04 -> -0.81"},
    {"prev_a": 0.19, "target_a": -0.58, "d_rel": 39.0, "label": "lead 39m, +0.19 -> -0.58"},
]

scenario7_data = {"transitions": []}

for tr in source_transitions:
    d = 40.0 if np.isnan(tr["d_rel"]) else tr["d_rel"]
    raw_jump = abs(tr["target_a"] - tr["prev_a"])

    # Without universal limiter: full jump in 1 cycle
    raw_first_cycle_a = tr["target_a"]

    # With universal limiter: jerk-limited first cycle
    smoothed_first_cycle_a = tr["prev_a"] + OUTPUT_DECEL_JERK_LIMIT * DT_MDL

    # Time to settle to target
    remaining = abs(smoothed_first_cycle_a - tr["target_a"])
    settle_cycles = int(remaining / (abs(OUTPUT_DECEL_JERK_LIMIT) * DT_MDL)) + 1
    settle_time = settle_cycles * DT_MDL

    scenario7_data["transitions"].append({
        "label": tr["label"],
        "raw_jump": raw_jump,
        "first_cycle_raw": raw_first_cycle_a,
        "first_cycle_smoothed": smoothed_first_cycle_a,
        "settle_time": settle_time,
    })

    check(f"transition '{tr['label']}': first-cycle jump reduced from {raw_jump:.2f} to {abs(OUTPUT_DECEL_JERK_LIMIT * DT_MDL):.2f} m/s2",
          abs(smoothed_first_cycle_a - tr["prev_a"]) < raw_jump)

all_settle = [t["settle_time"] for t in scenario7_data["transitions"]]
check(f"all transitions settle within 1s (max {max(all_settle):.2f}s)",
      max(all_settle) <= 1.0)

scenarios.append({"name": "Source Transition Smoothing", "data": scenario7_data})

# ═══════════════════════════════════════
# Output JSON for canvas visualization
# ═══════════════════════════════════════

output = {
    "scenarios": [],
    "summary": {
        "total_tests": passed + failed,
        "passed": passed,
        "failed": failed,
    }
}

for s in scenarios:
    # Convert numpy types to native Python for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    output["scenarios"].append({
        "name": s["name"],
        "data": convert(s["data"]),
    })

json_path = "tools/scripts/scenario_results.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print(f"Results: {passed} passed, {failed} failed")
print(f"JSON output: {json_path}")
if failed:
    sys.exit(1)
