#!/usr/bin/env python3
import sys
from openpilot.tools.lib.logreader import LogReader

path = sys.argv[1]
lr = LogReader(path, sort_by_time=True)
plans = []
for m in lr:
    if m.which() == "longitudinalPlan":
        t = m.logMonoTime / 1e9
        a = float(m.longitudinalPlan.aTarget)
        plans.append((t, a))
print("n plans", len(plans))
if plans:
    print("t range", plans[0][0], plans[-1][0])
for t, a in plans:
    if a <= -3.4:
        print(f"  emergency t={t:.2f} a={a:.2f}")
em = [p for p in plans if p[1] <= -3.4]
if em:
    print(f"suggest --t-lo {em[0][0]-5:.1f} --t-hi {em[-1][0]+2:.1f}")
