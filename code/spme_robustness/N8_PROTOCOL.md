# N8: long WLTC validation (protocol, fixed before any N8 result except the pilot unit below; 16 Oct 2026)

## Question
The 180 s validation profile of the paper is shorter than the slowest branches of the grid and moves the SOC by less than 0.8 %.
Do the results hold on a long, standard drive cycle from the same operating point?

## Design
Arm orig (the paper's spreads), seed base 9202610010, the same 96 units, the same identification records (sequence s4786,
reproduced exactly by arm orig), the same estimators and settings, known OCV. Only the validation changes:
- WLTC.csv of PyBaMM (pybamm-data v1.0.2, md5 ffac0af26fb9ad10918aa94686859247; power in kW at 0.1 s for 1800 s), taken at its
  1 Hz points, used as a current profile proportional to that power, scaled to a peak discharge of 28.6 A (the bound of the paper's
  profile), held for 1 s, sampled at 50 ms, followed by 600 s of rest: 2400 s, 48,000 samples, from rest at the unit's operating
  point. Peak regeneration -15.7 A; mean over the drive 4.05 A; SOC change about 3.4 %.
- Current noise 0.02 A (own random stream); SOC estimates for the deployment arm as evallib (own stream for the validation one).
- The SPMe is solved with the backend's solver and settings in pieces of at most 2000 samples (sim_wltc.simulate_light), because
  the backend's single call exhausts the memory on this profile. On the first 6000 samples the pieces differ from the single call
  by at most 9.2 microvolts.
Scores: RMSE over the whole profile (n6_analyze.py, all estimators); RMSE of the CGE and the linear variant over the drive and
the rest part separately (n8_summary.py).

## Reading, fixed in advance
Reported as it comes out. The paper's ranking holds on the long validation if the CGE has the lowest error in at least 11 of 12
pairs against each estimator in every condition (Holm-adjusted p at most 0.025).

## Pilot (cloud workspace, PyBaMM 26.10, unit nominal_p00_c0, not a result)
CGE 3.04 mV (drive 1.83, rest 5.19), linear variant 3.68 (2.18, 6.32), conventional pipeline 4.66, DRT baseline 3.57.
About 2 min per unit.
