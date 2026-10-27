# N9: longest time constant of the grid on the long WLTC validation (exploratory; 16 Oct 2026)

## Why exploratory
Proposed after the N8 result (the CGE has its largest error in the 600 s rest of the WLTC validation and loses its lead outside
the cold condition) and after the check of the grid on the 180 s profile (new_tests_ecc/taumax_check). The values below were
chosen before this run, but the idea was not fixed before N8, so the result is reported as exploratory, not as a change of the
estimator, whatever it shows.

## Design
The 96 units of N8 (arm orig, the paper's identification records, the WLTC validation of N8). The CGE and its linear variant
(no static basis functions) are refitted with the longest time constant of the grid at 50, 90, 265 (about the record length), 500
and 2000 s (the paper), 8 per decade from 5 ms, everything else unchanged, known OCV. Scores: RMSE over the drive (1800 s), the
rest (600 s) and the whole profile. The 2000 s values must equal those of N8 (checked in the cloud workspace on nominal_p00_c1:
CGE 0.6593 mV, linear variant 1.1489 mV, as on Vera).

## Analysis (n9_summary.py)
Per condition and grid end: median RMSE over units (all, drive, rest) for the CGE and the linear variant, and the number of pairs
in which the CGE has a lower error than the ECM fit, the DRT baseline and the linear variant of N8 (unchanged grid), with the
exact sign test, Holm over the four conditions.

## N10: the DRT baseline on the same grid ends (16 Oct 2026, after N9, before any N10 result)
The DRT baseline uses the grid of the CGE, so N9 compared the CGE on a 265 s grid with the DRT baseline on the 2000 s grid.
N10 refits the DRT baseline (ms_estimator.fit, lambda = 1, known OCV, as n2_ms.py arm C1) on the same five grid ends and scores it
on the WLTC validation (n10_drt.py -> n9out/drt_taumax.json). Check: at 2000 s it must equal N8 (cge|1|C1); confirmed in the cloud
workspace on nominal_p00_c1 (1.1439 mV). Reported, with N9, as exploratory: the CGE against the DRT baseline on the same grid end.
