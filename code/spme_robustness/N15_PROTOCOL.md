# N15: confirmatory test of the grid that ends at the record length (protocol, fixed before any N15 result; 18 Oct 2026)

## Question
In N9 and N10, run on the units of the paper, the CGE with its grid ending at 265 s had the lowest error on the 2400 s drive cycle
with rest in every pair. That grid end was chosen from five values after the results of the paper's grid (2000 s) were known, so the
paper labels it exploratory. Does it hold on new units, with the rule fixed in advance?

## Rule under test
The grid of time constants ends at the length of the identification record: tau_max = 265 s (record 265.23 s), 8 per decade from
5 ms. Only this value is tested.

## Design
New seed base 9302610010 (the paper: 9202610010; the derived seeds do not overlap), otherwise as N8: the paper's SPMe and cell spreads
(arm orig), four conditions, 12 packs each, 2 units per pack with orientations +1 and -1, sequence s4786, the long WLTC validation of
N8 (2400 s, drive and 600 s of rest), known OCV. Per unit (n15_unit.sh): sim_wltc.py, sim_seq.py, ref_impedance.py, n8_fit_all.py
(conventional pipeline, ECM fit, DRT baseline, linear variant and CGE on the paper's grid; unchanged code), n15_fit265.py (CGE, linear
variant and DRT baseline on the 265 s grid; the code of n9_taumax.py and n10_drt.py for one unit).

## Analysis and criterion, fixed in advance (n15_analyze.py)
Comparators: the ECM fit and the conventional pipeline (as fitted in the paper), and the DRT baseline and the linear variant on the same
265 s grid. Pairs = mean of the two units of a pack; exact two-sided sign test; Holm over the four conditions for each comparator.
The rule is confirmed if, in every condition and against every comparator, the CGE on the 265 s grid has the lower WLTC error in at
least 11 of 12 pairs with a Holm-adjusted p-value of at most 0.025. Otherwise it is not confirmed, and the paper says so. The medians of
all estimators, the CGE on the 2000 s grid, and the drive and rest parts are reported as they come out.

## Check before the runs
`bash submit_n15.sh check` runs unit cold_p00_c0 on Vera and `n15_check.py` compares it with the same unit run in the cloud workspace
(PyBaMM 26.10.1, `chk_n15/`) with the tolerances of Amendment 1 of N11 (same solver settings): currents and OCV within 1e-6; truth and
measured voltages within 0.5 mV at most and 0.02 mV RMS; the RMSE of every estimator within 0.01 mV. `submit_n15.sh run` refuses to
start before N15_CHECK PASSED. A failure is examined and recorded as an amendment before any other unit is run.

## The check unit (seen before the runs; it counts as one of the 96 units)
cold_p00_c0, cloud workspace: 123 s in total. WLTC RMSE (mV): CGE on the 265 s grid 0.947, CGE on the paper's grid 2.992, linear
variant on the 265 s grid 5.891, DRT baseline on the 265 s grid 5.899, ECM fit and conventional pipeline as in chk_n15/. One unit is
not a result and does not change the design.
