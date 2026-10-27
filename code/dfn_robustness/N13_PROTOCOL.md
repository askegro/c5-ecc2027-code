# N13: DFN truth model, hot and aged conditions (protocol, fixed before any N13 result on Vera; 16 Oct 2026)

## Question
N11 found that the results of the SPMe study hold with the DFN as the truth model in the cold and nominal conditions. Do they also
hold in the hot and aged conditions, so that the DFN check covers all four conditions of the paper?

## Design
The N11 package unchanged except for the units: the same backend (`campaign_v4_backend_n11.py`, DFN, default mesh, default solver
tolerances, the cached OCV curve of N11), arm orig, seed base 9202610010, and the 48 units of the hot and aged conditions
(12 packs each, 2 units per pack, orientations +1 and -1; jobs_n13.txt). Known OCV, the five estimators with unchanged code.
Output in n13out/orig.

## Analysis
- `python n6_analyze.py --arm n13out/orig --e2 <results_confirm> --new <new> --lin <posthoc_linear_rows.json>`: as N11.
- All four conditions together: copy the per-unit files of N11 (n11out/orig) and N13 (n13out/orig) into one folder, e.g.
  dfn_all/orig/{fit,nodrift,ms}, and run n6_analyze.py on it. This is the analysis behind any statement on the four conditions; its
  Holm adjustment is over the four conditions, as in the paper.

## Reading, fixed in advance
The paper's conclusion holds on the DFN in the hot and aged conditions if, in both, the CGE has the lowest validation RMSE in at least
11 of 12 pairs against each of the four other estimators and its median impedance error is below that of each of them at 0.008,
0.05 and 1 Hz (the criterion of N11). If it holds here and in N11, the paper states the DFN result for all four conditions;
otherwise it states it for the conditions in which it holds and reports the others. The DFN minus SPMe differences are reported as
they come out.

## Check before the runs
`bash submit_n13.sh check` runs unit hot_p00_c0 on Vera and `n13_check.py` compares it with the same unit run in the cloud workspace
(PyBaMM 26.10.1, `chk_n13/`), with the tolerances of Amendment 1 of N11: currents and OCV within 1e-6; truth and measured voltages
within 0.5 mV at most and 0.02 mV RMS; reference impedance within 1e-6 relative; the RMSE of each of the five estimators within
0.01 mV. `submit_n13.sh run` refuses to start before DFN_CHECK PASSED. A failure is examined and recorded as an amendment before any
other unit is run.

## Timing and memory (cloud workspace, one core of an Intel Xeon at 2.10 GHz, PyBaMM 26.10.1; not results)
hot_p00_c0: 154 + 297 + 34 + 74 s = 559 s (9.3 min), peak 1.55 GB; aged_p00_c0: 174 + 344 + 48 + 67 s = 633 s (10.5 min), peak 1.62 GB
(validation profile, sequence, reference impedance, all fits). On Vera N11 took 0.65 times the workspace time: about 6.5 min per unit
and about 5 core-hours for 48 units are expected; jobs ask for 4 GB and 1.5 h.

## Observations from the two units above (not results; they do not change the design)
CGE / linear variant / conventional pipeline / ECM fit / DRT baseline, RMSE in mV, DFN (SPMe of the paper):
hot_p00_c0 0.222 (0.242) / 0.397 (0.406) / 1.711 (1.739) / 0.350 (0.355) / 0.387 (0.395);
aged_p00_c0 0.384 (0.407) / 1.623 (1.609) / 2.302 (2.262) / 1.500 (1.485) / 1.611 (1.598).
