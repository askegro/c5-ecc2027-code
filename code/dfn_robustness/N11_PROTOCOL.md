# N11: DFN truth model, cold and nominal conditions (protocol, fixed before any N11 result; 16 Oct 2026)

## Question
The truth model of the paper is the SPMe. Do its results hold when the truth is the Doyle-Fuller-Newman model (DFN, PyBaMM
`lithium_ion.DFN`), which resolves the electrolyte and the particles across the electrode thickness?

## Design
`campaign_v4_backend_n11.py` is the N6 backend with `pybamm.lithium_ion.SPMe(` replaced by `pybamm.lithium_ion.DFN(` in its three
places: the C/20 OCV curve, the time-domain simulation, and `impedance()`. The OCV cache has its own folder. Nothing else changes:
default DFN mesh, isothermal, "surface form" off as in the paper, the same solver and dt_max regions, the spreads of the paper
(arm orig), seed base 9202610010, packs, units, orientations (+1 for unit 0, -1 for unit 1), sequence s4786, noise, pack current,
operating states, validation profile and reference impedance (time-domain sine on the same DFN). Known OCV (bulk OCV of the DFN
along the true trajectory), as in Table II.

Conditions: cold and nominal, 12 packs each, 2 units per pack (48 units, jobs_n11.txt). Cold has the strongest static
nonlinearity and the largest gain of the CGE; nominal is the reference point.

## Estimators (unchanged code)
CGE, conventional pipeline, linear variant (`n6_fit.py`), ECM fit without drift (`n1_ecm.py nodrift`), DRT baseline
(`n2_ms.py e2`, key cge|1|C1), impedance at 0.008, 0.05, 1 Hz.

## Analysis (n6_analyze.py, unchanged)
`python n6_analyze.py --arm n11out/orig --e2 <results_confirm> --new <new> --lin <posthoc_linear_rows.json>`: unit medians;
pack-level paired differences CGE minus each estimator (median, 95 % bootstrap interval, exact two-sided sign test, Holm over the
four conditions of the paper, the two conditions not run entering with p = 1); impedance errors; DFN minus SPMe for every estimator,
paired by unit with the stored results of the paper; change of the CGE gain.

## Reading, fixed in advance
The paper's conclusion holds on the DFN if, in both conditions, the CGE has the lowest validation RMSE in at least 11 of 12 pairs
against each of the four other estimators (Holm-adjusted p of the sign test then at most 0.025, as rounded in the paper) and its
median impedance error is below that of each of the four other estimators at 0.008, 0.05 and 1 Hz. Otherwise it does not hold on
the DFN, and the paper says so. The DFN minus SPMe differences are reported as they come out, whichever direction.

## Check before the runs
`bash submit_n11.sh check` runs unit cold_p00_c0 on Vera (PyBaMM 26.9) and `n11_check.py` compares it with the same unit run in the
cloud workspace (PyBaMM 26.10.1, `chk_dfn/`): records within 1e-6 V; CGE, linear variant, DRT baseline and reference impedance
within 1e-6 relative; conventional pipeline and ECM fit within 1e-3 mV (Amendment 1 of N6). `submit_n11.sh run` refuses to start
before DFN_CHECK PASSED. A failure is examined and recorded as an amendment before any other unit is run.

## Timing and memory (cloud workspace, one core of an Intel Xeon at 2.10 GHz, PyBaMM 26.10.1; not results)
| Unit | Validation profile | Sequence s4786 | Reference impedance | All fits | Total | Peak memory |
|---|---|---|---|---|---|---|
| cold_p00_c0 | 191 s | 364 s | 48 s | 42 s | 645 s (10.7 min) | 1.63 GB |
| nominal_p00_c0 | 182 s | 336 s | 44 s | 42 s | 604 s (10.1 min) | 1.70 GB |

48 units are about 8.3 core-hours; jobs ask for 4 GB and 1.5 h.

## Observations from the two units above (not results; they do not change the design)
CGE / linear variant / conventional pipeline / ECM fit / DRT baseline, RMSE in mV, DFN (SPMe of the paper):
cold_p00_c0 0.180 (0.191) / 5.770 (5.754) / 5.742 (5.733) / 5.836 (5.819) / 5.765 (5.749);
nominal_p00_c0 0.329 (0.345) / 1.176 (1.171) / 2.016 (2.022) / 1.164 (1.157) / 1.167 (1.162).
For nominal_p00_c0, the DFN truth voltage on the validation profile differs from the SPMe by 0.25 mV RMS (0.60 mV at most), and the
reference impedance at 0.008 Hz by 0.7 % (real part 3.013 against 3.035 mOhm).

## Amendment 1 (16 Oct 2026, after the check on Vera, before any other N11 unit)
The check on Vera (PyBaMM 26.9) gave the same currents, OCV and reference impedance as the cloud workspace (PyBaMM 26.10.1) to 1e-8 V
and 1e-7 relative, but truth voltages that differ by up to 0.16 mV (sequence) and 0.22 mV (validation profile), and fitted RMSE values
that differ by 0.0002 to 0.0007 mV (CGE, linear variant, DRT baseline) and 0.0002 to 0.0003 mV (conventional pipeline, ECM fit):
DFN_CHECK FAILED. Diagnosis in the cloud workspace, on the validation profile of the same unit:
- PyBaMM 26.9 installed in the workspace does not reproduce Vera either (CGE 0.180139 against 0.179340 mV on Vera and 0.179990 mV
  with 26.10.1), so the version is not the cause.
- With the CasadiSolver tolerances tightened from the default 1e-6 to 1e-9, PyBaMM 26.9 and 26.10.1 agree to 0.0003 mV. At the
  default tolerance, each solve is up to 0.19 mV (26.9) and 0.22 mV (26.10.1) from the tight solve at isolated samples, 0.007 mV RMS.
The differences are therefore solver error at the default tolerance, which the SPMe study of the paper also uses and whose machine-
dependent part is larger for the DFN (more states and an algebraic part). The solver settings stay as they are, so that the truth
model is the only change against the SPMe study. The check is amended: truth and measured voltages within 0.5 mV at most and
0.02 mV RMS; currents and OCV within 1e-6; reference impedance within 1e-6 relative; the RMSE of each of the five estimators within
0.01 mV, the last digit the paper reports. All 48 units run on Vera in one environment. The reading of the results is unchanged.
