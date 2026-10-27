# N12: DFN truth model with solver tolerances 1e-9 (protocol, fixed before any N12 result on Vera; 16 Oct 2026)

## Question
N11 ran the DFN with the default CasadiSolver tolerances (rtol = atol = 1e-6), as the SPMe study of the paper does. Amendment 1 of
N11 found that these tolerances leave up to 0.22 mV of solver error at single samples of the truth voltage (0.007 mV RMS against a
1e-9 solve) and that this error depends on the machine. Do the N11 results change when the solver error is removed?

## Design
The N11 package with one change: in `campaign_v4_backend_n12.py` the CasadiSolver of the time-domain simulation is built with
rtol = 1e-9 and atol = 1e-9. This solver produces the identification records, the validation profile and the reference impedance.
The C/20 OCV curve (IDAKLUSolver) is unchanged and is the cached curve of N11. Everything else as N11: DFN, arm orig, seed base
9202610010, the same 48 units of the cold and nominal conditions (jobs_n12.txt), orientations, sequence s4786, noise, pack current,
operating states, known OCV, the five estimators with unchanged code. Output in n12out/tight.

## Analysis
- `python n6_analyze.py --arm n12out/tight --e2 <results_confirm> --new <new> --lin <posthoc_linear_rows.json>`: as N11.
- `python n12_compare.py <N11 results>/n11out/orig n12out/tight`: for every estimator and condition, the unit median RMSE in N11 and
  N12 and its change, the median pack difference N12 - N11 with the number of packs lower and higher, and the largest unit difference.

## Reading, fixed in advance
(a) The N11 criterion holds at 1e-9: in both conditions the CGE has the lowest validation RMSE in at least 11 of 12 pairs against
each of the four other estimators, and its median impedance error is below that of each of them at 0.008, 0.05 and 1 Hz.
(b) The median RMSE of every estimator in each condition changes by at most 0.01 mV between N11 and N12 (the last digit the paper
reports).
If (a) and (b) hold, the N11 result does not depend on the solver tolerance. Otherwise the paper reports the 1e-9 result and the
change. Every change is reported as it comes out.

## Check before the runs
`bash submit_n12.sh check` runs unit cold_p00_c0 on Vera and `n12_check.py` compares it with the same unit run in the cloud workspace
with PyBaMM 26.10.1 (`chk_tight/`). `submit_n12.sh run` refuses to start before DFN_CHECK PASSED. Tolerances, fixed from the same unit
run in the workspace with PyBaMM 26.9 and 26.10.1 at 1e-9 (observed differences in brackets): currents and OCV within 1e-6 (below 1e-8 V);
truth and measured voltages within 0.005 mV at most and 0.001 mV RMS (0.0003 mV and 0.00002 mV); reference impedance within 1e-5 of
|Zref| (1.7e-7; the paper reports impedance errors to 0.1 % of |Zref|); CGE, linear variant and DRT baseline within 1e-4 mV (1.1e-6 mV); conventional pipeline and ECM fit within 0.01 mV
(0.0003 mV; these nonlinear fits remain machine-dependent: with 26.9 the ECM fit selected order 3, with 26.10.1 order 4, RMSE
5.836123 against 5.836029 mV). A failure is examined and recorded as an amendment before any other unit is run.

## Timing and memory (cloud workspace, one core of an Intel Xeon at 2.10 GHz, PyBaMM 26.10.1; not results)
cold_p00_c0: validation profile 286 s, sequence s4786 938 s, reference impedance 86 s, all fits 46 s, total 1356 s (22.6 min, 2.1 times
N11), peak memory 1.69 GB. On Vera N11 took 0.65 times the workspace time, so about 15 min per unit and 12 core-hours for 48 units are
expected; jobs ask for 4 GB and 1.5 h.

## Observation from the test unit (not a result)
cold_p00_c0, PyBaMM 26.10.1, 1e-9 against the default tolerance: truth voltage on the sequence 0.115 mV at most and 0.009 mV RMS apart,
on the validation profile 0.22 mV and 0.007 mV; RMSE CGE 0.180472 (0.179990), linear variant 5.770220 (5.770152), conventional
pipeline 5.741324 (5.741543), ECM fit 5.836029 (5.836056), DRT baseline 5.765454 (5.765385) mV.
