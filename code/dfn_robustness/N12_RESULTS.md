# N12 results (16 Oct 2026)

Check on Vera: DFN_CHECK PASSED (records within 3.1e-7 V; CGE, linear variant, DRT baseline within 9e-7 mV; conventional pipeline
2.2e-3 mV and ECM fit 9.6e-5 mV, within the 0.01 mV fixed in N12_PROTOCOL.md). 48 of 48 units; no failed fit; peak memory 1.53 GB.

Deviation from the protocol text (no effect on any number): the protocol names the output folder n12out/tight, but n6_analyze.py
requires the folder to be named after the spread arm recorded in the traces (orig) and stops otherwise. The analysis was run on a copy
of n12out/tight named orig: `python n6_analyze.py --arm <copy>/orig --e2 <results_confirm> --new <new> --lin <posthoc_linear_rows.json>`
(n12_analysis.txt, n12_contrast.json). n12_compare.py ran on n12out/tight as written (n12_compare.txt, n12_compare.json).

Reading (a), the N11 criterion at 1e-9: MET. In cold and nominal the CGE has the lowest RMSE in 12 of 12 pairs against each of the
four other estimators (largest Holm-adjusted p 0.002); median impedance error CGE 0.8-1.5 %, others 6.2-25.8 %, CGE lower in every pack
at 0.008, 0.05 and 1 Hz.
Reading (b), median changes against N11 at most 0.01 mV: MET. Largest change of a condition median 0.0008 mV (conventional pipeline,
nominal); CGE -0.0004 (cold) and +0.0002 mV (nominal); largest change of a single unit 0.011 mV (conventional pipeline).
The N11 result therefore does not depend on the solver tolerance. make_numbers_n11.py gives the same numbers_n11.tex from the N12
results as from the N11 results.
