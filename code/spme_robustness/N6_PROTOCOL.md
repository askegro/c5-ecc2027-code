# N6: literature cell spreads (protocol, fixed before any N6 result)

## Question
Do the results of the SPMe study (Table II, impedance errors) change when the cell-to-cell spreads of the paper are replaced by
values measured on fresh cells of one batch?

## Finding that shaped the design (verified before this protocol)
The truth backend of the paper (`campaign_v4_backend_ocv.py`) sets a per-unit "Contact resistance [Ohm]", but its SPMe model is
built without the PyBaMM option `"contact resistance": "true"`, so the value has no effect on the voltage (checked with PyBaMM 26.10:
1 mOhm changes the voltage at 20 A by 0.000 mV without the option and by 20.000 mV with it). The only cell spreads that act in the
paper's study are the electrode-count factor (capacity 2 % shared by the pack and 2 % per unit; it also scales every resistance by
its inverse) and the per-unit SOH spread (standard deviation 0.012).

## Arms (spread_arms.py)
- `orig`: the paper's spreads, reproduced exactly (used only for the check below).
- `lit`: Rumpf et al., J. Energy Storage 14 (2017) 224-243 (1100 fresh LFP/graphite cells, two batches): relative variation of
  capacity 0.28 % and of impedance 0.72 %. Electrode-count factor 0.28 % per unit, no pack-shared factor, no SOH spread; an active
  series (contact) resistance per unit with mean 0.1 mOhm and standard deviation 0.0190 mOhm, so that the spread of the real
  impedance at 0.008 Hz of the nominal cell (2.751 mOhm + 0.1 mOhm) is 0.72 %. The mean keeps the resistance positive and adds
  0.1 mOhm to every unit. The literature cells are LFP; the cell model is NMC (Chen2020). Primary arm.
- `zero`: no capacity, resistance or SOH spread. Optional bracket.
SOC (standard deviation 0.012) and temperature (1.5 C) of the units keep their spreads in every arm: they are operating states.

## Design
Same seed base (9202610010), conditions, packs (12 per condition), units (2 per pack), sequence (s4786), validation profile, noise,
pack current and operating states as the paper. The spread draws are standard normals taken in the same order and number as in
the original backend, so every arm consumes the same random stream and the comparison with the paper is paired by unit.
Known OCV (arm C1) only, as in the paper.

## Estimators (unchanged code)
CGE (`evallib.eval_new`), conventional pipeline (`evallib.eval_frozen`, seed + 50,000 + cell), ECM fit without drift or initial
states (`n1_ecm.py nodrift`), DRT baseline (`n2_ms.py e2`, primary key cge|1|C1), linear variant (CGE without static basis
functions), complex impedance at 0.008, 0.05, 1 Hz against the time-domain reference (`ref_impedance.py`, drop_bulk).

## Analysis (n6_analyze.py)
For each arm, as in the paper: unit medians; pack-level paired differences CGE minus each estimator (median, 95 % percentile
bootstrap with 10^4 resamples and seed 20261001, exact two-sided sign test, Holm over the four conditions); agreement of the three
linear estimators (largest absolute median pair difference); impedance errors |Z - Zref| / |Zref|. Paired with the paper's stored
results (same packs): arm minus paper per estimator, and the change of the CGE gain (X - CGE).

## Reading, fixed in advance
The paper's conclusions hold under `lit` if the CGE has the lowest error in at least 11 of 12 packs against each estimator in every
condition (Holm-adjusted p at most 0.025) and its impedance error stays below that of the four other estimators. Any change of the
medians is reported as it comes out.

## Check before the runs
`bash submit_n6.sh check` runs arm `orig` on unit nominal_p00_c0 and must reproduce the stored traces and fits of the paper
(ORIG_CHECK PASSED). Done in the cloud workspace with PyBaMM 26.10: traces bit for bit, fits to 1e-14 mV.

## Amendment 1 (14 Oct 2026, after the check on Vera, before any run of arm lit or zero)
The check on Vera (PyBaMM 26.9) reproduced the CGE, the linear variant, the DRT baseline and the reference impedance of the stored
unit to 12 digits, but the conventional pipeline gave 2.02187191202 mV against 2.02184972475 mV stored (difference 2.2e-5 mV) and
the ECM fit 1.15693054186 mV against 1.15695992267 mV (2.9e-5 mV). Both are nonlinear least-squares fits that iterate from starting
points, and their last digits depend on the machine: the stored linear-variant file of the paper (posthoc_linear_rows.json), made
in a separate run, holds 2.0218719120153335 mV for the same conventional fit, the Vera value. The check therefore requires equality
to 1e-9 for the records, the convex fits and the reference impedance, and agreement to 1e-3 mV for the two nonlinear fits.
Differences of this size (below 1e-4 mV) are reported with the paired comparison and are not read as effects of the spreads.

## Amendment 2 (16 Oct 2026, after the runs of arms lit, zero, dl and of N8)
n6_unit.sh and n8_unit.sh simulated the identification record of every unit with orientation +1. The paper uses +1 for unit 0
and -1 for unit 1 of each pair (confirm.list). The units 0 of all arms are correct; the units 1 of arms lit, zero, dl and of N8
were therefore not the paper's units 1, and the two units of a pair saw the same, not complementary, currents. The check run
only unit nominal_p00_c0 and did not catch it. Fix: the orientation follows the unit (n6_unit.sh, n8_unit.sh), n6_fit.py stops if
a record has the wrong orientation, and the check now also reproduces unit cold_p03_c1 of the paper (records bit for bit, convex
fits to 1e-12, nonlinear fits to 5e-6 mV in the cloud workspace). The units 1 are rerun; the earlier files are kept in
wrong_orientation_c1/ on Vera. Nothing else changes.
