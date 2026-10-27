# N7: electrode double layer (protocol, fixed before any N7 result; 16 Oct 2026)

## Question
The static term of the CGE assumes that the kinetic overpotential responds instantly to the current, as in the SPMe of the paper
(PyBaMM option "surface form": "false"). With the electrode double layer ("surface form": "differential", Chen2020 double-layer
capacity 0.2 F/m^2 in both electrodes), the kinetic overpotential lags the current with a time constant near 15 ms. Do the
results of the SPMe study change?

## Arm dl (spread_arms.py)
The spreads of the paper (arm orig) and the double layer in the time-domain cell model, for the records, the validation
profile and the reference impedance. Everything else as N6: seed base 9202610010, same packs, units, sequence s4786, noise,
pack current and operating states, known OCV, the same estimators and analysis (n6_analyze.py), paired by unit with the paper.

## Reading, fixed in advance
The paper's conclusion holds with the double layer if the CGE has the lowest error in at least 11 of 12 pairs against each
estimator in every condition (Holm-adjusted p at most 0.025) and its impedance error stays below that of the four other
estimators. The change of every median against the paper is reported as it comes out, whichever direction.

## Check before the runs
One unit (nominal_p00_c0) ran end to end in the cloud workspace (PyBaMM 26.10), about 5 min: CGE 0.373 mV (paper 0.345),
linear variant 1.166 (1.171), conventional pipeline 6.258 (2.022), DRT baseline 1.473 (1.162). This unit is not a result.
