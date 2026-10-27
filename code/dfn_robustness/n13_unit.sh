#!/bin/bash
set -euo pipefail
TAG="$1"; PY="${2:-python3}"; SEED=9202610010
COND="${TAG%%_p*}"; REST="${TAG#*_p}"; PACK=$((10#${REST%%_c*})); CELL="${TAG##*_c}"
ORIENT=1; [ "$CELL" = "1" ] && ORIENT=-1
OUT="n13out/orig"; TR="$OUT/traces"; mkdir -p "$TR"
export N6_SPREAD=orig TRACES="$TR" NEWOUT="$OUT"
filt() { grep -v "Warn\|return super\|TIMING" || true; }
"$PY" -B sim_seq.py $SEED "$COND" $PACK $CELL val val "$TR" 2>&1 | filt
"$PY" -B sim_seq.py $SEED "$COND" $PACK $CELL seqs/s4786.npy s4786 "$TR" $ORIENT 2>&1 | filt
"$PY" -B ref_impedance.py $SEED "$COND" $PACK $CELL "$TR" 2>&1 | filt
for f in "$TR/$TAG.val.npz" "$TR/$TAG.s4786.npz" "$TR/$TAG.zref.json"; do [ -s "$f" ] || { echo "MISSING $f"; exit 3; }; done
"$PY" -B n6_fit.py "$TAG"
"$PY" -B n1_ecm.py nodrift "$TAG"
"$PY" -B n2_ms.py e2 "$TAG"
echo "UNIT_DONE n13 $TAG"
