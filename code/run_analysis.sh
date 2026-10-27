#!/bin/bash
cd "$(dirname "$0")"
python3 e1_matched.py v3 >> e1_v3.log 2>&1
python3 confirm_fit.py traces_confirm results_confirm --loop >> confirm_fit.log 2>&1
python3 e1_matched.py v2 >> e1_v2.log 2>&1
