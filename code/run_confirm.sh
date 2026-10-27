#!/bin/bash
cd "$(dirname "$0")"
for pass in 1 2 3; do
  while IFS=$'\t' read -r out cmd; do
    [ -f "$out" ] && continue
    bash -c "$cmd" >> confirm_sim.log 2>&1
  done < confirm.list
done
echo "SIMS_DONE $(date +%T)" >> confirm_sim.log
