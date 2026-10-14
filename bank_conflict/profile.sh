#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p bank_conflict/build bank_conflict/reports

nvcc -O3 -lineinfo -arch=native bank_conflict/main.cu -o bank_conflict/build/bank_read

if [ "$#" -eq 0 ]; then set -- 1 2 4 8 16 32 33 0; fi
for stride in "$@"; do
    bash ./ncu_sudo --section LaunchStats --section SpeedOfLight \
         --section SpeedOfLight_RooflineChart \
         --section SpeedOfLight_HierarchicalSingleRooflineChart \
         --section MemoryWorkloadAnalysis_Chart \
         --section MemoryWorkloadAnalysis_Tables --section SourceCounters \
         --import-source yes --force-overwrite \
         --export "bank_conflict/reports/stride_${stride}" \
         ./bank_conflict/build/bank_read "$stride"
    # ncu --import "bank_conflict/reports/stride_${stride}.ncu-rep" --page raw --csv --print-units base \
        #   > "bank_conflict/reports/stride_${stride}.csv"
done
