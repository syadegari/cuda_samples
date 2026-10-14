#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p soa_vs_aos/build soa_vs_aos/reports
nvcc -O3 -lineinfo -arch=native soa_vs_aos/main.cu -o soa_vs_aos/build/update
n=${RECORDS:-8388608}
launches=${LAUNCHES:-20}
if [ "$#" -eq 0 ]; then set -- aos soa; fi
for layout in "$@"; do
  report="soa_vs_aos/reports/${n}_${layout}"
  bash ./nsys_sudo profile --trace cuda --sample none --cpuctxsw none \
    --force-overwrite true --output "${report}_timing" \
    ./soa_vs_aos/build/update "$layout" "$n" "$launches"
  nsys stats --force-export true --report cuda_gpu_kern_sum --format table "${report}_timing.nsys-rep" \
    | tee "${report}_timing.txt"
  bash ./ncu_sudo --section LaunchStats --section SpeedOfLight \
    --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables \
    --section SourceCounters --import-source yes --force-overwrite \
    --export "$report" ./soa_vs_aos/build/update "$layout" "$n"
done
