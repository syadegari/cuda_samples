#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p cute_transpose/build cute_transpose/reports
cutlass_dir=${CUTLASS_PATH:-/opt/cutlass}
matrix_m=${TRANSPOSE_M:-4096}
matrix_n=${TRANSPOSE_N:-8192}

nvcc -std=c++17 -O3 -lineinfo -arch=native \
  -I "$cutlass_dir/include" -I "$cutlass_dir/tools/util/include" \
  cute_transpose/main.cu -o cute_transpose/build/transpose

if [ "$#" -eq 0 ]; then set -- 0 1 2 3 4 5 pad; fi
for layout in "$@"; do
    bash ./ncu_sudo --section LaunchStats --section SpeedOfLight \
         --section SpeedOfLight_RooflineChart \
         --section SpeedOfLight_HierarchicalSingleRooflineChart \
         --section MemoryWorkloadAnalysis_Chart \
         --section MemoryWorkloadAnalysis_Tables --section SourceCounters \
         --import-source yes --force-overwrite \
         --export "cute_transpose/reports/${matrix_m}x${matrix_n}_${layout}" \
         ./cute_transpose/build/transpose "$layout" "$matrix_m" "$matrix_n"
    # ncu --import "cute_transpose/reports/${matrix_m}x${matrix_n}_${layout}.ncu-rep" \
        #   --page raw --csv --print-units base > "cute_transpose/reports/${matrix_m}x${matrix_n}_${layout}.csv"
done
