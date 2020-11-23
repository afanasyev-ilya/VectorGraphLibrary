#!/bin/bash

file_name="./nec_stats.csv"
rm $file_name

./testing_scripts/benchmark_specific_app.sh "./bfs_sx" "-td -it 10"
./testing_scripts/benchmark_specific_app.sh "./sssp_sx" "-all-active -push -it 10"
./testing_scripts/benchmark_specific_app.sh "./sssp_sx" "-all-active -pull -it 10"
./testing_scripts/benchmark_specific_app.sh "./sssp_sx" "-partial-active -it 10"
./testing_scripts/benchmark_specific_app.sh "./cc_sx" ""
./testing_scripts/benchmark_specific_app.sh "./pr_sx" "-it 1"