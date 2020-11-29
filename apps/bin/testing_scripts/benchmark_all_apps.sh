#!/bin/bash

file_name="./nec_stats.csv"
rm $file_name

prefix="sx"

if [ $# -eq 1 ]
  then
    prefix=$1
    file_name="./"$prefix"_performance_stats.csv"
fi

./testing_scripts/benchmark_specific_app.sh "./bfs_"$prefix "-td -it 10 -device 1"
./testing_scripts/benchmark_specific_app.sh "./sssp_"$prefix "-all-active -push -it 10 -device 1"
./testing_scripts/benchmark_specific_app.sh "./sssp_"$prefix "-all-active -pull -it 10 -device 1"
./testing_scripts/benchmark_specific_app.sh "./sssp_"$prefix "-partial-active -it 10 -device 1"
./testing_scripts/benchmark_specific_app.sh "./pr_"$prefix "-it 10 -device 1"
./testing_scripts/benchmark_specific_app.sh "./cc_"$prefix "-device 1"