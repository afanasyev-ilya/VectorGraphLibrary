#!/bin/bash

arch="sx"
device="1"

if [ $# -eq 1 ]
  then
    arch=$1
fi

file_name="./"$arch"_performance_stats.csv"
rm $file_name

./testing_scripts/benchmark_specific_app.sh "./bfs_"$arch "-td -it 10 -device "$device $arch
./testing_scripts/benchmark_specific_app.sh "./sssp_"$arch "-all-active -push -it 10 -device "$device $arch
./testing_scripts/benchmark_specific_app.sh "./sssp_"$arch "-all-active -pull -it 10 -device "$device $arch
./testing_scripts/benchmark_specific_app.sh "./sssp_"$arch "-partial-active -it 10 -device "$device $arch
./testing_scripts/benchmark_specific_app.sh "./pr_"$arch "-it 10 -device "$device $arch
./testing_scripts/benchmark_specific_app.sh "./cc_"$arch " -device "$device $arch