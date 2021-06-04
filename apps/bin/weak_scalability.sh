#!/bin/sh

args=("$@")
number_of_arguments=$#

scale_limit=${args[0]}
program_name=${args[1]}

for (( c=2; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

scale=22
for num_proc in 1 2 4 8
do
    scale=$(( scale + 1 ))
    sbatch -p ssd --exclusive --gres=ve:${scale_limit} ./mpi_run.sh ${num_proc} ${program_name} "-load ./input_graphs/rmat_"${scale}"_32.vgraph" ${program_args[@]}
done