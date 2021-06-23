#!/bin/sh

args=("$@")
number_of_arguments=$#

scale_limit=${args[0]}
program_name=${args[1]}

for (( c=2; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

for (( num_proc=1; num_proc<=${scale_limit}; num_proc++ ))
do
   ve_limit=$(($num_proc-1))
   sbatch -p ssd --exclusive --gres=ve:${ve_limit} ./mpi_run.sh ${num_proc} ${program_name} ${program_args[@]}
done