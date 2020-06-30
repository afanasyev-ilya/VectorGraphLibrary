#!/bin/bash

args=("$@")
number_of_arguments=$#

program_name=${args[0]}

for (( c=1; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

echo test : ${program_name}
echo args : ${program_args[@]}

#export OMP_NUM_THREADS=64
#export CILK_NWORKERS=28
#export KMP_AFFINITY=scatter

echo common launch of application ${program_name}

#free -m

${program_name} ${program_args[@]}

#free -m
