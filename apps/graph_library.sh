#!/bin/bash

args=("$@")
number_of_arguments=$#

program_name=${args[0]}

for (( c=1; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

#export OMP_NUM_THREADS=64
#export CILK_NWORKERS=28
#export KMP_AFFINITY=scatter

echo common launch of application ${program_name}

#free -m

#source /opt/nec/ve/nlc/2.1.0/bin/nlcvars.sh
#source /opt/nec/ve/nlc/2.2.0/bin/nlcvars.csh
#source ./test.csh

if [[ $program_name == *_mpi ]]
then
  mpirun -np 1 -ve 0-1 ${program_name} ${program_args[@]}
  echo "----------------------------------------------------------------------------"
  mpirun -np 2 -ve 0-1 ${program_name} ${program_args[@]}
else
  ${program_name} ${program_args[@]}
fi



#free -m
