#!/bin/sh

#mpirun -v -np 1 -ve 0 ./sssp_mpi_sx -load ./input_graphs/rmat_21_32.vgraph -push -all-active
#mpirun -v -np 2 -ve 0-1 ./sssp_mpi_sx -load ./input_graphs/rmat_21_32.vgraph -push -all-active

args=("$@")
number_of_arguments=$#

program_name=${args[0]}

for (( c=1; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

echo test : ${program_name}
echo args : ${program_args[@]}


mpirun -np 1 -ve 0 ${program_name} ${program_args[@]}
printf "\n\n\n"
mpirun -np 2 -ve 0-1 ${program_name} ${program_args[@]}