#!/bin/sh

#mpirun -v -np 1 -ve 0 ./sssp_mpi_sx -load ./input_graphs/rmat_21_32.vgraph -push -all-active
#mpirun -v -np 2 -ve 0-1 ./sssp_mpi_sx -load ./input_graphs/rmat_21_32.vgraph -push -all-active

args=("$@")
number_of_arguments=$#

mpi_proc_num=${args[0]}
program_name=${args[1]}

for (( c=2; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

mpirun -np $mpi_proc_num -ve 0-$mpi_proc_num ${program_name} ${program_args[@]}