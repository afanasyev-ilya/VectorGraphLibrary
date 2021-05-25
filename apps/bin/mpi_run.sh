#!/bin/sh

#mpirun -v -np 1 -ve 0 ./sssp_mpi_sx -load ./input_graphs/rmat_21_32.vgraph -push -all-active
#mpirun -v -np 2 -ve 0-1 ./sssp_mpi_sx -load ./input_graphs/rmat_21_32.vgraph -push -all-active

mpirun -np 1 -ve 0 ./sssp_mpi_sx -load ./input_graphs/rmat_24_32.vgraph -push -all-active -check
printf "\n\n\n"
mpirun -np 2 -ve 0-1 ./sssp_mpi_sx -load ./input_graphs/rmat_24_32.vgraph -push -all-active -check