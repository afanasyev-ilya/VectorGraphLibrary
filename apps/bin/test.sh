#!/bin/bash
#PBS -q ve10b
#PBS -l elapstim_req=00:30:00
#PBS -T necmpi
#PBS --venode=4
#PBS --venum-lhost=4
#PBS --use-hca=mpi:2

cd $PBS_O_WORKDIR

source /opt/nec/ve/mpi/2.17.0/bin/necmpivars.sh
source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh

exe=./sssp_sx_mpi
args=(-load ./input_graphs/rmat_25_32.vgraph -pull -all-active -it 10)

mpirun -np 4 -ve 0-3 ${exe} "${args[@]}"