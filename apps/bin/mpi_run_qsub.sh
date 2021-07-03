#!/bin/bash
#PBS -q ve10b
#PBS -l elapstim_req=00:30:00
#PBS -T necmpi
#PBS --venode=8
#PBS --cpunum-lhost=1
#PBS --venum-lhost=4
#PBS --use-hca=mpi:2

cd $PBS_O_WORKDIR

source /opt/nec/ve/mpi/2.17.0/bin/necmpivars.sh
source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh

exe=./sssp_sharded_sx_mpi
args=(-load ./sh_rmat_24_32.sharded_graph -pull -all-active -it 1)

scale_limit=8
rm -rf MPI_scale_perf.txt

for (( num_proc=1; num_proc<=${scale_limit}; num_proc = num_proc * 2 ))
do
   ve_limit=$(($num_proc-1))
   mpirun -np ${num_proc} -ve 0-${ve_limit} ${exe} "${args[@]}"
done