#!/bin/bash
#PBS -q veany
#PBS -l elapstim_req=00:10:00
#PBS --venode=1
#PBS --venum-lhost=1

# Move to the working directory.
cd $PBS_O_WORKDIR

# Configure libraries requisite for VEs.
# Replace X.X.X to the target version in /opt/nec/ve/mpi directory.
source /opt/nec/ve/mpi/2.17.0/bin/necmpivars.sh
source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh

# Example of your work.
#./sssp_sx -s 20 -e 32 -format vcsr -check -it 3 -push -all-active
#./sssp_sx -s 22 -e 32 -format csr -check -it 4 -push -all-active

./bin/pr_sx -import ./bin/input_graphs/syn_rmat_18_32.vgraph.el_container -format vcsr -check -it 1
./bin/hits_sx -import ./bin/input_graphs/syn_rmat_18_32.vgraph.el_container -format vcsr -check -it 1
./bin/cc_sx -import ./bin/input_graphs/syn_rmat_18_32.vgraph.el_container -format vcsr -check -it 1
./bin/bfs_sx -import ./bin/input_graphs/syn_rmat_18_32.vgraph.el_container -format vcsr -check -it 1

#python3 ./run_tests.py --benchmark --verify --arch=sx --mode=tiny-only --format=vcsr --apps=bfs,sssp