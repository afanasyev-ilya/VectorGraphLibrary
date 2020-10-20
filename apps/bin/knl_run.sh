#!/bin/bash

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=64

numactl --membind 0 lscpu
OMP_NUM_THREADS=64   numactl --membind 1 ./test_knl

export OMP_NUM_THREADS=128
OMP_NUM_THREADS=128   numactl --membind 1 ./test_knl

export OMP_NUM_THREADS=256
OMP_NUM_THREADS=256   numactl --membind 1 ./test_knl