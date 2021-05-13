#!/bin/sh

#mpirun -v -np 1 -ve 0 ./a.out
mpirun -v -np 2 -ve 0-1 ./a.out