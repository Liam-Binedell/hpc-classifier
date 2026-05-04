#!/usr/bin/env sh

make clean
make
mpiexec -n 4 ./classifier_mpi
