#!/bin/bash


for N in 1 2 4 8
do
    mpirun -np $N python3 shape_opt_mpi.py
    for ((i=0;i<N;i++)); 
    do
        mv profile_out."$i" ./run_profiles_separate/num_cp_25/profile_256/output_256_"$N"_01."$i"
    done
done

