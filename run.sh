#!/usr/bin/env bash
set -e
cd build
make -j 4 VERBOSE=1

run(){
  IS_DDT_OPEN=false
  if pidof -x $(ps cax | grep ddt) >/dev/null; then
        IS_DDT_OPEN=true
  fi
  if [ "$IS_DDT_OPEN" = true ]; then
    ddt --connect ./test
  else
    ./test
  fi
}

run_ncu(){
  #profile must run in allocated node
  /apps/ACC/NVIDIA-HPC-SDK/23.9/Linux_x86_64/23.9/profilers/Nsight_Compute/ncu --target-processes=application-only --set full -f -o ../profile ./test
}

run_nsys(){
  #profile must run in allocated node
  /apps/ACC/NVIDIA-HPC-SDK/23.9/Linux_x86_64/23.9/profilers/Nsight_Systems/bin/nsys profile -f true --trace=mpi,cuda,nvtx,opengl,osrt --mpi-impl=openmpi -o ../profile mpirun -np 1 ./test
}

run
#run_ncu
#run_nsys
