/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#ifndef NSYS
    #include "nvToolsExt.h"
#endif

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void gpuKernel()
{
  clock_t start = clock();
  while (clock() - start < 700000000);
}

int main(int argc, char**argv) {
  
// Initialize MPI
  MPI_Init(&argc, &argv);
  clock_t start;

  // Get the number of processes
  int numProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Initialize CUDA for this process and set the device
  cudaSetDevice(rank % numProcs);

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Allocate device memory
  int *d_data;
  cudaMalloc(&d_data, sizeof(int) * 256);

    // Launch GPU kernel asynchronously
    double startTime = MPI_Wtime();
    gpuKernel<<<16, 16, 0, stream>>>();

    // Do CPU work concurrently
#ifndef NSYS
    nvtxRangePushA("CPU Code");
#endif
    double startTime2 = MPI_Wtime();
    // Simulate some CPU work with a 1 second delay
    printf("cpuFunction start\n");
    //start = clock();
    //while (clock() - start < 1000000);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "CPU work done by rank " << rank << std::endl;
    double timeCPU = (MPI_Wtime() - startTime2);
#ifndef NSYS
    nvtxRangePop();
#endif

    // Wait for the GPU to finish
    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    printf("GPU work done by rank %d\n", rank);
    printf("CPU Time %f\n",timeCPU);
    double timeCPUGPU = (MPI_Wtime() - startTime);
    printf("GPU Time %f\n",timeCPUGPU-timeCPU);
    printf("Total time: %f\n", timeCPUGPU);
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    MPI_Finalize();

  return 0;
}