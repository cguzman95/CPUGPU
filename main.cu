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
  while (clock() - start < 400000000);
}

int main(int argc, char **argv)
{
  
  MPI_Init(&argc, &argv);
  // Get the number of processes
  int numProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  //clock_t start;
  const int blockSize = 256, nStreams = 1;
  const int n = 4 * 1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );
  
  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

  float ms; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );
  
  // asynchronous version 2: 
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < 1; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
  }
  for (int i = 0; i < 1; ++i)
  {
    int offset = i * streamSize;
    gpuKernel<<<16, 16, 0, stream[i]>>>();
    cudaStreamQuery(stream[i]);
  }
#ifndef NSYS
    nvtxRangePushA("CPU Code");
#endif
    std::this_thread::sleep_for(std::chrono::seconds(1));
#ifdef NSYS
    nvtxRangePop();
#endif
  for (int i = 0; i < 1; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
    cudaStreamQuery(stream[i]);
  }
  cudaDeviceSynchronize();
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);
  // Finalize MPI
  MPI_Finalize();

  return 0;
}