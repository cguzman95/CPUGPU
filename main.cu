#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include "nvToolsExt.h"

// CUDA kernel function
__global__ void gpuKernel(int *d_data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_data[idx] = value;
}

void cpuFunction(int rank) {
    // Simulate some CPU work
    for (int i = 0; i < 100000000; ++i);
    std::cout << "CPU work done by rank " << rank << std::endl;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

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
    gpuKernel<<<16, 16, 0, stream>>>(d_data, rank);

    // Do CPU work concurrently
    nvtxRangePushA("CPU Code");
    cpuFunction(rank);
    nvtxRangePop();

    // Wait for the GPU to finish
    cudaStreamSynchronize(stream);

    // Free device memory
    cudaFree(d_data);

    // Destroy the CUDA stream
    cudaStreamDestroy(stream);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
