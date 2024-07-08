#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#ifndef NSYS
    #include "nvToolsExt.h"
#endif

// CUDA kernel function
__global__ void gpuKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Adding a delay (approx. 1 second)
    clock_t start = clock();
    while (clock() - start < 4000000000);
}

int main(int argc, char** argv) {
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
