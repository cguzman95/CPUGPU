#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#ifdef NSYS
    #include "nvToolsExt.h"
#endif

// CUDA kernel function
__global__ void gpuKernel(int *d_data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_data[idx] = value;
    // Adding a delay (approx. 1 second)
    clock_t start = clock();
    while (clock() - start < 4000000000);
}

void cpuFunction(int rank) {
    // Simulate some CPU work with a 1 second delay
    clock_t start = clock();
    printf("cpuFunction start\n");
    while (clock() - start < 1000000);
    //std::this_thread::sleep_for(std::chrono::seconds(1));
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
    double startTime = MPI_Wtime();
    gpuKernel<<<16, 16, 0, stream>>>(d_data, rank);

    // Do CPU work concurrently
#ifdef NSYS
    nvtxRangePushA("CPU Code");
#endif
    double startTime2 = MPI_Wtime();
    cpuFunction(rank);
    double timeCPU = (MPI_Wtime() - startTime2);
#ifdef NSYS
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

    // Free device memory
    cudaFree(d_data);

    // Destroy the CUDA stream
    cudaStreamDestroy(stream);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
