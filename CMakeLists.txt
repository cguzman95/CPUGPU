cmake_minimum_required(VERSION 3.10)

# Project name
project(MPI_CUDA_Example)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Set compilation flags for MPI and CUDA
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} ${CUDA_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -arch=sm_70")#--default-stream per-thread 

# Specify the source files
set(SOURCES main.cu)
# Add the executable
cuda_add_executable(test ${SOURCES})

# Link MPI and CUDA libraries
#find_library(LIBNVTOOLSEXT nvToolsExt PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#target_link_libraries(test ${LIBNVTOOLSEXT})
target_link_libraries(test ${CUDA_LIBRARIES})
target_link_libraries(test ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
