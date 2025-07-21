# Barely Groking CUDA

As an ML Engineer most of my job is focused on building systems and infrastructure. Therefore

## Threads, Blocks, and Grids

There are N threads per block (highly device dependent - often 256). There are 

## Outline

# Mimick lecture 1 of CUDA mode - show how to do a simple Pytorch speedup

## The Indices

There are better explainers out there, 

```C++
#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    int i = threadIdx.x + blockDim.x * blockIdx.x; 

    if(i < N){
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {S
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

## Column / Row Major

## Vectors To Speedup GPU Code



## Speeding Up Vector Addition
