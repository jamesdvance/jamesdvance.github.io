

## Threads, Blocks, and Grids

There are N threads per block (highly device dependent - often 256). There are 


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
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

## Vectors To Speedup GPU Code



## Speeding Up Vector Addition
