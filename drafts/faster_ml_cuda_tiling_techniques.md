
---
layout: post
title: Faster CUDA Memory Access With Tiling
date: 2025-6-29 13:11:00
description: Speed Up Kernels with Tiling
tags: performance-optimization, cuda, memory, fasterml
categories: performance-optimization
featured: true
---

# Faster CUDA Memory Access With Tiling

## Glossary
**read-after-write dependence** - threads must wait for data to be written before reading (true dependences)
**write-after-read dependence** - threads must wait for data to be read before (over)writing (false dependence)
**strip-mining** - breaking a long-running loop into an outer loop and an inner loop, where the outer loop breaks the original loop into phases
**global memory** - memory R/W available to entire device
**shared memory** - memory R/W available to entire grid
**local memory** - memory R/W available to thread for array variables
**register memory** - memory R/W available to thread for non-array variables

## High Level


Memory tiling is a technique that exists to reduce calls to CUDA Global Memory. CUDA Global memory seeks are slow and shared memory seeks are much faster. Instead of going to the city's Costco 20 minutes away, threads can pick up what they need from the local corner store 1 minute away. 


## Simple Example

Matrix A: 

| | | |
| --- | --- | --- |
| 1 | 2 | 3 |
| 4 | 5 | 6 |

Matrix A row-major:

| 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 4 | 5 | 6 |

Matrix B:

| | |
| --- | --- |
| 7 | 8 |
| 9 | 10 |
| 11 | 12 |

Matrix B row-major: 

| 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 7 | 8 | 9 | 10 | 11 | 12 |

One thing I hate in most CUDA matMul examples is they start with a simple square matrix example to focus on fundementals, but everything goes out the window when the matrix needs to be adaptive to different shapes. 


For sake of example, let's pretend we're working with the world's smollest GPU. Each block has a width of 2. So, the first block will contain threads 0,0 , 0,1 , 1,0 and 1,1. These threads will all load in tiles together to perform their calculations.  We'll choose a TILE_WIDTH of 2 for illustration. 

The kernel will start by declaring the TILE_WIDTH constant available to all threads. We'll then declare our (row-major) arrays for A and B and our results array R. Finally, we include our Width, 4. 

```c++
# define TILE_WIDTH 2
__global__ void squareMatrixMulKernel(float* A, float* B, float* R, int Width) {
``` 

Now, each thread will initialize two square arrays that will contain all the rows and columns needed to complete each tile-based calculation. These arrays will exist in shared memory, which is R/W accessible to all threads within a block. 

```c++
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]
```

Next, we'll calculate the Row and column that we'll calculate the single value of R for. For example, in thread 0,0 in block 0,0 we'd calculate the sum product A row 0 and B col 0. In plain Python this looks like:
```python
sum([42,43,44,45] * [60,65,70,75])
```
In our cuda kernel, we define our row in terms y indices and columns in terms of x indices. The reason is to best coalesce memory reads from warps. In this case, CRow and Col define the two arrays we'll need to take the sumproduct of. 
```c++
int Row = blockIdx.y * TILE_WIDTH + threadIdx.y
int Col = blockIdx.x * TILE_WIDTH + threadIdx.x
```
Next, we begin our phased for loop. Each phase represents a loop over the memory tile, which is a subset of our full row and column. We will need as many phases as tiles fit into our total row and column. 
```c++
float value = 0.0f;
for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
    ...
}
```

Inside the loop, the threads collaborate by loading in a subsection of the matrices A and B into shared memory. Notice each one loads in one element unique to his x,y thread index. The index is nearly identical to the standard matrix multiplication index, except that we are only operating over the TILE portion of our row and column. After both values are loaded, we then call __syncthreads. This ensures no threads will continue executing the kernel until all threads have finished these operations. When we move on to the next step, both matrices Mds and Nds will be populated with all values. 
```c++
Mds[threadIdx.y][threadIdx.x] = M[Row*Width + ph*TILE_WIDTH +threadIdx.x];
Nds[threadIdx.y][threadIdx.x] = N[(ph * TILE_WIDTH + threadIdx.y) + Col];
__syncthreads();
```
Next, the kernel performs a for-loop over the row/column combination, much like in matrix multiplication, but only for the length of the tile. Notice this smaller shared memory array aligns to the number of threads who each will need to perform an operation over one of its rows + columns. At the end, we sync threads again, so that no one starts updating the shared memory before we are finished reading from it. This is called `write-after-read` dependence, or "false dependence". 
```c++
for (int i = 0; i < TILE_WIDTH; i++ ){
    value += Mds[threadIdx.y][i] *Ndx[i][threadIdx.x]
}
__syncthreads();
```
finally, once both the phase loop and the inner matmul loop have finished, we assign our value to the index of the results matrix R we are responsible for: 
```c++
P[Row*Width + Col] = Pvalue; 
```

Tiling is a unique tool for GPUs that allows them to utilize concurrent execution to coorporate and reduce global memory reads. While real-world matrix multiplication has been optimized to absurd degrees, the matmul example is a great way to build intuition for memory tiling. A common phrase on the GPU Mode discord is "It's the Memory Stupid". Meaning, memory access is a major source of latency in CUDA kernels. 

The full kernel below: 
```c++

```

## Proving The Concept

To prove our memory tiled matMul kernel is faster than a global memory approach, we could run an example on a single kernel and measure then tenths of milliseconds of difference. But to really believe our code could work, I prefer to register our kernel as a matMul within pytorch and measure the difference in real operations.






## Appendix

### Resources

[How To Optimize A CUDA Matmul KErnel for cuBLAS-like Performance: A Worklog](https://siboehm.com/articles/22/CUDA-MMM)

