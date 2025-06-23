
---
layout: post
title: 80/20 MLOps
date: 2025-6-29 13:11:00
description: A Basic MLOps Setup With Kubeflow
tags: performance-optimization, cuda, memory
categories: performance-optimization
featured: true
---

# Cuda Tiling Techniques

## Simple Example

Matrix A: 

| | | | |
| --- | --- | --- | --- |
| 42 | 43 | 44 | 45 |
| 46 | 47 | 48 | 49 |
| 50 | 51 | 52 | 53 |
| 54 | 55 | 56 | 57 |

Matrix A row-major:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 |

Matrix B:

| | | | | |
| --- | --- | --- | --- | --- |
| 60 | 61 | 62 | 63 | 64 |
| 65 | 66 | 67 | 68 | 69 |
| 70 | 71 | 72 | 73 | 74 |
| 75 | 76 | 77 | 78 | 79 |

Matrix B row-major: 

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 |

Our problem multiples one 4x4 matrix against a 4x5 matrix. The WIDTH of our problem is 4.


For sake of example, let's pretend we're working with the world's smollest GPU. Each block has a width of 2. So, the first block will contain threads 0,0 , 0,1 , 1,0 and 1,1. These threads will all load in tiles together to perform their calculations.  We'll choose a TILE_WIDTH of 2 for illustration. 

The kernel will start by declaring the TILE_WIDTH constant available to all threads. We'll then declare our (row-major) arrays for A and B and our results array R. Finally, we include our Width, 4. 

```c++
# define TILE_WIDTH 2
__global__ void matrixMulKernel(float* A, float* B, float* R, int Width) {
``` 

Now, each thread will initialize two square arrays that will contain all the rows and columns needed to complete each tile-based calculation. These arrays will exist in shared memory, which is R/W accessible to all threads within a block. 

```c++
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]
```

Next, we'll calculate the Row and column that we'll calculate the single value of R for. For example, in thread 0,0 in block 0,0 we'd calculate the sum product A row 0 and B col 0. In code terms:
```python
sum([42,43,44,45] * [60,65,70,75])
```



