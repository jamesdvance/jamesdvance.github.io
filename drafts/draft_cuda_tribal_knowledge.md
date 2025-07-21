# CUDA Tribal FAQs

*Little Facts Key to Understanding CUDA Kernels*

### Why Do We Index Rows With Block Dim/ Idx / Thread Idx Y In Row-Major?
**memory coalescing** - GPUs achieve high memory bandwidth with by coalescing memory accesses - so if multiple threads within the same warp access, the CPU loads memory together in a single 

In C++, 2D arrays are stored in row-major order. Elements of the same row are contiguous in memory 

### How Many Threads Are There In A Warp? 
32

### What is memory coalescing? 
GPU reads contiguous elements in memory and can do so in one read.

### 