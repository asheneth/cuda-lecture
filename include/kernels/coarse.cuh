#ifndef KERNELS_COARSE_CUH
#define KERNELS_COARSE_CUH

__global__ void coarse_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K);

#endif // KERNELS_COARSE_CUH

