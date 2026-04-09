#ifndef KERNELS_COALESCED_CUH
#define KERNELS_COALESCED_CUH

__global__ void coalesced_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K);

#endif // KERNELS_COALESCED_CUH

