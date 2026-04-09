#ifndef KERNELS_TILED_COALESCED_CUH
#define KERNELS_TILED_COALESCED_CUH

__global__ void tiled_coalesced_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K);

#endif // KERNELS_TILED_COALESCED_CUH

