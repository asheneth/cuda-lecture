#ifndef KERNELS_COALESCED_H
#define KERNELS_COALESCED_H

__global__ void coalesced_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K);

#endif // KERNELS_COALESCED_H

