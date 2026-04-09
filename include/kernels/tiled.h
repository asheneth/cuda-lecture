#ifndef KERNELS_TILED_H
#define KERNELS_TILED_H

__global__ void tiled_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K);

#endif // KERNELS_TILED_H


