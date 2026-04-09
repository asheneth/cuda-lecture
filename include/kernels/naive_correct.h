#ifndef KERNELS_NAIVE_CORRECT_H
#define KERNELS_NAIVE_CORRECT_H

__global__ void naive_correct_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K);

#endif // KERNELS_NAIVE_CORRECT_H

