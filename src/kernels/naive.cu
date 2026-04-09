#include "kernels/naive.h"

__global__ void naive_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K){
	int x = (blockIdx.x * blockDim.x) + threadIdx.y;
	int y = (blockIdx.y * blockDim.y) + threadIdx.x;

	if(x < N && y < M){
		float tmp = 0.0f;
		for(int i = 0; i < K; i++){
			tmp += a[(y * K) + i] * b[(i * N) + x];
		}

		c[(y * N) + x] = tmp;
	}
} // naive_kernel

