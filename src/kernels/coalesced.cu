#include "kernels/coalesced.cuh"

__global__ void coalesced_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K){
	for(int y = (blockIdx.y * blockDim.y) + threadIdx.x; y < M; y += blockDim.y * gridDim.y){
		for(int x = (blockIdx.x * blockDim.x) + threadIdx.y; x < N; x += blockDim.x * gridDim.x){
			float tmp = 0.0f;
			for(int i = 0; i < K; i++){
				tmp += a[(y * K) + i] * b[(i * N) + x];
			}

			c[(y * N) + x] = tmp;
		}
	}
} // coalesced_kernel

