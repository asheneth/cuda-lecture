#include "kernels/tiled_coalesced.cuh"

__global__ void tiled_coalesced_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K){
	int x = threadIdx.x;
	int y = threadIdx.y;
	int idx = (y * 32) + x;

	__shared__ float a_tile[32 * 32];
	__shared__ float b_tile[32 * 32];

	for(int t_y = blockIdx.y * 32; t_y < M; t_y += gridDim.y * 32){
		for(int t_x = blockIdx.x * 32; t_x < N; t_x += gridDim.x * 32){
			float tmp = 0.0f;
			for(int t_i = 0; t_i < K; t_i += 32){
				a_tile[idx] = a[((t_y + y) * K) + (t_i + x)];
				b_tile[idx] = b[((t_i + y) * N) + (t_x + x)];

				__syncthreads();

				for(int i = 0; i < 32; i++){
					tmp += a_tile[(y * 32) + i] * b_tile[(i * 32) + x];
				}

				__syncthreads();
			}

			c[((t_y + y) * N) + (t_x + x)] = tmp;
		}
	}
} // tiled_coalesced_kernel

