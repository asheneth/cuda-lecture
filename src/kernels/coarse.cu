#define TILE_COUNT 8

__global__ void coarse_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K){
	int x = threadIdx.x;
	int y = threadIdx.y;
	int idx = (y * 32) + x;

	__shared__ float a_tile[32 * 32];
	__shared__ float b_tile[TILE_COUNT][32 * 32];

	for(int t_y = blockIdx.y * 32; t_y < M; t_y += gridDim.y * 32){
		for(int t_x = blockIdx.x * 32 * TILE_COUNT; t_x < N; t_x += gridDim.x * 32 * TILE_COUNT){
			float tmp[TILE_COUNT] = {0.0f};
			for(int t_i = 0; t_i < K; t_i += 32){
				a_tile[idx] = a[((t_y + y) * K) + (t_i + x)];
				#pragma unroll
				for(int t = 0; t < TILE_COUNT; t++){
					if(t_x + (t * 32) >= N) break;

					b_tile[t][idx] = b[((t_i + y) * N) + (t_x + (t * 32) + x)];
				}

				__syncthreads();

				#pragma unroll
				for(int t = 0; t < TILE_COUNT; t++){
					if(t_x + (t * 32) >= N) break;

					for(int i = 0; i < 32; i++){
						tmp[t] += a_tile[(y * 32) + i] * b_tile[t][(i * 32) + x];
					}
				}

				__syncthreads();
			}

			#pragma unroll
			for(int t = 0; t < TILE_COUNT; t++){
				if(t_x + (t * 32) >= N) break;

				c[((t_y + y) * N) + (t_x + (t * 32) + x)] = tmp[t];
			}
		}
	}
} // coarse_kernel

