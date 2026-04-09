__global__ void naive_correct_kernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int K){
	for(int y = (blockIdx.y * blockDim.y) + threadIdx.y; y < M; y += blockDim.y * gridDim.y){
		for(int x = (blockIdx.x * blockDim.x) + threadIdx.x; x < N; x += blockDim.x * gridDim.x){
			float tmp = 0.0f;
			for(int i = 0; i < K; i++){
				tmp += a[(y * K) + i] * b[(i * N) + x];
			}

			c[(y * N) + x] = tmp;
		}
	}
} // naive_correct_kernel

