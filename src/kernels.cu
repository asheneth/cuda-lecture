#include <cstdlib>
#include <iostream>

#include "kernels/naive.cuh"
#include "kernels/naive_correct.cuh"
#include "kernels/tiled.cuh"
#include "kernels/tiled_coalesced.cuh"
#include "kernels/coarse.cuh"
#include "kernels.cuh"

cudaEvent_t start = nullptr;
cudaEvent_t stop;

int M = 0;
int N = 0;
int K = 0;

float *a_h = nullptr;
float *b_h;
float *c_h;

float *a_d;
float *b_d;
float *c_d;

void make_timer(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
} // make_timer

void make_buffers(){
	srand(42);

	a_h = (float *)malloc(M * K * sizeof(float));
	b_h = (float *)malloc(K * N * sizeof(float));
	c_h = (float *)malloc(M * N * sizeof(float));

	for(int i = 0; i < M * K; i++){
		a_h[i] = ((float)rand()) / ((float)RAND_MAX);
	}

	for(int i = 0; i < K * N; i++){
		b_h[i] = ((float)rand()) / ((float)RAND_MAX);
	}

	cudaMalloc(&a_d, M * K * sizeof(float));
	cudaMalloc(&b_d, K * N * sizeof(float));
	cudaMalloc(&c_d, M * N * sizeof(float));

	cudaMemcpy(a_d, a_h, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, K * N * sizeof(float), cudaMemcpyHostToDevice);
} // make_buffers

void free_buffers(){
	if(a_h != nullptr){
		free(a_h);
		free(b_h);
		free(c_h);

		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);
	}
} // free_buffers

bool check_c(bool b_col){
	cudaMemcpy(c_h, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	for(int y = 0; y < M; y++){
		for(int x = 0; x < N; x++){
			float tmp = 0.0f;
			for(int k = 0; k < K; k++){
				if(b_col){
					tmp += a_h[(y * K) + k] * b_h[(x * N) + k];
				}else{
					tmp += a_h[(y * K) + k] * b_h[(k * N) + x];
				}
			}

			if(c_h[(y * N) + x] < tmp * 0.99 || tmp * 1.01 < c_h[(y * N) + x]){
				std::cout << x << ", " << y << std::endl;
				std::cout << tmp << ", " << c_h[(y * N) + x] << std::endl;
				return false;
			}
		}
	}

	return true;
} // check_c

void do_kernel(int kernel, int i_M, int i_N, int i_K, bool check){
	if(i_M % 32 != 0){
		std::cerr << "M must be a multiple of 32..." << std::endl;
	}

	if(i_N % 32 != 0){
		std::cerr << "N must be a multiple of 32..." << std::endl;
	}

	if(i_K % 32 != 0){
		std::cerr << "K must be a multiple of 32..." << std::endl;
	}

	if(M != i_M || N != i_N || K != i_K){
		M = i_M;
		N = i_N;
		K = i_K;

		make_buffers();
	}

	if(start == nullptr){
		make_timer();
	}

	cudaEventRecord(start, 0);

	switch(kernel){
		case 1:{
			dim3 grid_size(N / 32, M / 32);
			dim3 block_size(32, 32);

			naive_kernel<<<grid_size, block_size>>>(a_d, b_d, c_d, M, N, K);

			break;
		}
		case 2:{
			dim3 grid_size(N / 32, M / 32);
			dim3 block_size(32, 32);

			naive_correct_kernel<<<grid_size, block_size>>>(a_d, b_d, c_d, M, N, K);

			break;
		}
		case 3:{
			dim3 grid_size(N / 32, M / 32);
			dim3 block_size(32, 32);

			tiled_kernel<<<grid_size, block_size>>>(a_d, b_d, c_d, M, N, K);

			break;
		}
		case 4:{
			dim3 grid_size(N / 32, M / 32);
			dim3 block_size(32, 32);

			tiled_coalesced_kernel<<<grid_size, block_size>>>(a_d, b_d, c_d, M, N, K);

			break;
		}
		case 5:{
			dim3 grid_size(N / 32, M / 32);
			dim3 block_size(32, 32);

			coarse_kernel<<<grid_size, block_size>>>(a_d, b_d, c_d, M, N, K);

			break;
		}
		default:
			std::cout << "Invalid kernel..." << std::endl;
			return;
	}

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();

	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	std::cout << "Time: " << ms << "ms" << std::endl;

	if(check){
		if(!check_c(false)){
			std::cout << "ERROR: c is incorrect" << std::endl;
		}
	}
} // do_kernel

void cleanup(){
	free_buffers();

	if(start != nullptr){
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
} // cleanup
