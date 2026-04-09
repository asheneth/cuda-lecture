#include <cstdio>
#include <iostream>
#include <string>

#include "kernels.cuh"

__global__ void hello_world(){
	printf("Hello, world!\n");
} // hello_world

int main(int argc, char **argv){
	if(argc != 2){
		std::cerr << "Missing kernel index..." << std::endl;

		exit(-1);
	}

	int kernel = std::stoi(argv[1]);

	if(kernel == 0){
		hello_world<<<1, 1>>>();

		cudaDeviceSynchronize();

		exit(0);
	}

	do_kernel(kernel, 128, 64, 96, true);

	for(int i = 1; i < 6; i++){
		do_kernel(i, 4096, 4096, 4096, false);
	}

	cleanup();
} // main
