#include <cstdio>

__global__ void hello_world(){
	printf("Hello, world!\n");
} // hello_world

int main(){
	hello_world<<<1, 1>>>();

	cudaDeviceSynchronize();
} // main

