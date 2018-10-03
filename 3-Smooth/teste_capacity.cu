#include <stdio.h>
	
__global__ void kernel(unsigned long long int *count) {
	atomicAdd(count, (unsigned long long int)1);
	//printf("%llu\n", *count);
}


int main() 
{
	unsigned long long int *count;
	unsigned long long int *d_count;
	int size = sizeof(unsigned long long int);
	
	cudaMalloc((void **)&d_count, size);
	count = (unsigned long long int *) malloc(size);
	
	dim3 dimBlock (481, 271);//Number of Blocks required
    dim3 dimGrid (32, 32);//Number of threads in each block
	
	cudaMemcpy(d_count, 0, size, cudaMemcpyHostToDevice);
	kernel <<< dimBlock, dimGrid >>> (d_count); 
		
	cudaMemcpy(count, d_count, size, cudaMemcpyDeviceToHost);
	
	printf("%llu\n", *count);
	
	free(count);
	cudaFree(d_count);
	
	return 0;
}
