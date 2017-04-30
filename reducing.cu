/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This example shows how to use the clock function to measure the performance of
// block of threads of a kernel accurately.
//
// Blocks are executed in parallel and out of order. Since there's no synchronization
// mechanism between blocks, we measure the clock once for each block. The clock
// samples are written to device memory.

// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>








#include<stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>



void rand_ints(int *a, int n)
{
	for(int i = 0; i< n; ++i)
	{
		*a = rand()%100;
		a++;
	}





}

#include<ctime>

int imin(int a, int b)
{
	if(a>b)
		return b;
	else
		return a;
}

const int N = 33 * 1024;
const int threadPerBlock = 256;

__global__ void dot(float *a, float *b, float *c)
{
	__shared__ float cache (threadPerBlock);
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int cacheIndex = threadIdx.x;
	
	float temp = 0;
		
	while(tid < N)
	{
		temp += a[tid]*b[tid];
		tid += blockDim.x*gridDim.x;


	}

	cache[cacheIndex] = temp;
	__syncthreads;
	
	int i = blockDim.x/2;
	while(i!= 0)
	{
		if(cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i/=2;		
	}


	if(cacheIndex == 0)
	{
		c[blockIdx.x] = cache[0];


	}

}





int main(int argc, char **argv)
{
	unsigned int start_time_1 = clock();

	int *a, *b, *c;
	int *da, *db, *dc;
	

	int size = N*sizeof(int);

	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);
	cudaMalloc((void**)&dc, size);


	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);
	rand_ints(a, N);
	rand_ints(b, N);
	
	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

	add3<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(da, db, dc);
	
	cudaMemcpy(a, da, size, cudaMemcpyDeviceToHost);
	
	free(a);
	free(b);
	free(c);


	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
	unsigned int end_time_1 = clock();

	printf("%d\n",  end_time_1 - start_time_1);


	unsigned int start_time_2 = clock();
	
	int *a1;
	int *b1;
	int *c1;
	a1 = (int*)malloc(N*sizeof(int));
	b1 = (int*)malloc(N*sizeof(int));
	c1 = (int*)malloc(N*sizeof(int));
	rand_ints(a1, N);
	rand_ints(b1, N);

	for(int i = 0; i< N; ++i)
	{
		c1[i] = a1[i] + b1[i];


	}
	

	unsigned int end_time_2 = clock();

	printf("%d\n", end_time_2 - start_time_2);


	













	return 0;
}
