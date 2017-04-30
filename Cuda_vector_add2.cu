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






__global__ void add(int *first, int * second, int * result, int n)
{
	
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	while(tid < n)
		{
			result[tid] = first[tid] + second[tid];
			tid += blockDim.x* gridDim.x; 
		}




}




int main(int argc, char **argv)
{


	int *a, *b, *c;


	int *da, *db, *dc;


	const int NU = 100000;

	a = (int*)malloc(NU*sizeof(int));
	b = (int*)malloc(NU*sizeof(int));
	
	c = (int*)malloc(NU*sizeof(int));

	
	for(int i = 0; i< NU; ++i)
	{
		a[i] = 2;
		b[i] = 5;
		
	}

	cudaMalloc((void**)&da, NU*sizeof(int));
	cudaMalloc((void**)&db, NU*sizeof(int));
	cudaMalloc((void**)&dc, NU*sizeof(int));

	cudaMemcpy(da, a, NU*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, NU*sizeof(int), cudaMemcpyHostToDevice);

	
	add<<<(N + threadPerBlock - 1)/threadPerBlock, threadPerBlock >>>(da, db, dc, NU);



	cudaMemcpy(c, dc, NU*sizeof(int), cudaMemcpyDeviceToHost);
	

	for(int i = 0; i< 100; ++i)
	{
		printf("%d\n", c[i]);


	}	













	return 0;
}
