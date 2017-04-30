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

// This kernel computes a standard parallel reduction and evaluates the
// time it takes to do that for each block. The timing results are stored
// in device memory.
__global__ static void timedReduction(const float *input, float *output, clock_t *timer)
{
    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    // Copy input.
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // Perform reduction to find minimum.
    for (int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid+gridDim.x] = clock();
}

#define NUM_BLOCKS    64
#define NUM_THREADS   256

// It's interesting to change the number of blocks and the number of threads to
// understand how to keep the hardware busy.
//
// Here are some numbers I get on my G80:
//    blocks - clocks
//    1 - 3096
//    8 - 3232
//    16 - 3364
//    32 - 4615
//    64 - 9981
//
// With less than 16 blocks some of the multiprocessors of the device are idle. With
// more than 16 you are using all the multiprocessors, but there's only one block per
// multiprocessor and that doesn't allow you to hide the latency of the memory. With
// more than 32 the speed scales linearly.

// Start the main CUDA Sample here




#include<stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

__global__ void MCpi(int *a, curandState_t* states, unsigned int NofMcarlo)
{
	int idx = blockIdx.x;
	float x, y;
	int a_1, b_1;

	
	for(int i =0; i < NofMcarlo; ++i)
	{
		x = float(curand(&states[blockIdx.x]) % 1000 )/1000;
		y = float(curand(&states[blockIdx.x]) % 1000 )/1000; 
		
		
		
		if(x*x + y*y < 1)
		{
			a[blockIdx.x] += 1;
		}			

	}

	



}



#define Number 512

int main(int argc, char **argv)
{

	int total_score2 = 0;
	float x;
	float y;
	
	

	srand(time(NULL));
	 int countcpu = 24; 
	for(long int i = 0; i < countcpu; ++i )
	{
		x = float(rand()%1000)/1000);
		y = float(rand()%1000)/1000);
		if(x*x + y*y < 1)
			{total_score2 += 1;
				printf("%d\n", total_score2);
			}
	}

	printf("%f %d\n", total_score2*4 / countcpu, total_score2);	

	curandState_t* states;

  /* allocate space on the GPU for the random states */
  	cudaMalloc((void**) &states, Number * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  	init<<<Number, 1>>>(time(0), states);


	int size = Number * sizeof(int);
	int *a;
	int *da;	
	
	a = (int*)malloc(size);
	cudaMalloc( (void **) &da, size);	

	


 	for(int i = 0; i < Number; ++i )
	{
		a[i] = 0;

	}   
	
	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);

	unsigned int Mcarlo = 10000;
	MCpi<<<Number, 1>>>(da, states, Mcarlo);
	
	
	float true_result;
	float as_is;
	true_result = 3.1415926;

	cudaMemcpy(a, da, size, cudaMemcpyDeviceToHost);

	long int  total_score = 0;
	
	as_is = float( total_score )/(Number*Mcarlo);
	printf("%d \n", total_score*4);
	printf("___________________________________\n");
	printf("%f \n", 4*as_is);
	

	 

	cudaFree(states);
	cudaFree(da);
	return 0;
}
