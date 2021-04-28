


// System includes
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <ctime>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


__device__ int ret_c[128];
__device__ int data_c[51200];

__global__ void arraySum(int count) {
	 int seq =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
	 int sum = 0;

	 #pragma unroll
	 for (int i = seq * count; i < (seq+1)*count ; i ++) {
	     sum += data_c[i];
	     sum %= 1000000007;
	 }
	 ret_c[seq] = sum;
}


int main(int argc, char **argv) {
  int ret[128] = {0};
  int data[51200] = {0};
  int sum = 0, sum_c = 0;
  
  for (int i = 0; i< 128; i++) {
      for (int j = 0; j < 400; j ++) {
      	  data[i*400 +j] = i;
      }
  }

  struct timeval pre_cuda{}, after_cuda{}, after_cpu{};
  gettimeofday(&pre_cuda, nullptr);
    
  checkCudaErrors(cudaMemcpyToSymbol(ret_c, &ret, 128 * sizeof(int)));
  checkCudaErrors(cudaMemcpyToSymbol(data_c, &data, 51200 * sizeof(int)));

  dim3 threadsPerBlock(32, 1);
  dim3 blockNums(4);
  arraySum<<<blockNums, threadsPerBlock>>>(400);

  cudaDeviceSynchronize();
  
  checkCudaErrors(cudaMemcpyFromSymbol(&ret, ret_c, 128 * sizeof(int)));
  
  for (int i = 0; i < 128; i++) {
      sum_c += ret[i];
  }
  gettimeofday(&after_cuda, nullptr);

  for (int i = 0; i < 51200; i ++) {
      sum += data[i];
      sum %= 1000000007;
  }

  gettimeofday(&after_cpu, nullptr);
printf("cuda time cost: %ld %ld, cpu time cost: %ld %ld\n", after_cuda.tv_sec - pre_cuda.tv_sec, after_cuda.tv_usec - pre_cuda.tv_usec,
	     	  	       	   	      	         after_cpu.tv_sec - after_cuda.tv_sec, after_cpu.tv_usec - after_cuda.tv_usec);
return 0;							 							 
}

