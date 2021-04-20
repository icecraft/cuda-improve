


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

# include "config.h"


const int BLOCK_SIZE = 32;
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void doZipCrack(int n, int check_byte) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  

  // 
}


void load_crc_table() {
  checkCudaErrors(cudaMemcpyToSymbol(c_crctable, &crctable, 256 * sizeof(long)));
}

int main(int argc, char **argv) {
  int dev = 0; // cuda device seq

  int check_byte = 105;   // get via command line
  int feature[] = {0x2d, 0x40, 0xa2, 0x29, 0x41, 0x65, 0xf5, 0x78, 0xce, 0x92, 0x23, 0x40}; // should read from zipfile
  
  checkCudaErrors(cudaMemcpyToSymbol(c_feature, &feature, 12 * sizeof(int)));

  // warmup 

  dim3 block(36, 6);
  dim3 grid(36, 6);

  return 0;
}
