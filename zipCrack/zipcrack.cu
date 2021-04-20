


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

# include "config.h"



__device__ long crc32(int ch, long crc) {
  return (crc >> 8) ^ c_crctable[(crc ^ ch) & 0xFF];
}


__global__ void doZipCrack36(int n, long turn, int check_byte) {
  // Block index
/*
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  

  // setup init env
  int prepwd[3] = {c_base36[bx], c_base36[by*6 + ty] ,c_base36[tx]};
  */
   int prepwd[3] = {49, 50, 51};
  int data[16] = {0};
  int dataIndex[16] = {0};

  for (int i = 0; i < n-3; i++) {
    data[i]  = c_base36[0];
    dataIndex[i] = 0;
  }

  
  long key0, key1, key2;
  int i = 0, tmp=0, k = 0, carry=0;
  int sum, ni;
  for (int j = 0; j < turn; j++) {
    key0 = 305419896;
    key1 = 591751049;
    key2 = 878082192;

    for (i = 0; i < 3; i++) {
      key0 = crc32(prepwd[i], key0);
      key1 = (key1 + (key0 & 0xFF)) & 0xFFFFFFFF;
      key1 = (key1 * 134775813 + 1) & 0xFFFFFFFF;
      key2 = crc32(key1 >> 24, key2);
    }

    for (i = 0; i < n-3; i++) {
      key0 = crc32(data[i], key0);
      key1 = (key1 + (key0 & 0xFF)) & 0xFFFFFFFF;
      key1 = (key1 * 134775813 + 1) & 0xFFFFFFFF;
      key2 = crc32(key1 >> 24, key2);
    }

    // 
    for (i = 0; i < 12; i++) {
      k = key2 | 2;
      tmp = c_feature[i] ^ ((k * (k^1)) >> 8) & 0xFF;
      key0 = crc32(tmp, key0);
      key1 = (key1 + (key0 & 0xFF)) & 0xFFFFFFFF;
      key1 = (key1 * 134775813 + 1) & 0xFFFFFFFF;
      key2 = crc32(key1 >> 24, key2);
    }
    

    if ( tmp == check_byte) {
      printf("%d %d %d ", prepwd[0], prepwd[1], prepwd[2]);
      for (i = 0; i < n-3; i++) {
        printf("%d ", data[i]);
      }
      printf("\n");
    }

    carry = 1;
    for (i = 0; i < n-3; i++) {
      sum = carry + dataIndex[i];
      carry = sum / 36;
      ni = sum % 36;
      dataIndex[i] = ni;
      data[i] = c_base36[ni];
    }
  }
  return ;
}


void load_constant() {
  checkCudaErrors(cudaMemcpyToSymbol(c_crctable, &crctable, 256 * sizeof(long)));
  checkCudaErrors(cudaMemcpyToSymbol(c_base36, &base36, 36 * sizeof(int)));
}


int main(int argc, char **argv) {
  int dev = 0; // cuda device seq

  int check_byte = 105;   // get via command line
  int feature[] = {0x2d, 0x40, 0xa2, 0x29, 0x41, 0x65, 0xf5, 0x78, 0xce, 0x92, 0x23, 0x40}; // should read from zipfile
  
  checkCudaErrors(cudaMemcpyToSymbol(c_feature, &feature, 12 * sizeof(int)));
  load_constant();
  // warmup 

  /*
  dim3 block(36, 6);
  dim3 grid(36, 6);
  */
  
  dim3 block(1);
  dim3 grid(1);
  
  doZipCrack36 <<< grid, block >>>(6, pow(36, 6-3), check_byte);
  cudaDeviceSynchronize();
  return 0;
}
