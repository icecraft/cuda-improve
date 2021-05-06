#ifndef MY_CONFIG_H_
#define MY_CONFIG_H_
#include <cuda_runtime.h>

const int N = 59136;
const int D = 784;
const int H = 1000;
const int C = 10;

__device__ float MNIST_data[N][D];
__device__ int MNIST_label[N];

__device__ float fc1[H][D];
__device__ float b1[H];


__device__ float fc2[C][H];
__device__ float b2[C];
#endif 