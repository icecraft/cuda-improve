#ifndef MY_CONFIG_H_
#define MY_CONFIG_H_
#include <cuda_runtime.h>

const int N = 59136;
const int D = 784;
const int H = 1000;
const int C = 10;

__global__ float MNIST_data[N][D];
__global__ int MNIST_label[N];

__constant__ float fc1[H][D];
__constant__ float b1[H];


__constant__ float fc2[C][H];
__constant__ float b2[C];
#endif 