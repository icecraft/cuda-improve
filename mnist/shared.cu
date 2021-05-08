# include <cuda_runtime.h>
# include "config.h"

__device__ float MNIST_data[NN][D];
__device__ int MNIST_label[NN];

__device__ float fc1[H][D];
__device__ float b1[H];
__device__ float fc2[C][H];
__device__ float b2[C];
__device__ float d_loss;


float h_fc1[H][D];
float h_b1[H];
float h_fc2[C][H];
float h_b2[C];
float h_loss;

__device__ float d_g_d_fc1_w[H][D]; // device(d) global(g) deviration(d)
__device__ float d_g_d_fc1_b[H];
__device__ float d_g_d_fc2_w[C][H];
__device__ float d_g_d_fc2_b[C];

float h_g_d_fc1_w[H][D];
float h_g_d_fc1_b[H];
float h_g_d_fc2_w[C][H];
float h_g_d_fc2_b[C];