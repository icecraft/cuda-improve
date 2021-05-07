#ifndef MY_CONFIG_H_
#define MY_CONFIG_H_

const int NN = 59136;
const int D = 784;
const int H = 1000;
const int C = 10;

extern __device__ float MNIST_data[NN][D];
extern __device__ int MNIST_label[NN];

extern __device__ float fc1[H][D];
extern __device__ float b1[H];
extern __device__ float fc2[C][H];
extern __device__ float b2[C];


extern float h_fc1[H][D];
extern float h_b1[H];
extern float h_fc2[C][H];
extern float h_b2[C];

extern __device__ float d_g_d_fc1_w[H][D]; // device(d) global(g) deviration(d)
extern __device__ float d_g_d_fc1_b[H];
extern __device__ float d_g_d_fc2_w[C][H];
extern __device__ float d_g_d_fc2_b[C];

extern float h_g_d_fc1_w[H][D];
extern float h_g_d_fc1_b[H];
extern float h_g_d_fc2_w[C][H];
extern float h_g_d_fc2_b[C];


#endif 