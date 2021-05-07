
# include "config.h"
# include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

void get_mnist_grad() {
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc1_w, d_g_d_fc1_w, H*D*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc1_b, d_g_d_fc1_b, H*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc2_w, d_g_d_fc2_w, H*C*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc2_b, d_g_d_fc2_b, C*sizeof(float)));
}

void reset_mnist_grad() {
    memset(h_g_d_fc1_w, 0, H*D*sizeof(float));
    memset(h_g_d_fc1_b, 0, H*sizeof(float));
    memset(h_g_d_fc2_w, 0, C*H*sizeof(float));
    memset(h_g_d_fc2_b, 0, C*sizeof(float));

    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc1_w, &h_g_d_fc1_w, H*D*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc1_b, &h_g_d_fc1_b, H*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc2_w, &h_g_d_fc2_w, H*C*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc2_b, &h_g_d_fc2_b, C*sizeof(float)));
}

template <int N, int M> void __global__ init_affine_layer(float layer[N][M], int size, curandState state[]) {
    int seq = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
    curand_init(1234, seq, 0, &state[idx]);
#pragma unroll
    for (int i=seq*size; i<(seq+1)*size; i++) {
        layer[i/M][i%M] = curand_uniform(state+idx);
    }
}

void __global__ init_bias(float bias[], int size, curandState state[]) {
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, seq, 0, &state[idx]);
#pragma unroll
    for (int i=seq*size; i<(seq+1)*size; i++) {
        bias[i] = curand_uniform(state+idx);
    }
}

// 128 * 20 * 300
void init_mnist_network() {
    curandState *d_state;
    checkCudaErrors(cudaMalloc(&d_state, 128 * 49 * sizeof(curandState)));

    // init fc1 
    dim3 fc1_block_w(128, 1);
    dim3 fc1_grid_w(49, 1);
    init_affine_layer<H, D><<<fc1_grid_w, fc1_block_w>>>(fc1, 125, d_state);

    dim3 fc1_block_b(20, 1);
    dim3 fc1_grid_b(50, 1);
    init_bias<<<fc1_grid_b, fc1_block_b>>>(b1, 10, d_state);

    // init fc2 
    dim3 fc2_block_w(100, 1);
    dim3 fc2_grid_w(20, 1);
    init_affine_layer<1000, 10><<<fc2_grid_w, fc2_block_w>>>(fc2, 5, d_state);

    dim3 fc2_block_b(10, 1);
    dim3 fc2_grid_w(1, 1);
    init_bias<<<fc2_grid_b, fc2_block_b>>>(b2, 1, d_state);

    // sync data from gpu to cpu 
    checkCudaErrors(cudaMemcpyFromSymbol(&h_fc1, fc1, H * D * sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_b1, b1, H * sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_fc2, fc2, C * H * sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_b2, b2, C * sizeof(float)));

    // free resource
    checkCudaErrors(cudaFree(&d_state));
}


void sync_mnist_model_to_gpu() {
    checkCudaErrors(cudaMemcpyToSymbol(fc1, &h_fc1, H * D * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(b1, &h_b1, H * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(fc2, &h_fc2, C * H * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(b2, &h_b2, C * sizeof(float)));
}


template <int N, int M> void update_matrix(float dmatrix[N][M], float det[N][M], float lr) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            dmatrix[i][j] += lr * det[i][j];
        }
    }
}

template <int N> void update_array(float darr[N], float det[N], float lr) {
    for (int i = 0; i < N; i++) {
        darr[i] += lr * det[i];
    }
}

void update_mnist_model(float lr) {
    get_mnist_grad();

    update_matrix<H, D>(h_fc1, h_g_d_fc1_w, lr);
    update_array<H>(h_b1, h_g_d_fc1_b, lr);
    update_matrix<C, H>(h_fc2, h_g_d_fc2_w, lr);
    update_array<C>(h_b2, h_g_d_fc2_b, lr);

    sync_mnist_model_to_gpu();
    reset_mnist_grad();
}


