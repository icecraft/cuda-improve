
# include "config.h"
# include "shared.h"

# include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdlib.h>

template <int N, int M> void randomDumpMatrixEle(float layer [][M], int nums) {
    int total = N * M, tmp;
    for (int i=0; i < nums; i++) {
        tmp = rand() % total;
        printf(" %f ", layer[tmp/M][tmp%M]);
    }
    printf("\n");
}

void get_mnist_grad() {
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc1_w, d_g_d_fc1_w, H*D*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc1_b, d_g_d_fc1_b, H*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc2_w, d_g_d_fc2_w, H*C*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc2_b, d_g_d_fc2_b, C*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_loss, d_loss, sizeof(float)));
}

void reset_mnist_grad() {
    memset(h_g_d_fc1_w, 0, H*D*sizeof(float));
    memset(h_g_d_fc1_b, 0, H*sizeof(float));
    memset(h_g_d_fc2_w, 0, C*H*sizeof(float));
    memset(h_g_d_fc2_b, 0, C*sizeof(float));
    h_loss = 0;

    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc1_w, &h_g_d_fc1_w, H*D*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc1_b, &h_g_d_fc1_b, H*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc2_w, &h_g_d_fc2_w, H*C*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc2_b, &h_g_d_fc2_b, C*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_loss, &h_loss, sizeof(float)));
}

template <int M> __global__ void init_affine_layer_fc1(int size, curandState state[]) {
    int seq = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
    curand_init(1234, seq, 0, &state[seq]);
#pragma unroll
    for (int i=seq*size; i<(seq+1)*size; i++) {
        fc1[i/M][i%M] = curand_uniform(state+seq);
    }
}

template <int M> __global__ void init_affine_layer_fc2(int size, curandState state[]) {
    int seq = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
    curand_init(1234, seq, 0, &state[seq]);
#pragma unroll
    for (int i=seq*size; i<(seq+1)*size; i++) {
        fc2[i/M][i%M] = curand_uniform(state+seq);
    }
}

__global__ void init_bias(float bias[], int size, curandState state[]) {
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, seq, 0, &state[seq]);
#pragma unroll
    for (int i=seq*size; i<(seq+1)*size; i++) {
        bias[i] = curand_uniform(state+seq);
    }
}

// 128 * 20 * 300
void init_mnist_network() {
    curandState *d_state;
    checkCudaErrors(cudaMalloc(&d_state, 784 * 30 * sizeof(curandState)));

    float *b_arr;

    // init fc1 
    dim3 fc1_block_w(32, 1);
    dim3 fc1_grid_w(49, 1);
    init_affine_layer_fc1<D><<<fc1_grid_w, fc1_block_w>>>(15, d_state);
    cudaDeviceSynchronize();

    dim3 fc1_block_b(H, 1);
    dim3 fc1_grid_b(1, 1);
    checkCudaErrors(cudaGetSymbolAddress((void **)&b_arr, b1));
    init_bias<<<fc1_grid_b, fc1_block_b>>>(b_arr, 1, d_state);
    cudaDeviceSynchronize();

    // init fc2 
    dim3 fc2_block_w(H, 1);
    dim3 fc2_grid_w(C, 1);
    init_affine_layer_fc2<H><<<fc2_grid_w, fc2_block_w>>>(1, d_state);
    cudaDeviceSynchronize();

    dim3 fc2_block_b(C, 1);
    dim3 fc2_grid_b(1, 1);
    checkCudaErrors(cudaGetSymbolAddress((void **)&b_arr, b2));
    init_bias<<<fc2_grid_b, fc2_block_b>>>(b_arr, 1, d_state);
    cudaDeviceSynchronize();

    // sync data from gpu to cpu 
    checkCudaErrors(cudaMemcpyFromSymbol(&h_fc1, fc1, H * D * sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_b1, b1, H * sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_fc2, fc2, C * H * sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_b2, b2, C * sizeof(float)));

    // free resource
    checkCudaErrors(cudaFree(d_state));
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
            dmatrix[i][j] -= lr * det[i][j];
        }
    }
}

template <int N> void update_array(float darr[N], float det[N], float lr) {
    for (int i = 0; i < N; i++) {
        darr[i] -= lr * det[i];
    }
}

void update_mnist_model(float lr) {
    get_mnist_grad();
    printf("loss: %f\n", h_loss);

    update_matrix<H, D>(h_fc1, h_g_d_fc1_w, lr);
    update_array<H>(h_b1, h_g_d_fc1_b, lr);
    update_matrix<C, H>(h_fc2, h_g_d_fc2_w, lr);
    update_array<C>(h_b2, h_g_d_fc2_b, lr);

    sync_mnist_model_to_gpu();
    reset_mnist_grad();
}

