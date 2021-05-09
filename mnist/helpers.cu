
# include "config.h"
# include "shared.h"

# include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdlib.h>

template <int N, int M> void randomDumpMatrixEle(float layer [][M], int nums, float scale) {
    srand (0);
    int total = N * M, tmp;
    for (int i=0; i < nums; i++) {
        tmp = rand() % total;
        printf(" %0.4f ", layer[tmp/M][tmp%M]*scale);
    }
    printf("\n");
}

template <int N, int M> void dumpMatrixEle(float layer [][M]) {
    for (int i=0; i < N; i++) {
        for (int j=0; j < M; j++) {
            printf(" %0.2f ", layer[i][j]);
        }
        printf("\n");
    }
}

 void dumpArray(float arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%0.2f ", arr[i]);
    }
    printf("\n");
}

void get_mnist_grad() {
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc1_w, d_g_d_fc1_w, H*D*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc1_b, d_g_d_fc1_b, H*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc2_w, d_g_d_fc2_w, H*C*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_g_d_fc2_b, d_g_d_fc2_b, C*sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_loss, d_loss, sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_count, d_count, sizeof(float)));
}

void reset_mnist_grad() {
    memset(h_g_d_fc1_w, 0, H*D*sizeof(float));
    memset(h_g_d_fc1_b, 0, H*sizeof(float));
    memset(h_g_d_fc2_w, 0, C*H*sizeof(float));
    memset(h_g_d_fc2_b, 0, C*sizeof(float));
    h_loss = 0;
    h_count = 0;

    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc1_w, &h_g_d_fc1_w, H*D*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc1_b, &h_g_d_fc1_b, H*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc2_w, &h_g_d_fc2_w, H*C*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_g_d_fc2_b, &h_g_d_fc2_b, C*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_loss, &h_loss, sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_count, &h_count, sizeof(float)));
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


// 128 * 20 * 300
void init_mnist_network() {
    curandState *d_state;
    checkCudaErrors(cudaMalloc(&d_state, 784 * 30 * sizeof(curandState)));

    // init fc1 
    dim3 fc1_block_w(32, 1);
    dim3 fc1_grid_w(49, 1);
    init_affine_layer_fc1<D><<<fc1_grid_w, fc1_block_w>>>(15, d_state);
    cudaDeviceSynchronize();

    memset(h_b1, 0, H * sizeof(int));
    
    // init fc2 
    dim3 fc2_block_w(H, 1);
    dim3 fc2_grid_w(C, 1);
    init_affine_layer_fc2<H><<<fc2_grid_w, fc2_block_w>>>(1, d_state);
    cudaDeviceSynchronize();

    memset(h_b2, 0, C * sizeof(int));


    // sync data from gpu to cpu 
    checkCudaErrors(cudaMemcpyFromSymbol(&h_fc1, fc1, H * D * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(b1, &h_b1, H * sizeof(float)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_fc2, fc2, C * H * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(b2, &h_b2, C * sizeof(float)));

    // free resource
    checkCudaErrors(cudaFree(d_state));
}


void sync_mnist_model_to_gpu() {
    checkCudaErrors(cudaMemcpyToSymbol(fc1, &h_fc1, H * D * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(b1, &h_b1, H * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(fc2, &h_fc2, C * H * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(b2, &h_b2, C * sizeof(float)));
}


template <int N, int M> void update_matrix(float dmatrix[N][M], float det[N][M], float lr, float reg) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            dmatrix[i][j] *= (1 - reg);
            dmatrix[i][j] -= lr * det[i][j]/TC ;
        }
    }
}

template <int N> void update_array(float darr[N], float det[N], float lr, float reg) {
    for (int i = 0; i < N; i++) {
        darr[i] *= (1 - reg);
        darr[i] -= lr * det[i]/TC;
    }
}

void update_mnist_model(float lr, float reg) {
    get_mnist_grad();
    printf("loss: %f, accuracy: %f\n", h_loss, float(h_count)/TNN);

    update_matrix<H, D>(h_fc1, h_g_d_fc1_w, lr, reg);
    update_array<H>(h_b1, h_g_d_fc1_b, lr, reg);
    update_matrix<C, H>(h_fc2, h_g_d_fc2_w, lr, reg);
    update_array<C>(h_b2, h_g_d_fc2_b, lr, reg);
    // randomDumpMatrixEle<C, H>(h_g_d_fc2_w, 10, lr);
    // randomDumpMatrixEle<H, D>(h_g_d_fc1_w, 10, 1.0);
    // dumpMatrixEle<C, H>(h_g_d_fc2_w);

    sync_mnist_model_to_gpu();
 
    reset_mnist_grad();
    //randomDumpMatrixEle<C, H>(h_fc2, 10, 1.0);
    //randomDumpMatrixEle<H, D>(h_fc1, 10, 1.0);

}

