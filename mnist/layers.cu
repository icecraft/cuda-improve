
// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

# include "config.h"
# include "helpers.h"


template <int N, int M> __device__ void affine_forward(float layer[N][M], float bias[N], float data[M], float ret[N]) {
#pragma unroll
    for (int i =0; i < N; i++) {
        ret[i] = 0;
        for (int j = 0; j < M; j++) {
            ret[i] += data[j] * layer[i][j] + bias[i];
        }
    }
}

template <int N, int M> __device__ void affine_backward(float layer[N][M], float bias[N], float data[M], float dout[N], float ddata[M], float dbias[N], float dlayer[N][M]) {
// get deviration of input data
#pragma unroll
for (int i=0; i < M; i++) {
    ddata[i] = 0;
    for (int j=0; j < N; j++) {
        ddata[i] += dout[j] * layer[j][i];
    }
}

// get deviration of bias
#pragma unroll
for (int i=0; i < N; i++) {
    dbias[i] = 0;
    for (int j=0; j < M; j++) {
        dbias[i] += dout[i];
    }
}

// get deviration of matrix
#pragma unroll
for (int i=0; i < N; i++) {
    for (int j=0; j < M; j++) {
        dlayer[i][j] = dout[i] * data[j];
    }
}
}

template <int N> __device__ void relu_forward(float data[], float ret[]) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        ret[i] = fmaxf(0.0, data[i]);
    }
}

template <int N> __device__ void relu_backward(float data[], float dout[], float ret[]) {
#pragma unroll 
for (int i=0; i < N; i++) {
    ret[i] = data[i] > 0 ? dout[i]:0;
}
}

template <int N> __device__ void softmax_loss(float data[], int label, float *loss, float grad[]) {
float mm = data[0];
#pragma unroll
    for (int i=0; i < N; i++) {
        mm = fmaxf(mm, data[i]);
    }

float esum = 0;
#pragma unroll
    for (int i=0; i < N; i++) {
        grad[i] =  expf(data[i]-mm);
        esum += grad[i];
    }

#pragma unroll 
    for (int i=0; i < N; i++) {
        grad[i] = grad[i]/esum -1;
    }
    *loss = logf(esum/grad[label]);
}

template <int DATA_PER_THREAD> __global__ void train_mnist_cuda(void) {
    int seq = blockIdx.x * blockDim.x + threadIdx.x;

    float ret_fc1[H], ret_fc2[C], ret_relu[H], loss;
    float d_softmax[C], d_relu[H], d_fc2_w[C][H], d_fc2_b[C], d_fc2_data[C], d_fc1_w[H][D], d_fc1_b[H], d_fc1_data[H];
    float s_d_fc2_w[C][H]={0}, s_d_fc2_b[C]={0}, s_d_fc1_w[H][D]={0}, s_d_fc1_b[H]={0};

#pragma unroll 
    for (int i=seq*DATA_PER_THREAD; i < (seq+1)*DATA_PER_THREAD; i++) {
        // forward
        affine_forward<H, D>(fc1, b1, MNIST_data[i], ret_fc1);
        relu_forward<H>(ret_fc1, ret_relu);
        affine_forward<C, H>(fc2, b2, ret_relu, ret_fc2);

        // get loss and grad
        softmax_loss<C>(ret_fc2, MNIST_label[i], &loss, d_softmax);

        // backward
        affine_backward<C, H>(fc2, b2, ret_relu, d_softmax, d_fc2_data, d_fc2_b, d_fc2_w);
        relu_backward<H>(ret_fc1, d_fc2_data, d_relu);
        affine_backward<H, D>(fc1, b1, MNIST_data[i], d_relu, d_fc1_data, d_fc1_b, d_fc1_w);

        // sum grad and bias
        for (int j=0; j<H; j++) {
            for (int k=0; k<D; k++) {
                s_d_fc1_w[j][k] += d_fc1_w[j][k]/N;
            }
            s_d_fc1_b[j] += d_fc1_b[j]/N;
        }

            // sum grad and bias
        for (int j=0; j<C; j++) {
            for (int k=0; k<H; k++) {
                s_d_fc2_w[j][k] += d_fc2_w[j][k]/N;
            }
            s_d_fc2_b[j] += d_fc2_b[j]/N;
        }
    }
  return;
}


void train_mnist() {
    
    // load mnist data to gpu

    // random initialize fc1, fc2
    init_mnist_network();

    dim3 block(128, 1);
    dim3 grid(14, 1);

    // train 10 epoch for test
    for (int i=0; i<1; i< 10) {
        train_mnist_cuda<33><<< grid, block >>>();  
        update_mnist_model();
    }
}