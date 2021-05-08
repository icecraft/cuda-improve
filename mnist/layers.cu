
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
# include "shared.h"
# include "dataset.h"

template <int N, int M> __device__ void affine_forward_fc1(float data[M], float ret[N]) {
#pragma unroll
    for (int i =0; i < N; i++) {
        ret[i] = 0;
        for (int j = 0; j < M; j++) {
            ret[i] += data[j] * fc1[i][j] + b1[i];
        }
    }
}

template <int N, int M> __device__ void affine_forward_fc2(float data[M], float ret[N]) {
#pragma unroll
    for (int i =0; i < N; i++) {
        ret[i] = 0;
        for (int j = 0; j < M; j++) {
            ret[i] += data[j] * fc2[i][j] + b2[i];
        }
    }
}

template <int N, int M> __device__ void affine_backward_fc1(float data[], float dout[], float ddata[], float dbias[], float dlayer[][M]) {
// get deviration of input data
#pragma unroll
for (int i=0; i < M; i++) {
    ddata[i] = 0;
    for (int j=0; j < N; j++) {
        ddata[i] += dout[j] * fc1[j][i];
    }
}

// get deviration of bias
#pragma unroll
for (int i=0; i < N; i++) {
    dbias[i] = dout[i];
}

// get deviration of matrix
#pragma unroll
for (int i=0; i < N; i++) {
    for (int j=0; j < M; j++) {
        dlayer[i][j] = dout[i] * data[j];
    }
}
}

template <int N, int M> __device__ void affine_backward_fc2(float data[], float dout[], float ddata[], float dbias[], float dlayer[][M]) {
// get deviration of input data
#pragma unroll
for (int i=0; i < M; i++) {
    ddata[i] = 0;
    for (int j=0; j < N; j++) {
        ddata[i] += dout[j] * fc2[j][i];
    }
}

// get deviration of bias
#pragma unroll
for (int i=0; i < N; i++) {
    dbias[i] = dout[i];
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
        grad[i] =  expf(data[i]/mm);
        esum += grad[i];
    }
#pragma unroll 
    for (int i=0; i < N; i++) {
        grad[i] = grad[i]/esum;
    }

    *loss = -logf(grad[label]);
    grad[label] -= 1;
}

template <int DATA_PER_THREAD> __global__ void train_mnist_cuda(void) {
    int seq = blockIdx.x * blockDim.x + threadIdx.x;

    float ret_fc1[H], ret_fc2[C], ret_relu[H], loss, total_loss=0;
    float d_softmax[C], d_relu[H], d_fc2_w[C][H], d_fc2_b[C], d_fc2_data[C], d_fc1_w[H][D], d_fc1_b[H], d_fc1_data[H];
    float s_d_fc2_w[C][H]={0}, s_d_fc2_b[C]={0}, s_d_fc1_w[H][D]={0}, s_d_fc1_b[H]={0};

#pragma unroll 
    for (int i=seq*DATA_PER_THREAD; i < (seq+1)*DATA_PER_THREAD; i++) {
        // forward
        affine_forward_fc1<H, D>(MNIST_data[i], ret_fc1);
        relu_forward<H>(ret_fc1, ret_relu);
        affine_forward_fc2<C, H>(ret_relu, ret_fc2);

        // get loss and grad
        softmax_loss<C>(ret_fc2, MNIST_label[i], &loss, d_softmax);
        total_loss += loss;
        // backward
       
        affine_backward_fc2<C, H>(ret_relu, d_softmax, d_fc2_data, d_fc2_b, d_fc2_w);
        relu_backward<H>(ret_fc1, d_fc2_data, d_relu);
        affine_backward_fc1<H, D>(MNIST_data[i], d_relu, d_fc1_data, d_fc1_b, d_fc1_w);

        // sum grad and bias
        for (int j=0; j<H; j++) {
            for (int k=0; k<D; k++) {
                s_d_fc1_w[j][k] += d_fc1_w[j][k]/NN;
            }
            s_d_fc1_b[j] += d_fc1_b[j]/NN;
        }

            // sum grad and bias
        for (int j=0; j<C; j++) {
            for (int k=0; k<H; k++) {
                s_d_fc2_w[j][k] += d_fc2_w[j][k]/NN;
            }
            s_d_fc2_b[j] += d_fc2_b[j]/NN;
        }
    }

    // cuda write need lock ?
    // sum fc1 network grad
#pragma unroll 
    for (int j=0; j<H; j++) {
        for (int k=0; k<D; k++) {
              __threadfence();
             d_g_d_fc1_w[j][k] += s_d_fc1_w[j][k];
        }
          __threadfence();
        d_g_d_fc1_b[j] += s_d_fc1_b[j];
    }

    // sum fc2 network grad
#pragma unroll 
    for (int j=0; j<C; j++) {
        for (int k=0; k<H; k++) {
              __threadfence();
            d_g_d_fc2_w[j][k] += s_d_fc2_w[j][k];
        }
          __threadfence();
        d_g_d_fc2_b[j] += s_d_fc2_b[j];
    }
  __threadfence();
    d_loss += total_loss;
  return;
}


void train_mnist() {
    
    // load mnist data to gpu
    load_mnist();
    checkCudaErrors(cudaMemcpyToSymbol(MNIST_data, &train_image, NN * D * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(MNIST_label, &train_label, NN * sizeof(int)));

    // random initialize fc1, fc2
    init_mnist_network();

    dim3 block(128, 1); // 128
    dim3 grid(14, 1); // 14

    // train 10 epoch for test
    for (int i=0; i<1000; i++) {
        train_mnist_cuda<33><<< grid, block >>>(); 
        cudaDeviceSynchronize(); 
        update_mnist_model(0.0001);
    }
}