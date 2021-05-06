
# include <cuda_runtime.h>
# include "layers.h"

int main(int argc, char **argv) {
  
  dim3 block(128, 1);
  dim3 grid(14, 1);
  train_mnist<33><<< grid, block >>>();
}