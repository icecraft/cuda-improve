# MNIST Example with the PyTorch C++ Frontend

This folder contains an example of training a computer vision model to recognize
digits in images from the MNIST dataset, using the PyTorch C++ frontend.

The entire training code is contained in `mnist.cpp`.

To build the code, run the following commands from your terminal:

## usage 

```shell
    meson build
    ninja -C build
    build/mnist
```

## result

```shell
Training on CPU.
Train Epoch: 1 [59584/60000] Loss: 0.1794
Test set: Average loss: 0.4296 | Accuracy: 0.884
Train Epoch: 2 [59584/60000] Loss: 0.1092
Test set: Average loss: 0.3396 | Accuracy: 0.904
Train Epoch: 3 [59584/60000] Loss: 0.0883
Test set: Average loss: 0.3092 | Accuracy: 0.913
Train Epoch: 4 [59584/60000] Loss: 0.0794
Test set: Average loss: 0.2911 | Accuracy: 0.917
Train Epoch: 5 [59584/60000] Loss: 0.0753
Test set: Average loss: 0.2768 | Accuracy: 0.921
Train Epoch: 6 [59584/60000] Loss: 0.0734
Test set: Average loss: 0.2642 | Accuracy: 0.924
Train Epoch: 7 [59584/60000] Loss: 0.0722
Test set: Average loss: 0.2526 | Accuracy: 0.928
Train Epoch: 8 [59584/60000] Loss: 0.0716
Test set: Average loss: 0.2415 | Accuracy: 0.930
Train Epoch: 9 [59584/60000] Loss: 0.0724
Test set: Average loss: 0.2312 | Accuracy: 0.933
Train Epoch: 10 [59584/60000] Loss: 0.0724
Test set: Average loss: 0.2219 | Accuracy: 0.935
Train Epoch: 11 [59584/60000] Loss: 0.0728
Test set: Average loss: 0.2135 | Accuracy: 0.937
```

## 网络结构

```shell
affine layer (784*30) ---> relu ---> affine layer (30 * 10) ----> softmax
```

## 总结

* 性能远超 cuda 版本的 mnist（准确率最高只有 40%），这是本例 batch size 为 64， cuda impl 中的 batch size 为 59136。 即本实现中 net 的更新次数是 cuda impl 中的 624 倍。 cuda impl 没有实现正则化项，batch norm 等功能导致随之 epoch 增大，准确率会下降
