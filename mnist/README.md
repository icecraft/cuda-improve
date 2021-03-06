
# mnist

## usage

```bash
meson build
ninja -C build
```

## draft

```text
train data num: 59136 = 128 (threads) * 14 (sms) * 33 (the nums per thread processed)
test data num: 9600 =  128 * 15(sms) * 5 (every thread only process 5 data! what a amazing day!)


two layer net: input - fully connected layer - ReLU - fully connected layer - softmax
                       (D, H)                          (H, C)             


workflow:

init env                  ( cpu )
random generate networks  ( cpu )
transfer train data to gpu


## train 
for every epoch:
    load network to cuda constant memory
    
    for all data:
        one thread iter and cal loss
    aggerate loss on the first thread on warp (if can)
    
    cudaDeviceSynchronize();

    cpu backward and update networks

printf("train time : %d, loss: %f, train data");

## test 
cuda_free train data
transfer test data to gpu
load networks (trained, aka. model)
for item in test_data:
    loss += nn(item)
cudaDeviceSynchronize();

print("loss: %f")
```

## TODO

* RELU cuda impl

* softmax cuda impl

* fully connected layer cuda impl

* compose the network

* use mnist dataset

* use cifar10 dataset

* use dynamic parallelism  (p1 level)

* use nvprof

## 总结

* 本实现性能不好，L1 和 L2 cache、shared memory 均不能起到有效的作用。 constant memory 完全没有被使用到
* bugs:
  * __device__ float arr[100] 中的 arr 是符号，不能把 arr 当作指针
  * cuda function 中 auto variable 占用空间过大会导致 kernel 启动错误（错误信息没有没有任何 debug 意义）
  
