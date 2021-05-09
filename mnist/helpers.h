#ifndef MY_HELPERS_H_
#define MY_HELPERS_H_

void init_mnist_network();
void update_mnist_model(float, float);
template <int N, int M> void dumpMatrixEle(float layer [][M]);
void dumpArray(float arr[], int);
#endif