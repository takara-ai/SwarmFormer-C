#ifndef LINEAR_H
#define LINEAR_H

#include "../core/tensor.h"

typedef struct {
    Tensor* weight;
    Tensor* bias;
} Linear;

Linear* create_linear(int in_features, int out_features);
void free_linear(Linear* l);
void linear_forward(Linear* l, tensor_t* x, tensor_t* output, int batch_size, int in_features, int out_features);

#endif