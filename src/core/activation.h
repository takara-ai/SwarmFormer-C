#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include "tensor.h"
#include "quantization.h"

typedef struct {
    int8_t exp_lut[256];
    int8_t exp_lut_4bit[16];
    int8_t sigmoid_lut[256];
    int8_t sigmoid_lut_4bit[16];
    int8_t gelu_lut[256];
    int8_t gelu_lut_4bit[16];
    bool initialized;
} ActivationTables;

extern ActivationTables activation_tables;

void init_activation_tables(float input_scale, float output_scale);

static inline float compute_exp(float x) {
    return expf(x);
}

static inline float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float compute_gelu(float x) {
    const float sqrt_2_over_pi = sqrt(2.0f / M_PI);
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline int8_t quantize_activation(float x, const int8_t* lut, int lut_size, float scale) {
    int idx = (int)((x / scale) * (lut_size / 2) + (lut_size / 2));
    if (idx < 0) idx = 0;
    if (idx >= lut_size) idx = lut_size - 1;
    return lut[idx];
}

void relu_forward(tensor_t* x, int size, QuantMode mode);
void sigmoid_forward(tensor_t* x, int size, QuantMode mode);
void gelu_forward(tensor_t* x, int size, QuantMode mode);
void softmax_forward(tensor_t* x, int size, QuantMode mode);

#endif 