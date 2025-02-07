#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include "quantization.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _MSC_VER
#define ALIGN(x) __declspec(align(x))
#define RESTRICT __restrict
#else
#define ALIGN(x) __attribute__((aligned(x)))
#define RESTRICT restrict
#endif

#define ALIGN_SIZE 32

typedef union {
    float f32;
    int8_t i8;
    int8_t i4_packed;
} ALIGN(4) tensor_t;

typedef struct {
    union {
        tensor_t* RESTRICT data;
        float* RESTRICT f32_data;
        int8_t* RESTRICT i8_data;
    };
    int* RESTRICT shape;
    int ndim;
    QuantMode quant_mode;
    QuantParams quant_params;
} Tensor;

Tensor* create_tensor(int ndim, int* shape);
Tensor* create_quantized_tensor(int ndim, int* shape, QuantMode mode);
void free_tensor(Tensor* t);

void quantize_tensor(Tensor* t, QuantMode target_mode);
void dequantize_tensor(Tensor* t);

void print_tensor_stats(const char* name, tensor_t* RESTRICT data, int size);
void matmul(tensor_t* RESTRICT out, const tensor_t* RESTRICT a, const tensor_t* RESTRICT b, 
            int m, int k, int n, QuantMode mode);

#endif