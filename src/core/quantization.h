#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stdint.h>

typedef enum {
    QUANT_FLOAT32,
    QUANT_INT8,
    QUANT_INT4
} QuantMode;

#define FIXED_POINT_BITS 8
#define FIXED_POINT_ONE (1 << FIXED_POINT_BITS)
#define FIXED_POINT_HALF (FIXED_POINT_ONE >> 1)

typedef struct {
    float scale;
    int32_t zero_point;
    float min_val;
    float max_val;
} QuantParams;

float clamp(float x, float min_val, float max_val);

int8_t float_to_int8(float x, const QuantParams* params);
float int8_to_float(int8_t x, const QuantParams* params);

int8_t pack_int4(int8_t high, int8_t low);
void unpack_int4(int8_t packed, int8_t* high, int8_t* low);

int32_t fixed_mul(int32_t a, int32_t b);
int32_t fixed_div(int32_t a, int32_t b);

QuantParams calculate_quant_params(const float* data, int size, QuantMode mode);

#endif 