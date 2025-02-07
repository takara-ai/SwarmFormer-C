#include "quantization.h"
#include <math.h>
#include <float.h>

float clamp(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

int8_t float_to_int8(float x, const QuantParams* params) {
    float scaled = x * params->scale + params->zero_point;
    int val = (int)roundf(scaled);
    return (int8_t)clamp((float)val, -128.0f, 127.0f);
}

float int8_to_float(int8_t x, const QuantParams* params) {
    return (float)((float)x - params->zero_point) / params->scale;
}

int8_t pack_int4(int8_t high, int8_t low) {
    return (high << 4) | (low & 0x0F);
}

void unpack_int4(int8_t packed, int8_t* high, int8_t* low) {
    *high = (packed >> 4) & 0x0F;
    *low = packed & 0x0F;
}

int32_t fixed_mul(int32_t a, int32_t b) {
    int64_t result = (int64_t)a * (int64_t)b;
    return (int32_t)((result + FIXED_POINT_HALF) >> FIXED_POINT_BITS);
}

int32_t fixed_div(int32_t a, int32_t b) {
    int64_t temp = (int64_t)a << FIXED_POINT_BITS;
    return (int32_t)((temp + (b >> 1)) / b);
}

QuantParams calculate_quant_params(const float* data, int size, QuantMode mode) {
    QuantParams params = {0};
    
    if (mode == QUANT_INT8) {
        float min_val = data[0];
        float max_val = data[0];
        
        for (int i = 1; i < size; i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        
        params.scale = (max_val - min_val) / 255.0f;
        params.zero_point = (int32_t)roundf(-min_val / params.scale);
        params.min_val = min_val;
        params.max_val = max_val;
    } else if (mode == QUANT_INT4) {
        float min_val = data[0];
        float max_val = data[0];
        
        for (int i = 1; i < size; i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        
        params.scale = (max_val - min_val) / 15.0f;
        params.zero_point = (int32_t)roundf(-min_val / params.scale);
        params.min_val = min_val;
        params.max_val = max_val;
    }
    
    return params;
} 