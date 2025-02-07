#include "activation.h"
#include "quantization.h"
#include <math.h>

ActivationTables activation_tables = {0};

void init_activation_tables(float input_scale, float output_scale) {
    if (activation_tables.initialized) return;
    
    const float sqrt_2_over_pi = (float)(sqrt(2.0 / M_PI));
    const float coeff = 0.044715f;
    
    for (int i = 0; i < 256; i++) {
        float x = ((float)i - 128.0f) * input_scale;
        float exp_val = expf(x);
        activation_tables.exp_lut[i] = (int8_t)((exp_val / output_scale) * 127.0f);
    }
    
    for (int i = 0; i < 16; i++) {
        float x = ((float)i - 8.0f) * input_scale;
        float exp_val = expf(x);
        activation_tables.exp_lut_4bit[i] = (int8_t)((exp_val / output_scale) * 7.0f);
    }
    
    for (int i = 0; i < 256; i++) {
        float x = ((float)i - 128.0f) * input_scale;
        float sigmoid_val = 1.0f / (1.0f + expf(-x));
        activation_tables.sigmoid_lut[i] = (int8_t)((sigmoid_val / output_scale) * 127.0f);
    }
    
    for (int i = 0; i < 16; i++) {
        float x = ((float)i - 8.0f) * input_scale;
        float sigmoid_val = 1.0f / (1.0f + expf(-x));
        activation_tables.sigmoid_lut_4bit[i] = (int8_t)((sigmoid_val / output_scale) * 7.0f);
    }
    
    for (int i = 0; i < 256; i++) {
        float x = ((float)i - 128.0f) * input_scale;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float gelu_val = 0.5f * x * (1.0f + tanhf(inner));
        activation_tables.gelu_lut[i] = (int8_t)((gelu_val / output_scale) * 127.0f);
    }
    
    for (int i = 0; i < 16; i++) {
        float x = ((float)i - 8.0f) * input_scale;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float gelu_val = 0.5f * x * (1.0f + tanhf(inner));
        activation_tables.gelu_lut_4bit[i] = (int8_t)((gelu_val / output_scale) * 7.0f);
    }
    
    activation_tables.initialized = true;
}

void relu_forward(tensor_t* x, int size, QuantMode mode) {
    switch (mode) {
        case QUANT_INT8:
            for (int i = 0; i < size; i++) {
                x[i].i8 = (x[i].i8 > 0) ? x[i].i8 : 0;
            }
            break;
            
        case QUANT_INT4: {
            for (int i = 0; i < size; i++) {
                int8_t upper = (x[i].i8 >> 4) & 0xF;
                int8_t lower = x[i].i8 & 0xF;
                
                upper = (upper > 0) ? upper : 0;
                lower = (lower > 0) ? lower : 0;
                
                x[i].i8 = (upper << 4) | lower;
            }
            break;
        }
            
        default:  // QUANT_FLOAT32
            for (int i = 0; i < size; i++) {
                x[i].f32 = (x[i].f32 > 0) ? x[i].f32 : 0;
            }
            break;
    }
}

void sigmoid_forward(tensor_t* x, int size, QuantMode mode) {
    switch (mode) {
        case QUANT_INT8:
            for (int i = 0; i < size; i++) {
                x[i].i8 = activation_tables.sigmoid_lut[x[i].i8 + 128];
            }
            break;
            
        case QUANT_INT4: {
            for (int i = 0; i < size; i++) {
                int8_t upper = (x[i].i8 >> 4) & 0xF;
                int8_t lower = x[i].i8 & 0xF;
                
                upper = activation_tables.sigmoid_lut_4bit[upper];
                lower = activation_tables.sigmoid_lut_4bit[lower];
                
                x[i].i8 = (upper << 4) | lower;
            }
            break;
        }
            
        default:  // QUANT_FLOAT32
            for (int i = 0; i < size; i++) {
                x[i].f32 = 1.0f / (1.0f + expf(-x[i].f32));
            }
            break;
    }
}

void gelu_forward(tensor_t* x, int size, QuantMode mode) {
    switch (mode) {
        case QUANT_INT8:
            for (int i = 0; i < size; i++) {
                x[i].i8 = activation_tables.gelu_lut[x[i].i8 + 128];
            }
            break;
            
        case QUANT_INT4: {
            for (int i = 0; i < size; i++) {
                int8_t upper = (x[i].i8 >> 4) & 0xF;
                int8_t lower = x[i].i8 & 0xF;
                
                upper = activation_tables.gelu_lut_4bit[upper];
                lower = activation_tables.gelu_lut_4bit[lower];
                
                x[i].i8 = (upper << 4) | lower;
            }
            break;
        }
            
        default:  // QUANT_FLOAT32
            for (int i = 0; i < size; i++) {
                x[i].f32 = compute_gelu(x[i].f32);
            }
            break;
    }
}

void softmax_forward(tensor_t* x, int size, QuantMode mode) {
    switch (mode) {
        case QUANT_INT8: {
            int8_t max_val = x[0].i8;
            for (int i = 1; i < size; i++) {
                if (x[i].i8 > max_val) max_val = x[i].i8;
            }

            int32_t sum = 0;
            for (int i = 0; i < size; i++) {
                x[i].i8 = activation_tables.exp_lut[(x[i].i8 - max_val) + 128];
                sum += x[i].i8;
            }
            
            for (int i = 0; i < size; i++) {
                x[i].i8 = (int8_t)((x[i].i8 * 127) / sum);
            }
            break;
        }
            
        case QUANT_INT4: {
            for (int i = 0; i < size; i += 2) {
                int8_t upper1 = (x[i].i8 >> 4) & 0xF;
                int8_t lower1 = x[i].i8 & 0xF;
                int8_t upper2 = (x[i+1].i8 >> 4) & 0xF;
                int8_t lower2 = x[i+1].i8 & 0xF;
                
                int8_t max_val = upper1;
                if (lower1 > max_val) max_val = lower1;
                if (upper2 > max_val) max_val = upper2;
                if (lower2 > max_val) max_val = lower2;
                
                int16_t sum = 0;
                upper1 = activation_tables.exp_lut_4bit[(upper1 - max_val) + 8];
                lower1 = activation_tables.exp_lut_4bit[(lower1 - max_val) + 8];
                upper2 = activation_tables.exp_lut_4bit[(upper2 - max_val) + 8];
                lower2 = activation_tables.exp_lut_4bit[(lower2 - max_val) + 8];
                
                sum = upper1 + lower1 + upper2 + lower2;
                
                upper1 = (int8_t)((upper1 * 7) / sum);
                lower1 = (int8_t)((lower1 * 7) / sum);
                upper2 = (int8_t)((upper2 * 7) / sum);
                lower2 = (int8_t)((lower2 * 7) / sum);
                
                x[i].i8 = (upper1 << 4) | lower1;
                x[i+1].i8 = (upper2 << 4) | lower2;
            }
            break;
        }
            
        default:  // QUANT_FLOAT32
            float max_val = x[0].f32;
            for (int i = 1; i < size; i++) {
                if (x[i].f32 > max_val) max_val = x[i].f32;
            }
            
            float sum = 0.0f;
            for (int i = 0; i < size; i++) {
                x[i].f32 = expf(x[i].f32 - max_val);
                sum += x[i].f32;
            }
            
            for (int i = 0; i < size; i++) {
                x[i].f32 /= sum;
            }
            break;
    }
} 