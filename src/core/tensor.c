#include "tensor.h"

Tensor* create_tensor(int ndim, int* shape) {
    return create_quantized_tensor(ndim, shape, QUANT_FLOAT32);
}

Tensor* create_quantized_tensor(int ndim, int* shape, QuantMode mode) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    
    t->data = (tensor_t*)calloc(size, sizeof(tensor_t));
    t->quant_mode = mode;
    t->quant_params = (QuantParams){1.0f, 0, 0.0f, 0.0f};
    
    return t;
}

void free_tensor(Tensor* t) {
    free(t->data);
    free(t->shape);
    free(t);
}

void quantize_tensor(Tensor* t, QuantMode target_mode) {
    if (t->quant_mode == target_mode) return;
    
    int size = 1;
    for (int i = 0; i < t->ndim; i++) {
        size *= t->shape[i];
    }
    
    if (t->quant_mode != QUANT_FLOAT32) {
        dequantize_tensor(t);
    }
    
    if (target_mode != QUANT_FLOAT32) {
        t->quant_params = calculate_quant_params(t->f32_data, size, target_mode);
        
        if (target_mode == QUANT_INT8) {
            for (int i = 0; i < size; i++) {
                t->i8_data[i] = float_to_int8(t->f32_data[i], &t->quant_params);
            }
        } else if (target_mode == QUANT_INT4) {
            for (int i = 0; i < size; i += 2) {
                int8_t high = float_to_int8(t->f32_data[i], &t->quant_params) & 0xF;
                int8_t low = (i + 1 < size) ? 
                    float_to_int8(t->f32_data[i + 1], &t->quant_params) & 0xF : 0;
                t->i8_data[i/2] = pack_int4(high, low);
            }
        }
    }
    
    t->quant_mode = target_mode;
}

void dequantize_tensor(Tensor* t) {
    if (t->quant_mode == QUANT_FLOAT32) return;
    
    int size = 1;
    for (int i = 0; i < t->ndim; i++) {
        size *= t->shape[i];
    }
    
    float* temp = (float*)malloc(size * sizeof(float));
    
    if (t->quant_mode == QUANT_INT8) {
        for (int i = 0; i < size; i++) {
            temp[i] = int8_to_float(t->i8_data[i], &t->quant_params);
        }
    } else if (t->quant_mode == QUANT_INT4) {
        for (int i = 0; i < size; i += 2) {
            int8_t high, low;
            unpack_int4(t->i8_data[i/2], &high, &low);
            temp[i] = int8_to_float(high, &t->quant_params);
            if (i + 1 < size) {
                temp[i + 1] = int8_to_float(low, &t->quant_params);
            }
        }
    }
    
    memcpy(t->f32_data, temp, size * sizeof(float));
    free(temp);
    
    t->quant_mode = QUANT_FLOAT32;
    t->quant_params = (QuantParams){1.0f, 0, 0.0f, 0.0f};
}

void matmul(tensor_t* out, const tensor_t* a, const tensor_t* b, int m, int k, int n, QuantMode mode) {
    for (int i = 0; i < m * n; i++) {
        out[i].f32 = 0.0f;
    }

    if (mode == QUANT_FLOAT32) {
        const int BLOCK_SIZE = 32;
        const int SIMD_WIDTH = 8;
        
        int i0, j0;
        #pragma omp parallel for collapse(2) private(i0, j0)
        for (i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
            for (j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
                for (int k0 = 0; k0 < k; k0 += BLOCK_SIZE) {
                    int i_end = (i0 + BLOCK_SIZE < m) ? i0 + BLOCK_SIZE : m;
                    int j_end = (j0 + BLOCK_SIZE < n) ? j0 + BLOCK_SIZE : n;
                    int k_end = (k0 + BLOCK_SIZE < k) ? k0 + BLOCK_SIZE : k;
                    
                    for (int i = i0; i < i_end; i++) {
                        for (int j = j0; j < j_end; j++) {
                            __m256 sum_vec = _mm256_setzero_ps();
                            
                            for (int l = k0; l < k_end - (k_end - k0) % SIMD_WIDTH; l += SIMD_WIDTH) {
                                __m256 a_vec = _mm256_loadu_ps(&a[i * k + l].f32);
                                __m256 b_vec = _mm256_loadu_ps(&b[j * k + l].f32);
                                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                            }
                            
                            float sum_array[8];
                            _mm256_storeu_ps(sum_array, sum_vec);
                            float sum = 0.0f;
                            for (int l = 0; l < 8; l++) {
                                sum += sum_array[l];
                            }
                            
                            for (int l = k_end - (k_end - k0) % SIMD_WIDTH; l < k_end; l++) {
                                sum += a[i * k + l].f32 * b[j * k + l].f32;
                            }
                            
                            out[i * n + j].f32 += sum;
                        }
                    }
                }
            }
        }
    } else if (mode == QUANT_INT8) {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int32_t acc = 0;
                for (int l = 0; l < k; l++) {
                    acc += (int32_t)a[i * k + l].i8 * (int32_t)b[j * k + l].i8;
                }
                float scaled = (float)acc / 128.0f;
                out[i * n + j].i8 = (int8_t)clamp(scaled, -128.0f, 127.0f);
            }
        }
    } else {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int32_t acc = 0;
                for (int l = 0; l < k; l += 2) {
                    int8_t a_high, a_low, b_high, b_low;
                    unpack_int4(a[i * k + l/2].i4_packed, &a_high, &a_low);
                    unpack_int4(b[j * k + l/2].i4_packed, &b_high, &b_low);
                    
                    acc += (int32_t)a_high * (int32_t)b_high;
                    if (l + 1 < k) {
                        acc += (int32_t)a_low * (int32_t)b_low;
                    }
                }
                float scaled = (float)acc / 8.0f;
                out[i * n + j].i4_packed = pack_int4((int8_t)clamp(scaled, -8.0f, 7.0f), 0);
            }
        }
    }
}

void print_tensor_stats(const char* name, tensor_t* data, int size) {
    printf("%s:\n", name);
    printf("  Size: %d\n", size);
    
    if (data[0].f32 == data[0].f32) {  // Check if float (NaN check)
        double sum = 0.0;  // Use double for accumulation to avoid precision loss
        double sum_sq = 0.0;
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        
        for (int i = 0; i < size; i++) {
            float val = data[i].f32;
            sum += (double)val;
            sum_sq += (double)val * (double)val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        double size_f = (double)size;
        float mean = (float)(sum / size_f);
        float std = (float)sqrt((sum_sq / size_f) - (sum / size_f) * (sum / size_f));
        
        printf("  Mean: %.6f\n", mean);
        printf("  Std:  %.6f\n", std);
        printf("  Min:  %.6f\n", min_val);
        printf("  Max:  %.6f\n", max_val);
    } else {
        int hist[256] = {0};
        for (int i = 0; i < size; i++) {
            hist[(unsigned char)data[i].i8 + 128]++;
        }
        
        printf("  Histogram:\n");
        for (int i = 0; i < 256; i++) {
            if (hist[i] > 0) {
                printf("    %4d: %d\n", i - 128, hist[i]);
            }
        }
    }
} 