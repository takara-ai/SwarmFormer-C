#include "linear.h"

Linear* create_linear(int in_features, int out_features) {
    Linear* l = (Linear*)malloc(sizeof(Linear));
    int weight_shape[] = {out_features, in_features};
    int bias_shape[] = {out_features};
    l->weight = create_tensor(2, weight_shape);
    l->bias = create_tensor(1, bias_shape);
    return l;
}

void free_linear(Linear* l) {
    free_tensor(l->weight);
    free_tensor(l->bias);
    free(l);
}

void linear_forward(Linear* l, tensor_t* x, tensor_t* output, int batch_size, int in_features, int out_features) {
    matmul(output, x, l->weight->data, batch_size, in_features, out_features, l->weight->quant_mode);
    
    if (l->weight->quant_mode == QUANT_FLOAT32) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < out_features; j++) {
                output[i * out_features + j].f32 = output[i * out_features + j].f32 + l->bias->data[j].f32;
            }
        }
    } else if (l->weight->quant_mode == QUANT_INT8) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < out_features; j++) {
                int32_t val = output[i * out_features + j].i8 + l->bias->data[j].i8;
                output[i * out_features + j].i8 = (int8_t)clamp((float)val / 128.0f, -128.0f, 127.0f);
            }
        }
    } else { // QUANT_INT4
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < out_features; j++) {
                int32_t val = output[i * out_features + j].i4_packed + (l->bias->data[j].i4_packed & 0xF);
                output[i * out_features + j].i4_packed = pack_int4((int8_t)clamp((float)val / 8.0f, -8.0f, 7.0f), 0);
            }
        }
    }
} 