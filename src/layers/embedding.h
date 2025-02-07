#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../core/tensor.h"

typedef struct {
    Tensor* embed;
} TokenEmbedding;

TokenEmbedding* create_token_embedding(int vocab_size, int d_model);
void free_token_embedding(TokenEmbedding* te);
void token_embedding_forward(TokenEmbedding* te, int* input, tensor_t* output, 
                           int batch_size, int seq_len, int d_model);

#endif