#include "embedding.h"

TokenEmbedding* create_token_embedding(int vocab_size, int d_model) {
    TokenEmbedding* te = (TokenEmbedding*)malloc(sizeof(TokenEmbedding));
    int embed_shape[] = {vocab_size, d_model};
    te->embed = create_tensor(2, embed_shape);
    return te;
}

void free_token_embedding(TokenEmbedding* te) {
    free_tensor(te->embed);
    free(te);
}

void token_embedding_forward(TokenEmbedding* te, int* input, tensor_t* output,
                           int batch_size, int seq_len, int d_model) {
    if (!te || !te->embed || !te->embed->data || !input || !output) {
        printf("Error: NULL pointer in token_embedding_forward\n");
        return;
    }

    if (d_model != te->embed->shape[1]) {
        printf("Error: d_model mismatch. Expected %d, got %d\n", te->embed->shape[1], d_model);
        return;
    }
    
    int total_tokens = batch_size * seq_len;
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < total_tokens; b++) {
        int batch_idx = b / seq_len;
        int seq_idx = b % seq_len;
        int token = input[b];
        if (token >= te->embed->shape[0]) {
            printf("Warning: Token %d at position (batch=%d, seq=%d) exceeds vocab size %d, using padding token\n",
                   token, batch_idx, seq_idx, te->embed->shape[0]);
            token = 0;  // padding token if it's not within the vocab
        }
        
        tensor_t* src = &te->embed->data[token * d_model];
        tensor_t* dst = &output[b * d_model];
        memcpy(dst, src, d_model * sizeof(tensor_t));
    }
} 