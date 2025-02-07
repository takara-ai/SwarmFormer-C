#ifndef SWARMFORMER_H
#define SWARMFORMER_H

#include "../layers/linear.h"
#include "../layers/embedding.h"
#include "../layers/swarm_layers.h"
#include <stdbool.h>

typedef struct {
    TokenEmbedding* embedding;
    SwarmFormerLayer** layers;
    Linear* classifier;
    int num_layers;
    float dropout_rate;
    int d_model;
    int seq_len;
} SwarmFormerModel;

SwarmFormerModel* create_swarmformer_model(int vocab_size, int d_model, int seq_len,
                                         int cluster_size, int num_layers, int T_local);
void free_swarmformer_model(SwarmFormerModel* model);
void swarmformer_forward(SwarmFormerModel* model, int* input, tensor_t* output, int batch_size, bool verbose);

#endif