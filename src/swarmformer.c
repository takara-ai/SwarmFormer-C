#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

typedef float tensor_t;

typedef struct {
    tensor_t* data;
    int* shape;
    int ndim;
} Tensor;

typedef struct {
    Tensor* weight;
    Tensor* bias;
} Linear;

typedef struct {
    Tensor* embed;
} TokenEmbedding;

typedef struct {
    Linear* query;
    Linear* key;
    Linear* value;
    float scale;
    float dropout_rate;
} GlobalClusterAttention;

typedef struct {
    Linear* mlp_fc1;
    Linear* mlp_fc2;
    Linear* gate_fc1;
    Linear* gate_fc2;
    float dropout_rate;
} LocalSwarmAggregator;

typedef struct {
    Linear* linear;
    Linear* gate_fc1;
    Linear* gate_fc2;
    float dropout_rate;
} BroadcastUpdater;

typedef struct {
    LocalSwarmAggregator* local_agg;
    GlobalClusterAttention* global_attn;
    BroadcastUpdater* broadcast;
    int cluster_size;
    int T_local;
} SwarmFormerLayer;

typedef struct {
    TokenEmbedding* embedding;
    SwarmFormerLayer** layers;
    Linear* classifier;
    int num_layers;
    float dropout_rate;
    int d_model;
    int seq_len;
} SwarmFormerModel;

void linear_forward(Linear* l, tensor_t* x, tensor_t* output, int batch_size, int in_features, int out_features);
void local_swarm_aggregator_forward(LocalSwarmAggregator* lsa, tensor_t* x, tensor_t* output, 
                                  int batch_size, int seq_len, int d_model);
void broadcast_updater_forward(BroadcastUpdater* bu, tensor_t* x, tensor_t* output,
                             int batch_size, int seq_len, int d_model);
void softmax(tensor_t* x, int size);
void token_embedding_forward(TokenEmbedding* te, int* input, tensor_t* output, 
                           int batch_size, int seq_len, int d_model);
void print_tensor_stats(const char* name, tensor_t* data, int size);

void global_cluster_attention_forward(GlobalClusterAttention* gca, tensor_t* x, tensor_t* output,
                          int batch_size, int num_clusters, int d_model) {
    tensor_t* q = (tensor_t*)malloc(batch_size * num_clusters * d_model * sizeof(tensor_t));
    tensor_t* k = (tensor_t*)malloc(batch_size * num_clusters * d_model * sizeof(tensor_t));
    tensor_t* v = (tensor_t*)malloc(batch_size * num_clusters * d_model * sizeof(tensor_t));
    tensor_t* attn_scores = (tensor_t*)malloc(batch_size * num_clusters * num_clusters * sizeof(tensor_t));

    linear_forward(gca->query, x, q, batch_size * num_clusters, d_model, d_model);
    linear_forward(gca->key, x, k, batch_size * num_clusters, d_model, d_model);
    linear_forward(gca->value, x, v, batch_size * num_clusters, d_model, d_model);

    for (int b = 0; b < batch_size; b++) {
        // Q * K^T / sqrt(d_model) (yikes)
        for (int i = 0; i < num_clusters; i++) {
            for (int j = 0; j < num_clusters; j++) {
                tensor_t sum = 0.0f;
                for (int d = 0; d < d_model; d++) {
                    sum += q[b * num_clusters * d_model + i * d_model + d] * 
                          k[b * num_clusters * d_model + j * d_model + d];
                }
                attn_scores[b * num_clusters * num_clusters + i * num_clusters + j] = sum * gca->scale;
            }
        }

        for (int i = 0; i < num_clusters; i++) {
            tensor_t* row = &attn_scores[b * num_clusters * num_clusters + i * num_clusters];

            tensor_t max_val = row[0];
            for (int j = 1; j < num_clusters; j++) {
                if (row[j] > max_val) max_val = row[j];
            }

            tensor_t sum = 0.0f;
            for (int j = 0; j < num_clusters; j++) {
                row[j] = expf(row[j] - max_val);
                sum += row[j];
            }

            for (int j = 0; j < num_clusters; j++) {
                row[j] /= sum;
            }
        }
        
        // (attn_scores * V) - weighted sums
        for (int i = 0; i < num_clusters; i++) {
            for (int d = 0; d < d_model; d++) {
                tensor_t sum = 0.0f;
                for (int j = 0; j < num_clusters; j++) {
                    sum += attn_scores[b * num_clusters * num_clusters + i * num_clusters + j] * 
                          v[b * num_clusters * d_model + j * d_model + d];
                }
                output[b * num_clusters * d_model + i * d_model + d] = sum;
            }
        }
    }
    
    free(q);
    free(k);
    free(v);
    free(attn_scores);
}

void swarmformer_layer_forward(SwarmFormerLayer* sfl, tensor_t* x, tensor_t* output,
                             int batch_size, int seq_len, int d_model) {
    tensor_t* local_out = (tensor_t*)malloc(batch_size * seq_len * d_model * sizeof(tensor_t));
    memcpy(local_out, x, batch_size * seq_len * d_model * sizeof(tensor_t));
    
    for (int t = 0; t < sfl->T_local; t++) {
        tensor_t* temp = (tensor_t*)malloc(batch_size * seq_len * d_model * sizeof(tensor_t));
        local_swarm_aggregator_forward(sfl->local_agg, local_out, temp, batch_size, seq_len, d_model);
        memcpy(local_out, temp, batch_size * seq_len * d_model * sizeof(tensor_t));
        free(temp);
    }
    
    int num_clusters = seq_len / sfl->cluster_size;
    tensor_t* cluster_reps = (tensor_t*)malloc(batch_size * num_clusters * d_model * sizeof(tensor_t));
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_clusters; c++) {
            for (int d = 0; d < d_model; d++) {
                float sum = 0.0f;
                for (int i = 0; i < sfl->cluster_size; i++) {
                    sum += local_out[(b * seq_len + c * sfl->cluster_size + i) * d_model + d];
                }
                cluster_reps[(b * num_clusters + c) * d_model + d] = sum / sfl->cluster_size;
            }
        }
    }
    
    tensor_t* global_out = (tensor_t*)malloc(batch_size * num_clusters * d_model * sizeof(tensor_t));
    global_cluster_attention_forward(sfl->global_attn, cluster_reps, global_out, batch_size, num_clusters, d_model);
    
    for (int i = 0; i < batch_size * num_clusters * d_model; i++) {
        global_out[i] = global_out[i] + cluster_reps[i];
    }
    
    tensor_t* broadcast_input = (tensor_t*)malloc(batch_size * seq_len * d_model * sizeof(tensor_t));
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_clusters; c++) {
            for (int i = 0; i < sfl->cluster_size; i++) {
                memcpy(&broadcast_input[(b * seq_len + c * sfl->cluster_size + i) * d_model],
                       &global_out[(b * num_clusters + c) * d_model],
                       d_model * sizeof(tensor_t));
            }
        }
    }
    
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        broadcast_input[i] = broadcast_input[i] + local_out[i];
    }
    
    broadcast_updater_forward(sfl->broadcast, broadcast_input, output, batch_size, seq_len, d_model);
    
    free(local_out);
    free(cluster_reps);
    free(global_out);
    free(broadcast_input);
}

void swarmformer_forward(SwarmFormerModel* model, int* input, tensor_t* output, int batch_size) {
    printf("\nStarting forward pass...\n");
    
    int embed_size = batch_size * model->seq_len * model->d_model;
    tensor_t* embedded = (tensor_t*)malloc(embed_size * sizeof(tensor_t));
    
    token_embedding_forward(model->embedding, input, embedded, batch_size, model->seq_len, model->d_model);
    printf("\nAfter embedding:\n");
    print_tensor_stats("Embedded input", embedded, embed_size);
    
    tensor_t* layer_input = embedded;
    tensor_t* layer_output = (tensor_t*)malloc(embed_size * sizeof(tensor_t));
    
    for (int i = 0; i < model->num_layers; i++) {
        printf("\nProcessing layer %d:\n", i);
        
        swarmformer_layer_forward(model->layers[i], layer_input, layer_output,
                                batch_size, model->seq_len, model->d_model);
        
        if (i < model->num_layers - 1) {
            tensor_t* temp = layer_input;
            layer_input = layer_output;
            layer_output = temp;
        }
    }
    
    tensor_t* pooled = (tensor_t*)malloc(batch_size * model->d_model * sizeof(tensor_t));
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < model->d_model; j++) {
            tensor_t sum = 0.0f;
            for (int i = 0; i < model->seq_len; i++) {
                sum += layer_output[b * model->seq_len * model->d_model + i * model->d_model + j];
            }
            pooled[b * model->d_model + j] = sum / model->seq_len;
        }
    }
    printf("\nAfter pooling:\n");
    print_tensor_stats("Pooled output", pooled, batch_size * model->d_model);
    
    linear_forward(model->classifier, pooled, output, batch_size, model->d_model, 2);
    printf("\nFinal logits:\n");
    print_tensor_stats("Logits", output, batch_size * 2);
    
    free(embedded);
    if (layer_input != embedded) {
        free(layer_input);
    }
    free(pooled);
} 