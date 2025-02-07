#ifndef SWARM_LAYERS_H
#define SWARM_LAYERS_H

#include "../core/tensor.h"
#include "linear.h"

typedef struct {
    Linear* mlp_fc1;
    Linear* mlp_fc2;
    Linear* gate_fc1;
    Linear* gate_fc2;
    float dropout_rate;
} LocalSwarmAggregator;

typedef struct {
    Linear* query;
    Linear* key;
    Linear* value;
    float scale;
    float dropout_rate;
} GlobalClusterAttention;

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

LocalSwarmAggregator* create_local_swarm_aggregator(int d_model);
void free_local_swarm_aggregator(LocalSwarmAggregator* lsa);
void local_swarm_aggregator_forward(LocalSwarmAggregator* lsa, tensor_t* x, tensor_t* output, int batch_size, int seq_len, int d_model);

GlobalClusterAttention* create_global_cluster_attention(int d_model);
void free_global_cluster_attention(GlobalClusterAttention* gca);
void global_cluster_attention_forward(GlobalClusterAttention* gca, tensor_t* x, tensor_t* output, int batch_size, int num_clusters, int d_model);

BroadcastUpdater* create_broadcast_updater(int d_model);
void free_broadcast_updater(BroadcastUpdater* bu);
void broadcast_updater_forward(BroadcastUpdater* bu, tensor_t* x, tensor_t* output, int batch_size, int seq_len, int d_model);

SwarmFormerLayer* create_swarmformer_layer(int d_model, int cluster_size, int T_local);
void free_swarmformer_layer(SwarmFormerLayer* sfl);
void swarmformer_layer_forward(SwarmFormerLayer* sfl, tensor_t* x, tensor_t* output,
                             int batch_size, int seq_len, int d_model);

#endif