#ifndef LOADER_H
#define LOADER_H

#include "../model/swarmformer.h"

SwarmFormerModel* load_swarmformer_model(const char* weights_path);
Tensor* load_tensor(FILE* f);
void load_linear_layer(FILE* f, Linear* layer);
void load_local_swarm_aggregator(FILE* f, LocalSwarmAggregator* lsa);
void load_global_cluster_attention(FILE* f, GlobalClusterAttention* gca);
void load_broadcast_updater(FILE* f, BroadcastUpdater* bu);
void load_swarmformer_layer(FILE* f, SwarmFormerLayer* layer);

#endif