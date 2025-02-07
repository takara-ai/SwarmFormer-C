#include "loader.h"

Tensor* load_tensor(FILE* f) {
    int ndim;
    fread(&ndim, sizeof(int), 1, f);
    
    int* shape = (int*)malloc(ndim * sizeof(int));
    fread(shape, sizeof(int), ndim, f);
    
    Tensor* t = create_tensor(ndim, shape);
    
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    
    fread(t->data, sizeof(tensor_t), size, f);
    
    free(shape);
    return t;
}

void load_linear_layer(FILE* f, Linear* layer) {
    free_tensor(layer->weight);
    free_tensor(layer->bias);
    
    layer->weight = load_tensor(f);
    layer->bias = load_tensor(f);
}

void load_local_swarm_aggregator(FILE* f, LocalSwarmAggregator* lsa) {
    load_linear_layer(f, lsa->mlp_fc1);
    load_linear_layer(f, lsa->mlp_fc2);
    load_linear_layer(f, lsa->gate_fc1);
    load_linear_layer(f, lsa->gate_fc2);
}

void load_global_cluster_attention(FILE* f, GlobalClusterAttention* gca) {
    load_linear_layer(f, gca->query);
    load_linear_layer(f, gca->key);
    load_linear_layer(f, gca->value);
}

void load_broadcast_updater(FILE* f, BroadcastUpdater* bu) {
    load_linear_layer(f, bu->linear);
    load_linear_layer(f, bu->gate_fc1);
    load_linear_layer(f, bu->gate_fc2);
}

void load_swarmformer_layer(FILE* f, SwarmFormerLayer* layer) {
    load_local_swarm_aggregator(f, layer->local_agg);
    load_global_cluster_attention(f, layer->global_attn);
    load_broadcast_updater(f, layer->broadcast);
}

SwarmFormerModel* load_swarmformer_model(const char* weights_path) {
    if (!weights_path) {
        printf("Error: weights_path is NULL\n");
        return NULL;
    }
    
    FILE* f = fopen(weights_path, "rb");
    if (!f) {
        printf("Error: Could not open weights file %s\n", weights_path);
        return NULL;
    }
    
    int vocab_size, d_model, seq_len, cluster_size, num_layers, T_local;
    if (fread(&vocab_size, sizeof(int), 1, f) != 1 ||
        fread(&d_model, sizeof(int), 1, f) != 1 ||
        fread(&seq_len, sizeof(int), 1, f) != 1 ||
        fread(&cluster_size, sizeof(int), 1, f) != 1 ||
        fread(&num_layers, sizeof(int), 1, f) != 1 ||
        fread(&T_local, sizeof(int), 1, f) != 1) {
        printf("Error: Failed to read model configuration\n");
        fclose(f);
        return NULL;
    }
    
    SwarmFormerModel* model = create_swarmformer_model(
        vocab_size, d_model, seq_len, cluster_size, num_layers, T_local
    );
    
    if (!model) {
        printf("Error: Failed to create model\n");
        fclose(f);
        return NULL;
    }
    
    free_tensor(model->embedding->embed);
    model->embedding->embed = load_tensor(f);
    
    for (int i = 0; i < num_layers; i++) {
        load_swarmformer_layer(f, model->layers[i]);
    }
    
    load_linear_layer(f, model->classifier);
    
    fclose(f);
    return model;
} 