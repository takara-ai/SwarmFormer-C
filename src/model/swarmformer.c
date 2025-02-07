#include "swarmformer.h"
#include "../core/activation.h"

SwarmFormerModel* create_swarmformer_model(int vocab_size, int d_model, int seq_len,
                                         int cluster_size, int num_layers, int T_local) {
    SwarmFormerModel* model = (SwarmFormerModel*)malloc(sizeof(SwarmFormerModel));
    model->embedding = create_token_embedding(vocab_size, d_model);
    model->layers = (SwarmFormerLayer**)malloc(num_layers * sizeof(SwarmFormerLayer*));
    
    for (int i = 0; i < num_layers; i++) {
        model->layers[i] = create_swarmformer_layer(d_model, cluster_size, T_local);
    }
    
    model->classifier = create_linear(d_model, 2);
    model->num_layers = num_layers;
    model->dropout_rate = 0.4f;
    model->d_model = d_model;
    model->seq_len = seq_len;
    
    return model;
}

void free_swarmformer_model(SwarmFormerModel* model) {
    free_token_embedding(model->embedding);
    for (int i = 0; i < model->num_layers; i++) {
        free_swarmformer_layer(model->layers[i]);
    }
    free(model->layers);
    free_linear(model->classifier);
    free(model);
}

void swarmformer_forward(SwarmFormerModel* model, int* input, tensor_t* output, int batch_size, bool verbose) {
    int embed_size = batch_size * model->seq_len * model->d_model;
    
    tensor_t* embedded = (tensor_t*)malloc(embed_size * sizeof(tensor_t));
    token_embedding_forward(model->embedding, input, embedded, batch_size, model->seq_len, model->d_model);
    
    if (verbose) {
        print_tensor_stats("Embedding output", embedded, embed_size);
    }
    
    tensor_t* layer_input = embedded;
    tensor_t* layer_output = NULL;
    tensor_t* temp_buffer = (tensor_t*)malloc(embed_size * sizeof(tensor_t));
    
    for (int i = 0; i < model->num_layers; i++) {
        if (verbose) {
            printf("\nLayer %d:\n", i);
        }
        
        if (i == 0) {
            layer_output = (tensor_t*)malloc(embed_size * sizeof(tensor_t));
        }
        
        swarmformer_layer_forward(model->layers[i], layer_input, layer_output,
                                batch_size, model->seq_len, model->d_model);
        
        if (verbose) {
            print_tensor_stats("Layer output", layer_output, embed_size);
        }
        
        tensor_t* temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;
    }
    
    tensor_t* pooled = (tensor_t*)malloc(batch_size * model->d_model * sizeof(tensor_t));
    for (int b = 0; b < batch_size; b++) {
        for (int d = 0; d < model->d_model; d++) {
            float sum = 0.0f;
            for (int s = 0; s < model->seq_len; s++) {
                sum += layer_input[(b * model->seq_len + s) * model->d_model + d].f32;
            }
            pooled[b * model->d_model + d].f32 = sum / model->seq_len;
        }
    }
    
    if (verbose) {
        print_tensor_stats("Pooled output", pooled, batch_size * model->d_model);
    }
    
    linear_forward(model->classifier, pooled, output, batch_size, model->d_model, 2);
    
    for (int b = 0; b < batch_size; b++) {
        float max_val = output[b * 2].f32;
        if (output[b * 2 + 1].f32 > max_val) max_val = output[b * 2 + 1].f32;
        
        float sum = 0.0f;
        for (int i = 0; i < 2; i++) {
            output[b * 2 + i].f32 = expf(output[b * 2 + i].f32 - max_val);
            sum += output[b * 2 + i].f32;
        }
        
        for (int i = 0; i < 2; i++) {
            output[b * 2 + i].f32 /= sum;
        }
    }
    
    if (verbose) {
        print_tensor_stats("Classifier output", output, batch_size * 2);
    }
    
    free(embedded);
    free(layer_output);
    free(pooled);
} 