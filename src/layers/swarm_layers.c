#include "swarm_layers.h"
#include "../core/activation.h"
#include "../utils/profiler.h"

LocalSwarmAggregator* create_local_swarm_aggregator(int d_model) {
    LocalSwarmAggregator* lsa = (LocalSwarmAggregator*)malloc(sizeof(LocalSwarmAggregator));
    lsa->mlp_fc1 = create_linear(d_model, d_model);
    lsa->mlp_fc2 = create_linear(d_model, d_model);
    lsa->gate_fc1 = create_linear(2 * d_model, d_model);
    lsa->gate_fc2 = create_linear(d_model, d_model);
    lsa->dropout_rate = 0.3f;
    return lsa;
}

void free_local_swarm_aggregator(LocalSwarmAggregator* lsa) {
    free_linear(lsa->mlp_fc1);
    free_linear(lsa->mlp_fc2);
    free_linear(lsa->gate_fc1);
    free_linear(lsa->gate_fc2);
    free(lsa);
}

GlobalClusterAttention* create_global_cluster_attention(int d_model) {
    GlobalClusterAttention* gca = (GlobalClusterAttention*)malloc(sizeof(GlobalClusterAttention));
    gca->query = create_linear(d_model, d_model);
    gca->key = create_linear(d_model, d_model);
    gca->value = create_linear(d_model, d_model);
    gca->scale = 1.0f / sqrtf((float)d_model);
    gca->dropout_rate = 0.3f;
    return gca;
}

void free_global_cluster_attention(GlobalClusterAttention* gca) {
    free_linear(gca->query);
    free_linear(gca->key);
    free_linear(gca->value);
    free(gca);
}

BroadcastUpdater* create_broadcast_updater(int d_model) {
    BroadcastUpdater* bu = (BroadcastUpdater*)malloc(sizeof(BroadcastUpdater));
    bu->linear = create_linear(d_model, d_model);
    bu->gate_fc1 = create_linear(2 * d_model, d_model);
    bu->gate_fc2 = create_linear(d_model, d_model);
    bu->dropout_rate = 0.1f;
    return bu;
}

void free_broadcast_updater(BroadcastUpdater* bu) {
    free_linear(bu->linear);
    free_linear(bu->gate_fc1);
    free_linear(bu->gate_fc2);
    free(bu);
}

SwarmFormerLayer* create_swarmformer_layer(int d_model, int cluster_size, int T_local) {
    SwarmFormerLayer* sfl = (SwarmFormerLayer*)malloc(sizeof(SwarmFormerLayer));
    sfl->local_agg = create_local_swarm_aggregator(d_model);
    sfl->global_attn = create_global_cluster_attention(d_model);
    sfl->broadcast = create_broadcast_updater(d_model);
    sfl->cluster_size = cluster_size;
    sfl->T_local = T_local;
    return sfl;
}

void free_swarmformer_layer(SwarmFormerLayer* sfl) {
    free_local_swarm_aggregator(sfl->local_agg);
    free_global_cluster_attention(sfl->global_attn);
    free_broadcast_updater(sfl->broadcast);
    free(sfl);
}

double calculate_local_swarm_flops(int batch_size, int seq_len, int d_model) {
    double flops = 0;
    flops += 2.0 * batch_size * seq_len * d_model * d_model;
    flops += batch_size * seq_len * d_model;
    flops += 2.0 * batch_size * seq_len * d_model * d_model;
    
    flops += 2.0 * batch_size * seq_len * 2 * d_model * d_model;
    flops += batch_size * seq_len * d_model;
    flops += 2.0 * batch_size * seq_len * d_model * d_model;
    flops += batch_size * seq_len * d_model;
    
    return flops;
}

double calculate_global_attention_flops(int batch_size, int num_clusters, int d_model) {
    double flops = 0;
    flops += 3.0 * 2.0 * batch_size * num_clusters * d_model * d_model;

    flops += 2.0 * batch_size * num_clusters * num_clusters * d_model;
    flops += batch_size * num_clusters * num_clusters;
    flops += batch_size * num_clusters * num_clusters;

    flops += 2.0 * batch_size * num_clusters * num_clusters * d_model;
    
    return flops;
}

double calculate_broadcast_flops(int batch_size, int seq_len, int d_model) {
    double flops = 0;
    flops += 2.0 * batch_size * seq_len * d_model * d_model;
    
    flops += 2.0 * batch_size * seq_len * 2 * d_model * d_model;
    flops += batch_size * seq_len * d_model;
    flops += 2.0 * batch_size * seq_len * d_model * d_model;
    flops += batch_size * seq_len * d_model;
    
    return flops;
}

void local_swarm_aggregator_forward(LocalSwarmAggregator* lsa, tensor_t* x, tensor_t* output,
                                  int batch_size, int seq_len, int d_model) {
    start_profile("Local Swarm Aggregator");
    
    const int total_size = batch_size * seq_len * d_model;
    const int SIMD_WIDTH = 8;
    
    tensor_t* update = (tensor_t*)_mm_malloc(total_size * sizeof(tensor_t), ALIGN_SIZE);
    if (!update) {
        printf("Error: Failed to allocate memory for update\n");
        return;
    }
    
    tensor_t* gates = (tensor_t*)_mm_malloc(total_size * sizeof(tensor_t), ALIGN_SIZE);
    if (!gates) {
        printf("Error: Failed to allocate memory for gates\n");
        _mm_free(update);
        return;
    }
    
    tensor_t* mlp_hidden = (tensor_t*)_mm_malloc(total_size * sizeof(tensor_t), ALIGN_SIZE);
    if (!mlp_hidden) {
        printf("Error: Failed to allocate memory for mlp_hidden\n");
        _mm_free(update);
        _mm_free(gates);
        return;
    }

    int total_seqs = batch_size * seq_len;
    int idx;
    #pragma omp parallel for private(idx)
    for (idx = 0; idx < total_seqs; idx++) {
        int b = idx / seq_len;
        int i = idx % seq_len;
        const int base_idx = (b * seq_len + i) * d_model;
        
        if (i == 0) {
            for (int d = 0; d < d_model - (d_model % SIMD_WIDTH); d += SIMD_WIDTH) {
                __m256 x_curr = _mm256_loadu_ps(&x[base_idx + d].f32);
                __m256 x_next = _mm256_loadu_ps(&x[base_idx + d_model + d].f32);
                __m256 result = _mm256_mul_ps(_mm256_add_ps(x_curr, x_next), _mm256_set1_ps(0.5f));
                _mm256_storeu_ps(&update[base_idx + d].f32, result);
            }
            for (int d = d_model - (d_model % SIMD_WIDTH); d < d_model; d++) {
                update[base_idx + d].f32 = (x[base_idx + d].f32 + x[base_idx + d_model + d].f32) * 0.5f;
            }
        } else if (i == seq_len - 1) {
            for (int d = 0; d < d_model - (d_model % SIMD_WIDTH); d += SIMD_WIDTH) {
                __m256 x_prev = _mm256_loadu_ps(&x[base_idx - d_model + d].f32);
                __m256 x_curr = _mm256_loadu_ps(&x[base_idx + d].f32);
                __m256 result = _mm256_mul_ps(_mm256_add_ps(x_prev, x_curr), _mm256_set1_ps(0.5f));
                _mm256_storeu_ps(&update[base_idx + d].f32, result);
            }
            for (int d = d_model - (d_model % SIMD_WIDTH); d < d_model; d++) {
                update[base_idx + d].f32 = (x[base_idx - d_model + d].f32 + x[base_idx + d].f32) * 0.5f;
            }
        } else {
            for (int d = 0; d < d_model - (d_model % SIMD_WIDTH); d += SIMD_WIDTH) {
                __m256 x_prev = _mm256_loadu_ps(&x[base_idx - d_model + d].f32);
                __m256 x_curr = _mm256_loadu_ps(&x[base_idx + d].f32);
                __m256 x_next = _mm256_loadu_ps(&x[base_idx + d_model + d].f32);
                __m256 sum = _mm256_add_ps(_mm256_add_ps(x_prev, x_curr), x_next);
                __m256 result = _mm256_mul_ps(sum, _mm256_set1_ps(0.333333f));
                _mm256_storeu_ps(&update[base_idx + d].f32, result);
            }
            for (int d = d_model - (d_model % SIMD_WIDTH); d < d_model; d++) {
                update[base_idx + d].f32 = (x[base_idx - d_model + d].f32 + x[base_idx + d].f32 + 
                                          x[base_idx + d_model + d].f32) * 0.333333f;
            }
        }
        
        for (int j = 0; j < d_model; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int k = 0; k < d_model - (d_model % SIMD_WIDTH); k += SIMD_WIDTH) {
                __m256 update_vec = _mm256_loadu_ps(&update[base_idx + k].f32);
                __m256 weight_vec = _mm256_loadu_ps(&lsa->mlp_fc1->weight->data[j * d_model + k].f32);
                sum_vec = _mm256_fmadd_ps(update_vec, weight_vec, sum_vec);
            }
            
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += sum_array[k];
            }
            
            for (int k = d_model - (d_model % SIMD_WIDTH); k < d_model; k++) {
                sum += update[base_idx + k].f32 * lsa->mlp_fc1->weight->data[j * d_model + k].f32;
            }
            
            mlp_hidden[base_idx + j].f32 = sum + lsa->mlp_fc1->bias->data[j].f32;
        }
    }
    
    gelu_forward(mlp_hidden, total_size, lsa->mlp_fc1->weight->quant_mode);
    
    #pragma omp parallel for private(idx)
    for (idx = 0; idx < total_seqs; idx++) {
        int b = idx / seq_len;
        int i = idx % seq_len;
        const int base_idx = (b * seq_len + i) * d_model;
        
        for (int j = 0; j < d_model; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int k = 0; k < d_model - (d_model % SIMD_WIDTH); k += SIMD_WIDTH) {
                __m256 hidden_vec = _mm256_loadu_ps(&mlp_hidden[base_idx + k].f32);
                __m256 weight_vec = _mm256_loadu_ps(&lsa->mlp_fc2->weight->data[j * d_model + k].f32);
                sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
            }
            
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += sum_array[k];
            }
            
            for (int k = d_model - (d_model % SIMD_WIDTH); k < d_model; k++) {
                sum += mlp_hidden[base_idx + k].f32 * lsa->mlp_fc2->weight->data[j * d_model + k].f32;
            }
            
            update[base_idx + j].f32 = sum + lsa->mlp_fc2->bias->data[j].f32;
        }
    }
    
    #pragma omp parallel for private(idx)
    for (idx = 0; idx < total_seqs; idx++) {
        int b = idx / seq_len;
        int i = idx % seq_len;
        const int base_idx = (b * seq_len + i) * d_model;
        
        for (int j = 0; j < d_model; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int k = 0; k < d_model - (d_model % SIMD_WIDTH); k += SIMD_WIDTH) {
                __m256 x_vec = _mm256_loadu_ps(&x[base_idx + k].f32);
                __m256 update_vec = _mm256_loadu_ps(&update[base_idx + k].f32);
                __m256 weight_x = _mm256_loadu_ps(&lsa->gate_fc1->weight->data[j * d_model + k].f32);
                __m256 weight_u = _mm256_loadu_ps(&lsa->gate_fc1->weight->data[j * d_model + d_model + k].f32);
                sum_vec = _mm256_fmadd_ps(x_vec, weight_x, sum_vec);
                sum_vec = _mm256_fmadd_ps(update_vec, weight_u, sum_vec);
            }
            
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += sum_array[k];
            }
            
            for (int k = d_model - (d_model % SIMD_WIDTH); k < d_model; k++) {
                sum += x[base_idx + k].f32 * lsa->gate_fc1->weight->data[j * d_model + k].f32 +
                       update[base_idx + k].f32 * lsa->gate_fc1->weight->data[j * d_model + d_model + k].f32;
            }
            
            mlp_hidden[base_idx + j].f32 = sum + lsa->gate_fc1->bias->data[j].f32;
        }
    }
    
    gelu_forward(mlp_hidden, total_size, lsa->gate_fc1->weight->quant_mode);
    
    #pragma omp parallel for private(idx)
    for (idx = 0; idx < total_seqs; idx++) {
        int b = idx / seq_len;
        int i = idx % seq_len;
        const int base_idx = (b * seq_len + i) * d_model;
        
        for (int j = 0; j < d_model; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int k = 0; k < d_model - (d_model % SIMD_WIDTH); k += SIMD_WIDTH) {
                __m256 hidden_vec = _mm256_loadu_ps(&mlp_hidden[base_idx + k].f32);
                __m256 weight_vec = _mm256_loadu_ps(&lsa->gate_fc2->weight->data[j * d_model + k].f32);
                sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
            }
            
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += sum_array[k];
            }
            
            for (int k = d_model - (d_model % SIMD_WIDTH); k < d_model; k++) {
                sum += mlp_hidden[base_idx + k].f32 * lsa->gate_fc2->weight->data[j * d_model + k].f32;
            }
            
            gates[base_idx + j].f32 = sum + lsa->gate_fc2->bias->data[j].f32;
        }
    }
    
    sigmoid_forward(gates, total_size, lsa->gate_fc2->weight->quant_mode);
    
    #pragma omp parallel for private(idx)
    for (idx = 0; idx < total_seqs; idx++) {
        int b = idx / seq_len;
        int i = idx % seq_len;
        const int base_idx = (b * seq_len + i) * d_model;
        
        for (int d = 0; d < d_model - (d_model % SIMD_WIDTH); d += SIMD_WIDTH) {
            __m256 x_vec = _mm256_loadu_ps(&x[base_idx + d].f32);
            __m256 u_vec = _mm256_loadu_ps(&update[base_idx + d].f32);
            __m256 g_vec = _mm256_loadu_ps(&gates[base_idx + d].f32);
            
            __m256 diff = _mm256_sub_ps(u_vec, x_vec);
            __m256 weighted = _mm256_mul_ps(g_vec, diff);
            __m256 result = _mm256_add_ps(x_vec, weighted);
            
            _mm256_storeu_ps(&output[base_idx + d].f32, result);
        }
        
        for (int d = d_model - (d_model % SIMD_WIDTH); d < d_model; d++) {
            float x_val = x[base_idx + d].f32;
            float u_val = update[base_idx + d].f32;
            float g_val = gates[base_idx + d].f32;
            output[base_idx + d].f32 = x_val + g_val * (u_val - x_val);
        }
    }
    
    _mm_free(update);
    _mm_free(gates);
    _mm_free(mlp_hidden);
    
    end_profile("Local Swarm Aggregator", calculate_local_swarm_flops(batch_size, seq_len, d_model));
}

void global_cluster_attention_forward(GlobalClusterAttention* gca, tensor_t* x, tensor_t* output,
                                   int batch_size, int num_clusters, int d_model) {
    start_profile("Global Cluster Attention");
    
    const int qkv_size = batch_size * num_clusters * d_model;
    tensor_t* q = (tensor_t*)malloc(qkv_size * sizeof(tensor_t));
    tensor_t* k = (tensor_t*)malloc(qkv_size * sizeof(tensor_t));
    tensor_t* v = (tensor_t*)malloc(qkv_size * sizeof(tensor_t));
    tensor_t* attn_scores = (tensor_t*)malloc(batch_size * num_clusters * num_clusters * sizeof(tensor_t));
    
    if (!q || !k || !v || !attn_scores) {
        printf("Error: Failed to allocate memory\n");
        if (q) free(q);
        if (k) free(k);
        if (v) free(v);
        if (attn_scores) free(attn_scores);
        return;
    }
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int b = 0; b < batch_size; b++) {
                for (int c = 0; c < num_clusters; c++) {
                    const int base_idx = (b * num_clusters + c) * d_model;
                    
                    for (int j = 0; j < d_model; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < d_model; k++) {
                            sum += x[base_idx + k].f32 * gca->query->weight->data[j * d_model + k].f32;
                        }
                        q[base_idx + j].f32 = sum + gca->query->bias->data[j].f32;
                    }
                }
            }
        }
        
        #pragma omp section
        {
            for (int b = 0; b < batch_size; b++) {
                for (int c = 0; c < num_clusters; c++) {
                    const int base_idx = (b * num_clusters + c) * d_model;
                    
                    for (int j = 0; j < d_model; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < d_model; k++) {
                            sum += x[base_idx + k].f32 * gca->key->weight->data[j * d_model + k].f32;
                        }
                        k[base_idx + j].f32 = sum + gca->key->bias->data[j].f32;
                    }
                }
            }
        }
        
        #pragma omp section
        {
            for (int b = 0; b < batch_size; b++) {
                for (int c = 0; c < num_clusters; c++) {
                    const int base_idx = (b * num_clusters + c) * d_model;
                    
                    for (int j = 0; j < d_model; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < d_model; k++) {
                            sum += x[base_idx + k].f32 * gca->value->weight->data[j * d_model + k].f32;
                        }
                        v[base_idx + j].f32 = sum + gca->value->bias->data[j].f32;
                    }
                }
            }
        }
    }
    
    const float scale = gca->scale;
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_clusters; c++) {
            const int base_idx = (b * num_clusters + c) * d_model;
            for (int j = 0; j < d_model; j++) {
                q[base_idx + j].f32 *= scale;
            }
        }
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_clusters; i++) {
            const int score_offset = b * num_clusters * num_clusters + i * num_clusters;
            const int q_offset = (b * num_clusters + i) * d_model;
            
            for (int j = 0; j < num_clusters; j++) {
                const int k_offset = (b * num_clusters + j) * d_model;
                float dot_product = 0.0f;
                
                for (int d = 0; d < d_model; d++) {
                    dot_product += q[q_offset + d].f32 * k[k_offset + d].f32;
                }
                
                attn_scores[score_offset + j].f32 = dot_product;
            }
            
            float max_val = attn_scores[score_offset].f32;
            for (int j = 1; j < num_clusters; j++) {
                if (attn_scores[score_offset + j].f32 > max_val) {
                    max_val = attn_scores[score_offset + j].f32;
                }
            }
            
            float sum = 0.0f;
            for (int j = 0; j < num_clusters; j++) {
                float score = attn_scores[score_offset + j].f32;
                score = expf(score - max_val);
                attn_scores[score_offset + j].f32 = score;
                sum += score;
            }
            
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < num_clusters; j++) {
                attn_scores[score_offset + j].f32 *= inv_sum;
            }
        }
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_clusters; i++) {
            const int out_offset = (b * num_clusters + i) * d_model;
            const int score_offset = b * num_clusters * num_clusters + i * num_clusters;
            
            for (int d = 0; d < d_model; d++) {
                float sum = 0.0f;
                for (int j = 0; j < num_clusters; j++) {
                    float score = attn_scores[score_offset + j].f32;
                    sum += score * v[(b * num_clusters + j) * d_model + d].f32;
                }
                output[out_offset + d].f32 = sum;
            }
        }
    }
    
    free(q);
    free(k);
    free(v);
    free(attn_scores);
    
    end_profile("Global Cluster Attention", calculate_global_attention_flops(batch_size, num_clusters, d_model));
}

void broadcast_updater_forward(BroadcastUpdater* bu, tensor_t* x, tensor_t* output,
                             int batch_size, int seq_len, int d_model) {
    if (!bu || !x || !output) {
        printf("Error: NULL pointer passed to broadcast_updater_forward\n");
        return;
    }
    
    start_profile("Broadcast Updater");
    
    tensor_t* linear_out = (tensor_t*)malloc(batch_size * seq_len * d_model * sizeof(tensor_t));
    if (!linear_out) {
        printf("Error: Failed to allocate memory for linear_out\n");
        return;
    }
    
    tensor_t* concat = (tensor_t*)malloc(batch_size * seq_len * (2 * d_model) * sizeof(tensor_t));
    if (!concat) {
        printf("Error: Failed to allocate memory for concat\n");
        free(linear_out);
        return;
    }
    
    tensor_t* gate_hidden = (tensor_t*)malloc(batch_size * seq_len * d_model * sizeof(tensor_t));
    if (!gate_hidden) {
        printf("Error: Failed to allocate memory for gate_hidden\n");
        free(linear_out);
        free(concat);
        return;
    }
    
    tensor_t* gate = (tensor_t*)malloc(batch_size * seq_len * d_model * sizeof(tensor_t));
    if (!gate) {
        printf("Error: Failed to allocate memory for gate\n");
        free(linear_out);
        free(concat);
        free(gate_hidden);
        return;
    }
    
    linear_forward(bu->linear, x, linear_out, batch_size * seq_len, d_model, d_model);
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < d_model; d++) {
                int src_idx = (b * seq_len + i) * d_model + d;
                int dst_idx = (b * seq_len + i) * (2 * d_model) + d;
                concat[dst_idx].f32 = x[src_idx].f32;
                concat[dst_idx + d_model].f32 = linear_out[src_idx].f32;
            }
        }
    }
    
    linear_forward(bu->gate_fc1, concat, gate_hidden, batch_size * seq_len, 2 * d_model, d_model);
    sigmoid_forward(gate_hidden, batch_size * seq_len * d_model, bu->gate_fc1->weight->quant_mode);
    linear_forward(bu->gate_fc2, gate_hidden, gate, batch_size * seq_len, d_model, d_model);
    sigmoid_forward(gate, batch_size * seq_len * d_model, bu->gate_fc2->weight->quant_mode);
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < d_model; d++) {
                int idx = (b * seq_len + i) * d_model + d;
                output[idx].f32 = gate[idx].f32 * linear_out[idx].f32 + 
                                 (1.0f - gate[idx].f32) * x[idx].f32;
            }
        }
    }
    
    free(linear_out);
    free(concat);
    free(gate_hidden);
    free(gate);
    
    end_profile("Broadcast Updater", calculate_broadcast_flops(batch_size, seq_len, d_model));
}

void swarmformer_layer_forward(SwarmFormerLayer* sfl, tensor_t* x, tensor_t* output,
                             int batch_size, int seq_len, int d_model) {
    start_profile("SwarmFormer Layer");
    
    const int seq_size = batch_size * seq_len * d_model;
    const int num_clusters = seq_len / sfl->cluster_size;
    const int cluster_size = batch_size * num_clusters * d_model;
    
    tensor_t* local_out = (tensor_t*)malloc(seq_size * sizeof(tensor_t));
    tensor_t* cluster_reps = (tensor_t*)malloc(cluster_size * sizeof(tensor_t));
    tensor_t* global_out = (tensor_t*)malloc(cluster_size * sizeof(tensor_t));
    
    if (!local_out || !cluster_reps || !global_out) {
        if (local_out) free(local_out);
        if (cluster_reps) free(cluster_reps);
        if (global_out) free(global_out);
        return;
    }
    
    memcpy(local_out, x, seq_size * sizeof(tensor_t));
    for (int t = 0; t < sfl->T_local; t++) {
        local_swarm_aggregator_forward(sfl->local_agg, local_out, local_out, batch_size, seq_len, d_model);
    }
    
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_clusters; c++) {
            const int cluster_offset = (b * num_clusters + c) * d_model;
            const int seq_offset = (b * seq_len + c * sfl->cluster_size) * d_model;
            
            for (int d = 0; d < d_model; d++) {
                float sum = 0.0f;
                for (int i = 0; i < sfl->cluster_size; i++) {
                    sum += local_out[seq_offset + i * d_model + d].f32;
                }
                cluster_reps[cluster_offset + d].f32 = sum / (float)sfl->cluster_size;
            }
        }
    }
    
    global_cluster_attention_forward(sfl->global_attn, cluster_reps, global_out, batch_size, num_clusters, d_model);
    
    broadcast_updater_forward(sfl->broadcast, local_out, output, batch_size, seq_len, d_model);
    
    free(local_out);
    free(cluster_reps);
    free(global_out);
    
    end_profile("SwarmFormer Layer", 
                calculate_local_swarm_flops(batch_size, seq_len, d_model) * sfl->T_local +
                calculate_global_attention_flops(batch_size, num_clusters, d_model) +
                calculate_broadcast_flops(batch_size, seq_len, d_model));
} 