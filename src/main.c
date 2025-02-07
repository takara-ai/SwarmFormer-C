#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include "model/swarmformer.h"
#include "utils/loader.h"
#include "utils/tokenizer.h"
#include "utils/profiler.h"
#include "core/quantization.h"

bool verbose = false;
bool benchmark = false;
QuantMode quant_mode = QUANT_FLOAT32;

// test sentences for benchmarking
const char* benchmark_sentences[] = {
    "this movie was absolutely fantastic and I loved every minute of it",
    "the acting was terrible and the plot made no sense at all",
    "while it had some good moments, overall it was just mediocre",
    "I can't believe how amazing this film was, definitely worth watching",
    "the special effects were great but the story was lacking",
    "this might be one of the worst movies I've ever seen",
    "the performances were outstanding and the direction was superb",
    "it wasn't terrible but I wouldn't watch it again",
    "a masterpiece that will be remembered for generations",
    "the pacing was off and I found myself getting bored",
    "brilliant performances all around, especially from the lead actor",
    "the dialogue felt forced and unnatural throughout",
    "visually stunning but emotionally empty",
    "this exceeded all my expectations, truly remarkable",
    "a complete waste of time and money",
    "the cinematography was breathtaking in every scene",
    "poorly written and even more poorly executed",
    "an instant classic that deserves all the praise",
    "mediocre at best, forgettable at worst",
    "the director's vision really shines through in this one"
};
#define NUM_BENCHMARK_SENTENCES (sizeof(benchmark_sentences) / sizeof(char*))

typedef struct {
    double total_tokens;
    double total_time;
    double tokenization_time;
    double inference_time;
    double flops;
} BenchmarkStats;

double calculate_flops(SwarmFormerModel* model, int seq_len) {
    double flops = 0;
    int d_model = model->d_model;
    int cluster_size = model->layers[0]->cluster_size;
    int num_clusters = seq_len / cluster_size;
        
    for (int l = 0; l < model->num_layers; l++) {
        flops += 4 * d_model * d_model * seq_len;
        flops += 4 * d_model * d_model * seq_len;

        flops += 6 * d_model * d_model * num_clusters;
        flops += 2 * num_clusters * num_clusters * d_model;
        flops += 2 * num_clusters * num_clusters * d_model;

        flops += 2 * d_model * d_model * seq_len;
        flops += 4 * d_model * d_model * seq_len;
    }

    flops += seq_len * d_model;
    
    // 2 * d_model * 2
    flops += 4 * d_model;
    
    return flops;
}

void print_usage(const char* program_name) {
    printf("Usage:\n");
    printf("  %s [-v] [--quant MODE] <weights_file>                    # Run with example input\n", program_name);
    printf("  %s [-v] [--quant MODE] <weights_file> \"<input_text>\"     # Run inference on input text\n", program_name);
    printf("  %s --benchmark [--quant MODE] <weights_file>             # Run benchmark\n", program_name);
    printf("\nOptions:\n");
    printf("  -v           Enable verbose output with detailed tensor statistics\n");
    printf("  --benchmark  Run benchmark suite\n");
    printf("  --quant MODE Quantization mode (float32, int8, int4)\n");
}

QuantMode parse_quant_mode(const char* mode_str) {
    if (strcmp(mode_str, "float32") == 0) return QUANT_FLOAT32;
    if (strcmp(mode_str, "int8") == 0) return QUANT_INT8;
    if (strcmp(mode_str, "int4") == 0) return QUANT_INT4;
    printf("Warning: Unknown quantization mode '%s', using float32\n", mode_str);
    return QUANT_FLOAT32;
}

const char* quant_mode_str(QuantMode mode) {
    switch (mode) {
        case QUANT_FLOAT32: return "float32";
        case QUANT_INT8: return "int8";
        case QUANT_INT4: return "int4";
        default: return "unknown";
    }
}

void compute_probabilities(tensor_t* logits, tensor_t* probs, int size) {
    float max_val = logits[0].f32;
    for (int i = 1; i < size; i++) {
        if (logits[i].f32 > max_val) max_val = logits[i].f32;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i].f32 = expf(logits[i].f32 - max_val);
        sum += probs[i].f32;
    }
    
    for (int i = 0; i < size; i++) {
        probs[i].f32 /= sum;
    }
}

void run_benchmark(const char* weights_path) {
    printf("Loading model for benchmark...\n");
    
    SwarmFormerModel* model = load_swarmformer_model(weights_path);
    if (!model) {
        printf("Failed to load model\n");
        return;
    }
    
    if (quant_mode != QUANT_FLOAT32) {
        printf("Quantizing model to %s...\n", quant_mode_str(quant_mode));
        quantize_tensor(model->embedding->embed, quant_mode);
        for (int i = 0; i < model->num_layers; i++) {
            SwarmFormerLayer* layer = model->layers[i];
            quantize_tensor(layer->local_agg->mlp_fc1->weight, quant_mode);
            quantize_tensor(layer->local_agg->mlp_fc1->bias, quant_mode);
            quantize_tensor(layer->local_agg->mlp_fc2->weight, quant_mode);
            quantize_tensor(layer->local_agg->mlp_fc2->bias, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc1->weight, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc1->bias, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc2->weight, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc2->bias, quant_mode);
            
            quantize_tensor(layer->global_attn->query->weight, quant_mode);
            quantize_tensor(layer->global_attn->query->bias, quant_mode);
            quantize_tensor(layer->global_attn->key->weight, quant_mode);
            quantize_tensor(layer->global_attn->key->bias, quant_mode);
            quantize_tensor(layer->global_attn->value->weight, quant_mode);
            quantize_tensor(layer->global_attn->value->bias, quant_mode);
            
            quantize_tensor(layer->broadcast->linear->weight, quant_mode);
            quantize_tensor(layer->broadcast->linear->bias, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc1->weight, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc1->bias, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc2->weight, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc2->bias, quant_mode);
        }
        quantize_tensor(model->classifier->weight, quant_mode);
        quantize_tensor(model->classifier->bias, quant_mode);
    }
    
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%.*s.vocab", 
             (int)(strrchr(weights_path, '.') - weights_path), weights_path);
    
    Tokenizer* tokenizer = load_tokenizer(vocab_path);
    if (!tokenizer) {
        printf("Failed to load tokenizer\n");
        free_swarmformer_model(model);
        return;
    }
    
    BenchmarkStats stats = {0};
    tensor_t* output = (tensor_t*)malloc(2 * sizeof(tensor_t));
    
    printf("\nRunning benchmark with %zu sentences...\n", NUM_BENCHMARK_SENTENCES);
    
    init_profiler();
    enable_profiler(true);
    
    clock_t total_start = clock();
    
    for (size_t i = 0; i < NUM_BENCHMARK_SENTENCES; i++) {
        clock_t tok_start = clock();
        int num_tokens;
        int* tokens = tokenize_text(tokenizer, benchmark_sentences[i], model->seq_len, &num_tokens);
        clock_t tok_end = clock();
        
        stats.tokenization_time += ((double)(tok_end - tok_start)) / CLOCKS_PER_SEC;
        stats.total_tokens += num_tokens;
        
        clock_t inf_start = clock();
        swarmformer_forward(model, tokens, output, 1, false);
        clock_t inf_end = clock();
        
        stats.inference_time += ((double)(inf_end - inf_start)) / CLOCKS_PER_SEC;
        stats.flops += calculate_flops(model, num_tokens);
        
        free(tokens);
        
        printf("\rProgress: %zu/%zu", i + 1, NUM_BENCHMARK_SENTENCES);
        fflush(stdout);
    }
    
    clock_t total_end = clock();
    stats.total_time = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;
    
    printf("\n\nBenchmark Results:\n");
    printf("Total time: %.2f seconds\n", stats.total_time);
    printf("Average tokenization time: %.4f seconds\n", stats.tokenization_time / NUM_BENCHMARK_SENTENCES);
    printf("Average inference time: %.4f seconds\n", stats.inference_time / NUM_BENCHMARK_SENTENCES);
    printf("Throughput: %.2f tokens/second\n", stats.total_tokens / stats.total_time);
    printf("Tokenization speed: %.2f tokens/second\n", stats.total_tokens / stats.tokenization_time);
    printf("Average FLOPS: %.2f GFLOPS\n", stats.flops / stats.inference_time / 1e9);
    
    print_profile_stats();
    
    free(output);
    free_tokenizer(tokenizer);
    free_swarmformer_model(model);
}

int main(int argc, char** argv) {
    int arg_idx = 1;
    
    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "-v") == 0) {
            verbose = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--benchmark") == 0) {
            benchmark = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--quant") == 0) {
            if (arg_idx + 1 >= argc) {
                printf("Error: --quant requires a mode argument\n");
                print_usage(argv[0]);
                return 1;
            }
            quant_mode = parse_quant_mode(argv[arg_idx + 1]);
            arg_idx += 2;
        } else {
            printf("Unknown option: %s\n", argv[arg_idx]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (arg_idx >= argc || (!benchmark && arg_idx + 2 < argc)) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* weights_path = argv[arg_idx];
    
    if (verbose) {
        printf("Using quantization mode: %s\n", quant_mode_str(quant_mode));
    }
    
    if (benchmark) {
        run_benchmark(weights_path);
        return 0;
    }
    
    init_profiler();
    enable_profiler(verbose);
    
    const char* input_text = (arg_idx + 1 < argc) ? argv[arg_idx + 1] : "this movie was great!";
    
    if (verbose) {
        printf("Loading model from: %s\n", weights_path);
    }
    
    SwarmFormerModel* model = load_swarmformer_model(weights_path);
    if (!model) {
        printf("Failed to load model\n");
        return 1;
    }
    
    if (quant_mode != QUANT_FLOAT32) {
        if (verbose) {
            printf("Quantizing model to %s...\n", quant_mode_str(quant_mode));
        }
        quantize_tensor(model->embedding->embed, quant_mode);
        for (int i = 0; i < model->num_layers; i++) {
            SwarmFormerLayer* layer = model->layers[i];
            quantize_tensor(layer->local_agg->mlp_fc1->weight, quant_mode);
            quantize_tensor(layer->local_agg->mlp_fc1->bias, quant_mode);
            quantize_tensor(layer->local_agg->mlp_fc2->weight, quant_mode);
            quantize_tensor(layer->local_agg->mlp_fc2->bias, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc1->weight, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc1->bias, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc2->weight, quant_mode);
            quantize_tensor(layer->local_agg->gate_fc2->bias, quant_mode);
            
            quantize_tensor(layer->global_attn->query->weight, quant_mode);
            quantize_tensor(layer->global_attn->query->bias, quant_mode);
            quantize_tensor(layer->global_attn->key->weight, quant_mode);
            quantize_tensor(layer->global_attn->key->bias, quant_mode);
            quantize_tensor(layer->global_attn->value->weight, quant_mode);
            quantize_tensor(layer->global_attn->value->bias, quant_mode);
            
            quantize_tensor(layer->broadcast->linear->weight, quant_mode);
            quantize_tensor(layer->broadcast->linear->bias, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc1->weight, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc1->bias, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc2->weight, quant_mode);
            quantize_tensor(layer->broadcast->gate_fc2->bias, quant_mode);
        }
        quantize_tensor(model->classifier->weight, quant_mode);
        quantize_tensor(model->classifier->bias, quant_mode);
    }
    
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%.*s.vocab", 
             (int)(strrchr(weights_path, '.') - weights_path), weights_path);
    
    if (verbose) {
        printf("Loading tokenizer from: %s\n", vocab_path);
    }
    
    Tokenizer* tokenizer = load_tokenizer(vocab_path);
    if (!tokenizer) {
        printf("Failed to load tokenizer\n");
        free_swarmformer_model(model);
        return 1;
    }
    
    if (verbose) {
        printf("\nModel configuration:\n");
        printf("  vocab_size: %d\n", model->embedding->embed->shape[0]);
        printf("  d_model: %d\n", model->d_model);
        printf("  seq_len: %d\n", model->seq_len);
        printf("  cluster_size: %d\n", model->layers[0]->cluster_size);
        printf("  num_layers: %d\n", model->num_layers);
        printf("  T_local: %d\n", model->layers[0]->T_local);
        
        printf("\nInput text: \"%s\"\n", input_text);
    }
    
    int num_tokens;
    int* tokens = tokenize_text(tokenizer, input_text, model->seq_len, &num_tokens);
    
    if (verbose) {
        printf("\nTokenized (first 10 tokens):\n");
        for (int i = 0; i < (num_tokens < 10 ? num_tokens : 10); i++) {
            printf("  %d: %s\n", tokens[i], 
                   tokens[i] < tokenizer->vocab_size ? tokenizer->tokens[tokens[i]] : "<special>");
        }
        if (num_tokens > 10) printf("  ...\n");
        
        printf("\nRunning forward pass...\n");
    }
    
    tensor_t* output = (tensor_t*)malloc(2 * sizeof(tensor_t));
    
    swarmformer_forward(model, tokens, output, 1, verbose);
    
    printf("\"%s\" -> ", input_text);
    if (output[1].f32 > output[0].f32) {
        printf("Positive (%.1f%%)\n", output[1].f32 * 100.0f);
    } else {
        printf("Negative (%.1f%%)\n", output[0].f32 * 100.0f);
    }
    
    if (verbose) {
        print_profile_stats();
    }
    
    free(tokens);
    free(output);
    free_tokenizer(tokenizer);
    free_swarmformer_model(model);
    
    return 0;
} 