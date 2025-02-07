import torch
import numpy as np
import struct
import os
import json
from typing import Dict, Any
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file

def print_tensor_stats(name: str, tensor: torch.Tensor):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor.float().mean().item():.6f}")
    print(f"  Std: {tensor.float().std().item():.6f}")
    print(f"  Min: {tensor.float().min().item():.6f}")
    print(f"  Max: {tensor.float().max().item():.6f}")

def save_tensor(f, tensor: torch.Tensor):
    print_tensor_stats("Saving tensor", tensor)
    tensor_np = tensor.detach().cpu().float().numpy()
    f.write(struct.pack('i', len(tensor_np.shape)))
    for dim in tensor_np.shape:
        f.write(struct.pack('i', dim))
    f.write(struct.pack(f'{tensor_np.size}f', *tensor_np.flatten()))

def save_linear_layer(f, weight: torch.Tensor, bias: torch.Tensor):
    print("\nSaving linear layer:")
    print_tensor_stats("  Weight", weight)
    print_tensor_stats("  Bias", bias)
    save_tensor(f, weight)
    save_tensor(f, bias)

def save_local_swarm_aggregator(f, lsa_dict: Dict[str, Any]):
    save_linear_layer(f, lsa_dict['mlp.0.weight'], lsa_dict['mlp.0.bias'])
    save_linear_layer(f, lsa_dict['mlp.3.weight'], lsa_dict['mlp.3.bias'])
    save_linear_layer(f, lsa_dict['gate_net.0.weight'], lsa_dict['gate_net.0.bias'])
    save_linear_layer(f, lsa_dict['gate_net.2.weight'], lsa_dict['gate_net.2.bias'])

def save_global_cluster_attention(f, gca_dict: Dict[str, Any]):
    save_linear_layer(f, gca_dict['query.weight'], gca_dict['query.bias'])
    save_linear_layer(f, gca_dict['key.weight'], gca_dict['key.bias'])
    save_linear_layer(f, gca_dict['value.weight'], gca_dict['value.bias'])

def save_broadcast_updater(f, bu_dict: Dict[str, Any]):
    save_linear_layer(f, bu_dict['linear.weight'], bu_dict['linear.bias'])
    save_linear_layer(f, bu_dict['gate_net.0.weight'], bu_dict['gate_net.0.bias'])
    save_linear_layer(f, bu_dict['gate_net.2.weight'], bu_dict['gate_net.2.bias'])

def save_swarmformer_layer(f, layer_dict: Dict[str, Any]):
    local_agg_dict = {k.replace('local_agg.', ''): v for k, v in layer_dict.items() if k.startswith('local_agg.')}
    global_attn_dict = {k.replace('global_attn.', ''): v for k, v in layer_dict.items() if k.startswith('global_attn.')}
    broadcast_dict = {k.replace('broadcast.', ''): v for k, v in layer_dict.items() if k.startswith('broadcast.')}
    
    save_local_swarm_aggregator(f, local_agg_dict)
    save_global_cluster_attention(f, global_attn_dict)
    save_broadcast_updater(f, broadcast_dict)

def extract_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    embedding_weight = state_dict.get('embedding.weight', state_dict.get('embedding.embed.weight'))
    if embedding_weight is None:
        raise ValueError("Could not find embedding weight in state dict")
    
    config = {
        'vocab_size': embedding_weight.shape[0],
        'd_model': embedding_weight.shape[1],
        'num_layers': len(set(int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('layers.'))),
        'seq_len': 256,
        'cluster_size': 8,
        'T_local': 2
    }
    
    print("Extracted configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    return config

def convert_pytorch_to_c(model_path: str, output_path: str):
    print(f"Loading model from: {model_path}")
    
    try:
        if model_path.startswith("hf://"):
            repo_id = model_path[5:]
            print(f"Looking for SafeTensors files in {repo_id}...")
            files = list_repo_files(repo_id)
            safetensor_files = [f for f in files if f.endswith('.safetensors')]
            if not safetensor_files:
                raise ValueError(f"No SafeTensors files found in repository {repo_id}")
            model_file = safetensor_files[0]
            print(f"Found model file: {model_file}")
            safetensors_path = hf_hub_download(repo_id, model_file)
            state_dict = load_file(safetensors_path)
            
            config_file = next((f for f in files if f == 'config.json'), None)
            if config_file:
                print(f"Found config file: {config_file}")
                config_path = hf_hub_download(repo_id, config_file)
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    embedding_weight = state_dict.get('embedding.weight', state_dict.get('embedding.embed.weight'))
                    if embedding_weight is None:
                        raise ValueError("Could not find embedding weight in state dict")
                    config['vocab_size'] = embedding_weight.shape[0]
            else:
                config = extract_model_config(state_dict)
        else:
            if model_path.endswith('.safetensors'):
                state_dict = load_file(model_path)
                config = extract_model_config(state_dict)
            else:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    config = checkpoint.get('params', {}).copy() or extract_model_config(state_dict)
                else:
                    state_dict = checkpoint
                    config = extract_model_config(state_dict)
        
        print("\nPyTorch classifier weights:")
        print_tensor_stats("  Weight", state_dict['classifier.weight'])
        print_tensor_stats("  Bias", state_dict['classifier.bias'])
        
        embedding_weight = state_dict.get('embedding.weight', state_dict.get('embedding.embed.weight'))
        config['vocab_size'] = embedding_weight.shape[0]
        
        output_path = Path(output_path)
        if output_path.parent != Path('.'):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting weights to: {output_path}")
        print("\nModel configuration:")
        for k, v in sorted(config.items()):
            print(f"  {k}: {v}")
        
        print("\nWeight statistics:")
        with open(output_path, 'wb') as f:
            for key in ['vocab_size', 'd_model', 'seq_len', 'cluster_size', 'num_layers', 'T_local']:
                f.write(struct.pack('i', config[key]))
            
            print("\nEmbedding layer:")
            print_tensor_stats("  Weights", embedding_weight)
            save_tensor(f, embedding_weight)
            
            for i in range(config['num_layers']):
                print(f"\nLayer {i}:")
                prefix = f'layers.{i}.'
                layer_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                save_swarmformer_layer(f, layer_dict)
            
            print("\nClassifier layer:")
            save_linear_layer(f, state_dict['classifier.weight'], state_dict['classifier.bias'])
        
        print("\nConversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        print("\nState dict keys found:")
        if 'state_dict' in locals():
            for k in sorted(state_dict.keys()):
                print(f"  {k}")
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_model', type=str, required=True,
                        help='Path to PyTorch model checkpoint or HuggingFace model ID (prefix with "hf://")')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save C model weights')
    args = parser.parse_args()
    convert_pytorch_to_c(args.pytorch_model, args.output) 