from transformers import AutoTokenizer
import struct
import argparse
from pathlib import Path
import json

def convert_tokenizer_to_binary(output_path: str):
    print("Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    print(f"\nVocabulary size: {vocab_size}")
    print("First 10 tokens:")
    for token, idx in sorted_vocab[:10]:
        print(f"  {idx}: {repr(token)}")
    
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'unk_token': tokenizer.unk_token,
        'cls_token': tokenizer.cls_token,
        'sep_token': tokenizer.sep_token,
        'mask_token': tokenizer.mask_token
    }
    special_token_ids = {k: vocab[v] for k, v in special_tokens.items()}
    
    metadata = {
        'vocab_size': vocab_size,
        'special_tokens': special_token_ids,
        'max_length': tokenizer.model_max_length,
    }
    
    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")
    
    # binary format:
    # - uint32: vocab_size
    # - uint32: max_token_length
    # - For each token:
    #   - uint32: token_length
    #   - bytes: token_bytes (UTF-8 encoded)
    print(f"\nSaving binary format to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', vocab_size))

        max_token_len = max(len(token.encode('utf-8')) for token, _ in sorted_vocab)
        f.write(struct.pack('I', max_token_len))

        for token, idx in sorted_vocab:
            token_bytes = token.encode('utf-8')
            f.write(struct.pack('I', len(token_bytes)))
            f.write(token_bytes)
    
    print("\nConversion completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Convert BERT tokenizer to binary format')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save the binary vocabulary file')
    args = parser.parse_args()
    
    convert_tokenizer_to_binary(args.output)

if __name__ == '__main__':
    main() 