# SwarmFormer.c Setup Guide

Similar to llama.cpp, this features a C interface for the SwarmFormer architecture.

## Prerequisites

- Python 3.9+
- 2-3 GB of free disk space (for converting the weights, PyTorch is large)
- At least 50MB of free RAM (for running the model)
- AVX2 support

## Setup

1. Clone the repository
2. Install the dependencies
3. Convert the weights and the tokenizer
4. Build the C library
5. Run the model

### 1. Clone the repository

```bash
git clone https://github.com/takara-ai/SwarmFormer-C.git
cd SwarmFormer-C
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3.1. Convert the weights
Both the model and the tokenizer has to be named the same.
```bash
python convert_weights.py --pytorch_model "hf://takara-ai/SwarmFormer-Sentiment-Base" --output "model_weights.bin"
```

### 3.2. Convert the tokenizer

```bash
python convert_tokenizer.py --output "model_weights.vocab"
```

### 4. Build the C library

```bash
mkdir build
cd build
cmake ..
cmake --build . 
```

### 5. Run the model

```bash
./SwarmFormer.c "model_weights.bin" "The movie was terrible."
# "the movie was terrible" -> Negative (61.0%)
```

More examples can be found in the [usage.md](usage.md) file.

## Notes
1. Your model weights and vocab name need to be the same, for example: `model.vocab` and `model.bin`. It will try find the tokenizer automatically based off the name.
2. This may not compile on Apple Silicon, requires more testing.
