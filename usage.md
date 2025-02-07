# SwarmFormer.c Usage

### Basic Usage
```bash
./swarmformer "model_weights.bin" "The movie was terrible."
```

### Verbose Mode
Verbose mode will print the output of each layer, and a breakdown of the time taken by each layer.
```bash
./swarmformer -v "model_weights.bin" "The movie was terrible."
```

### Benchmark Mode

```bash
./swarmformer  --benchmark "model_weights.bin"
```