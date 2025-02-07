# Benchmarks

>**Note**: This list will be updated as more benchmarks are added, and more processors are tested.

## CPU (SwarmFormer-Sentiment-Base)

### AMD Ryzen 5 3600, Windows 11
- Total time: 2.30 seconds
- Average tokenization time: 0.0005 seconds
- Average inference time: 0.1144 seconds
- Throughput: 2228.02 tokens/second
- Tokenization speed: 465454.55 +- 100000 tokens/second (still insanely fast)
- Average FLOPS: 2.61 GFLOPS

| Component | Time (ms) | GFLOPS | Calls | GFLOPS/sec |
|-----------|-----------|---------|--------|------------|
| Local Swarm Aggregator | 1242.00 | 11.34 | 120 | 9.13 |
| Global Cluster Attention | 573.00 | 0.69 | 40 | 1.21 |
| Broadcast Updater | 447.00 | 3.02 | 40 | 6.76 |
| SwarmFormer Layer | 449.00 | 15.06 | 40 | 33.54 |
| Total | 2711.00 | 30.12 | - | 11.11 |
