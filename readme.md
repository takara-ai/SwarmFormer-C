# SwarmFormer.c
[PAPER](https://takara.ai/papers/SwarmFormer-Local-Global-Hierarchical-Attention-via-Swarming-Token-Representations.pdf) | [PYTORCH](https://github.com/takara-ai/SwarmFormer)

⚠️ **WARNING: AVX2 SUPPORT IS STRICTLY REQUIRED - THE MODEL WILL NOT RUN WITHOUT IT!** ⚠️

SwarmFormer.c is a C implementation of the SwarmFormer architecture.

## Prerequisites


### For converting the weights
- Python 3.9+
- 2-3 GB of free disk space (for converting the weights, PyTorch is large)

### For running the model
- At least 50MB of free RAM (for running the model)
- AVX2 support

## Setup

The following guide is for building the inference engine from source. See [setup.md](setup.md) for more information.

## Usage
For a guide on how to use the inference engine, see [usage.md](usage.md)

## Citing

If you use this code in your research, please cite:

```bibtex
@article{legg2025swarmformer,
  title={SwarmFormer: Local-Global Hierarchical Attention via Swarming Token Representations},
  author={Legg, Jordan and Sturmanis, Mikus and {Takara.ai}},
  journal={Takara.ai Research},
  year={2025},
  url={https://takara.ai/papers/SwarmFormer-Local-Global-Hierarchical-Attention-via-Swarming-Token-Representations.pdf}
}
```

## Contact
For questions, collaborations, or other inquiries:

- Email: research@takara.ai
- Repository: [https://github.com/takara-ai/SwarmFormer](https://github.com/takara-ai/SwarmFormer-C)
