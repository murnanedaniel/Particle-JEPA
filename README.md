# Particle-JEPA

A Joint-Embedding Predictive Architecture (JEPA) implementation for particle physics track reconstruction. This project implements a self-supervised learning approach to learn meaningful representations of particle trajectories in detector data.

## Overview

Particle-JEPA uses a novel architecture that learns to predict embeddings of target detector regions from context regions. The model consists of:
- A context encoder that processes input detector hits
- A target encoder (updated via EMA) that provides target embeddings
- A predictor network that maps context embeddings to target embedding space

The model learns by minimizing the distance between predicted and actual target embeddings, effectively learning the underlying physics of particle trajectories without explicit supervision.

## Features

- Self-supervised learning approach
- Transformer-based architecture for processing detector hits
- Geometric patch sampling using wedges/annuli of the detector
- Flexible embedding dimension and model architecture
- Support for both single-GPU and distributed training
- Wandb integration for experiment tracking

## Installation

```bash
# Clone repository with submodules
git clone https://github.com/murnanedaniel/Particle-JEPA.git --recurse-submodules
cd Particle-JEPA

# Create and activate conda environment
conda create -n jepa python=3.10
conda activate jepa

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training

To train the model:
```bash
# For local training
python run.py --config configs/your_config.yaml

# For SLURM-based training
sbatch batch/slurm_submit_job.sh --config configs/your_config.yaml
```

### Configuration

The model can be configured through YAML files in the `configs/` directory. Key parameters include:
- `d_input`: Input dimension of detector hits
- `d_model`: Internal dimension of transformer
- `d_embedding`: Final embedding dimension
- `n_layers`: Number of transformer layers
- `heads`: Number of attention heads
- `batch_size`: Training batch size
- `encoder_lr`: Learning rate for encoder
- `predictor_lr`: Learning rate for predictor
- `ema_decay`: Decay rate for target encoder updates

## Model Architecture

The JEPA model consists of three main components:
1. **Context Encoder**: Processes input detector hits and produces context embeddings
2. **Target Encoder**: Creates target embeddings (updated via exponential moving average)
3. **Predictor**: Maps context embeddings to predict target embeddings

The model uses a Smooth L1 loss to minimize the distance between predicted and actual target embeddings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

Pending

## Contact

Daniel Murnane, daniel.murnane@nbi.ku.dk