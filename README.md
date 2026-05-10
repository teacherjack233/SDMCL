# Spiking Dual-Memory Continual Learning (SDMCL)

Spiking Dual-Memory Continual Learning (SDMCL) is a task-free continual learning framework based on spiking experts and dual-memory drift detection. It is designed for non-stationary data streams where task boundaries are not provided during training.

The system keeps two memory pools:

- Short memory: a small FIFO buffer for recent stream samples.
- Long memory: a larger buffer for high-value historical samples.

When long memory is not full, incoming samples are stored in long memory and used to train the current expert. After long memory is full, incoming samples are stored in short memory without training. When short memory is full, the model compares the latent distributions of short and long memory. If the distance is above a threshold, a new expert is created. Otherwise, samples are consolidated back into long memory using an information-based selection rule.

## Features

- Task-free continual learning over sequential data streams.
- Dynamic expert expansion.
- Dual-memory capacity management.
- Latent-space drift detection.
- Information-based memory consolidation.
- Automatic checkpoint resume from unfinished stream checkpoints.
- Configurable classifier head, with a fast ANN classifier as the default.

## Installation

Python 3.8 to 3.11 is recommended.

Install the main dependencies:

```bash
pip install torch torchvision tensorboard numpy matplotlib
```

For CUDA support, install the PyTorch build that matches your GPU driver and CUDA version.

## Quick Start

Install dependencies after cloning the repository:

```bash
pip install -r requirements.txt
```

Run with the default MNIST setting:

```bash
python main.py
```

Run Split CIFAR-10:

```bash
python main.py --dataset cifar10
```

Run Split CIFAR-100:

```bash
python main.py --dataset cifar100
```

Run Permuted MNIST:

```bash
python main.py --dataset permuted_mnist
```

## Server Scripts

The `scripts/` directory provides both per-dataset scripts and an all-in-one script.

Run one dataset:

```bash
bash scripts/run_mnist.sh
bash scripts/run_cifar10.sh
bash scripts/run_cifar100.sh
```

Run MNIST, CIFAR-10, and CIFAR-100 sequentially:

```bash
bash scripts/run_all.sh
```

The scripts use:

| Dataset | Local epochs | Classifier | Long memory size | Drift threshold |
| --- | --- | --- | --- | --- |
| MNIST | `20` | `ann` | `2000` | `0.008` |
| CIFAR-10 | `50` | `resnet10` | `1000` | `0.008` |
| CIFAR-100 | `50` | `resnet10` | `5000` | `0.008` |

The scripts use separate checkpoint directories for each dataset:

```text
modelpth/mnist
modelpth/cifar10
modelpth/cifar100
modelpth/permuted_mnist
```

You can override common settings with environment variables:

```bash
GPU=1 DATASET_FRACTION=0.5 CIFAR_N_EPOCHS=50 bash scripts/run_all.sh
```

## Example Command

```bash
python main.py \
  --dataset mnist \
  --dataset_fraction 0.15 \
  --n_epochs 20 \
  --batch_size 64 \
  --stream_batch_size 50 \
  --test_batch_size 10 \
  --short_memory_size 128 \
  --long_memory_size 2000 \
  --threshold 0.008 \
  --classifier_type ann
```

## Common Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `mnist` | Dataset: `mnist`, `cifar10`, `cifar100`, or `permuted_mnist` |
| `--dataset_fraction` | `0.15` | Fraction of the dataset used for training and testing |
| `--n_epochs` | `20` | Local training epochs for the active expert |
| `--batch_size` | `64` | Minibatch size for expert training |
| `--stream_batch_size` | `50` | Batch size of the incoming training stream |
| `--test_batch_size` | `10` | Batch size used by test loaders |
| `--short_memory_size` | `128` | Capacity of the short memory |
| `--long_memory_size` | `2000` | Capacity of the long memory |
| `--threshold` | dataset-specific | Drift threshold for expert expansion |
| `--n_steps` | `16` | Number of spiking time steps |
| `--classifier_type` | `ann` | Classifier head: `ann`, `snn`, `resnet10`, or `resnet18` |
| `--model_dir` | `modelpth` | Directory for resumable checkpoints |
| `--save_dir` | `results` | Directory for experiment logs and outputs |

## Classifier Options

The default classifier is ANN for faster training:

```bash
python main.py --classifier_type ann
```

For MNIST-like grayscale inputs, the ANN classifier is an MLP. For CIFAR-style RGB inputs, it is a lightweight CNN.

To use a spiking classifier:

```bash
python main.py --classifier_type snn
```

To use a ResNet-18 classifier:

```bash
python main.py --classifier_type resnet18
```

To use a Torchvision ResNet-10 classifier:

```bash
python main.py --classifier_type resnet10
```

## Checkpoint Resume

The program checks `modelpth/` before training.

If an unfinished stream checkpoint exists, such as:

```text
modelpth/stream_3_model.pth
```

the program treats streams 1, 2, and 3 as finished and resumes from stream 4. Internally, this means the next zero-based `task_id` is 3.

If only this file exists:

```text
modelpth/model_final.pth
```

the previous run is treated as completed. The program starts a fresh training run instead of resuming.

## Outputs

Training writes outputs to:

```text
results/YYYY-MM-DD/...
modelpth/stream_X_model.pth
modelpth/model_final.pth
```

`results/` stores logs, arguments, and experiment outputs. `modelpth/` stores checkpoints used for automatic resume.

## Project Structure

```text
main.py                     Main training entry point
config/config.py            Command-line arguments
data/dataloaders.py         Split datasets and test loaders
models/component.py         Expert wrapper
models/classifier.py        ANN, SNN, and ResNet classifier heads
models/                     Expert, classifier, and latent model wrappers
utils/memory.py             Dual-memory buffer and drift detection
utils/testing.py            Evaluation utilities
training/trainer.py         Training, checkpoint saving, and checkpoint loading
```

## Notes

- Delete or move `modelpth/stream_X_model.pth` files if you want to force a fresh run.
- `model_final.pth` does not trigger resume.
- Large generated folders such as `data/`, `results/`, and `modelpth/` are ignored by Git.
