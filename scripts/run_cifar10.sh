#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
DATASET_FRACTION="${DATASET_FRACTION:-0.15}"
N_EPOCHS="${N_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STREAM_BATCH_SIZE="${STREAM_BATCH_SIZE:-50}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-10}"
SHORT_MEMORY_SIZE="${SHORT_MEMORY_SIZE:-128}"
CLASSIFIER_TYPE="${CLASSIFIER_TYPE:-resnet10}"
SAVE_DIR="${SAVE_DIR:-results}"

python main.py \
  --dataset cifar10 \
  --gpu "$GPU" \
  --dataset_fraction "$DATASET_FRACTION" \
  --n_epochs "$N_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --stream_batch_size "$STREAM_BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --short_memory_size "$SHORT_MEMORY_SIZE" \
  --long_memory_size 1000 \
  --threshold 0.008 \
  --classifier_type "$CLASSIFIER_TYPE" \
  --model_dir modelpth/cifar10 \
  --save_dir "$SAVE_DIR"
