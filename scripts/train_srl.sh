#!/bin/bash

CONFIG_PATH=./events_synergy/configs/training/t5_train.json
DATASETS=$1
OUTPUT_DIR=$2

python -m events_synergy.trainers.multi_task_trainer \
          $CONFIG_PATH \
          $DATASETS \
          --kv "output_dir=$OUTPUT_DIR"