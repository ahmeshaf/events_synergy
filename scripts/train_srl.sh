#!/bin/bash

CONFIG_PATH=./events_synergy/configs/training/t5_train.json
DATASETS=cu-kairos/propbank_srl_seq2seq
OUTPUT_DIR=./outputs/srl/propbank_srl_seq2seq

python -m events_synergy.trainers.multi_task_trainer \
          $CONFIG_PATH \
          $DATASETS \
          --kv "output_dir=$OUTPUT_DIR"