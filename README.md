# Synergetic Event Task Training

## Training Scripts

- finetune ent-sum
```sh
python events_synergy/trainers/multi_task_trainer.py events_synergy/configs/training/t5_train.json events-synergy/entsum_processed
```

## Table of Contents
- [Getting Started](#getting-started)
- [Event Tagging](#event-tagging)
- [Semantic Role Labeling](#semantic-role-labeling)
- [Event Coreference](#event-coreference)
- [Event Summarization](#event-summarization)

## Getting Started
- Install the required packages:

```shell
pip install -r requirements.txt
```

## Event Tagging
### Data Format
Check out the huggingface dataset at [ahmeshaf/ecb_plus_mentions](https://huggingface.co/datasets/ahmeshaf/ecb_plus_mentions)

### Training
```shell
python -m events_synergy.event_tagging.train_event_tagger \
        ./events_synergy/configs/training/t5_train.json \
        ahmeshaf/ecb_plus_mentions
```

## Semantic Role Labeling
### Data Format
Check out the huggingface dataset at [cu-kairos/propbank_srl_seq2seq](https://huggingface.co/datasets/cu-kairos/propbank_srl_seq2seq)


## Event Coreference
### Data Format
Check out the huggingface dataset at [ahmeshaf/ecb_plus_mentions](https://huggingface.co/datasets/ahmeshaf/ecb_plus_mentions)

### Training
```shell
python -m events_synergy.coreference.train_coref_resolver \
        ./events_synergy/configs/training/t5_train.json \
        ahmeshaf/ecb_plus_mentions
        --men-type evt
```

## Event Summarization
### Data Format
Check out the huggingface dataset at [EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)

### Training
```shell
python -m events_synergy.summarization.train_summarizer \
        ./events_synergy/configs/training/summ_train.json \
        xsum
```

## Running with Lepton.ai
- Install the required packages:

```shell
pip install -U leptonai
lep login
```

- Train SRL with Lepton.ai

```shell
export TRAIN_COMMANDS='cd /home/workspace/events_synergy \n
git pull \n
pip install . \n
chmod +x ./scripts/train_srl.sh \n
./scripts/train_srl.sh'
```
```shell
lep job create \
--resource-shape gpu.a10 \
--completions 1 \
--parallelism 1 \
--container-image default/lepton:photon-py3.11-runner-0.15.0 \
--command "$TRAIN_COMMANDS" \
--intra-job-communication=true \
--name train-srl-1
```
`--resource-shape` can be the following:
    
    1. gpu.a10 : $1.22/hr
    2. gpu.a100-40gb : $3.048/hr
    3. gpu.a100-80gb : $3.21/hr
    4. gpu.h100-pcie : $3.9/hr 
    5. gpu.h100-sxm : $4.2/hr

## Running with runpod.io

- Install the required packages:

```shell
pip install -U runpod
runpod config
```

- Launch a pod with runpod.io

```shell
python -m events_synergy.runpod test_a40 --gpu-type-id "NVIDIA A40"
```

`--gpu-type-id` can be the following:
    
    1. NVIDIA A30
    2. NVIDIA A40
    3. NVIDIA A100 40GB
    4. NVIDIA A100 80GB
    5. NVIDIA H100 PCIe
    6. NVIDIA H100 SXM
