# Synergetic Event Task Training

## Table of Contents
- [Getting Started](#getting-started)
- [Event Tagging](#event-tagging)
- [Event Coreference](#event-coreference)
- [Event Summarization](#event-summarization)

## Getting Started
- Install the required packages:

```shell
pip install -r requirements.txt
```

## Event Tagging
### Data Format
Check out the huggingface dataset at [ahemshaf/ecb_plus_mentions](https://huggingface.co/datasets/ahmeshaf/ecb_plus_mentions)

### Training
```shell
python -m events_synergy.event_tagging.train_event_tagger \
        ./events_synergy/configs/training/t5_train.json \
        ahemshaf/ecb_plus_mentions
```

## Event Coreference
### Data Format
Check out the huggingface dataset at [ahemshaf/ecb_plus_mentions](https://huggingface.co/datasets/ahmeshaf/ecb_plus_mentions)

### Training
```shell
python -m events_synergy.coreference.train_coref_resolver \
        ./events_synergy/configs/training/t5_train.json \
        ahemshaf/ecb_plus_mentions
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