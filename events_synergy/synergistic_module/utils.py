import os
import numpy as np
import torch

from datasets import concatenate_datasets
from transformers import T5Tokenizer

from datasets import Dataset
from pathlib import Path
from typing import Dict, List, Tuple


def get_tokenized_multitask_datasets(datasets_dict: Dict[str, Dict[str, Dataset]],
                                     tokenizer: T5Tokenizer,
                                     config: Dict):
    train_dataset = concatenate_datasets(
        [
            pre_process_eos(dataset["train"], tokenizer.eos_token)
            for dataset in datasets_dict.values()
        ]
    )

    eval_datasets = {
        dataset_name: (
            pre_process_eos(dataset["dev"], tokenizer.eos_token)
            if "dev" in dataset.keys()
            else pre_process_eos(dataset["validation"], tokenizer.eos_token)
        )
        for dataset_name, dataset in datasets_dict.items()
    }

    train_tokenized = train_dataset.map(
        preprocess_data,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config["generation"]["max_length"],
        },
    )
    evals_tokenized = {
        dataset_name: eval_dataset.map(
            preprocess_data,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": config["generation"]["max_length"],
            },
        )
        for dataset_name, eval_dataset in eval_datasets.items()
    }


def pre_process_eos(dataset, eos_token):
    prompts = [doc for doc in dataset["prompt"]]
    responses = [(doc + " " + eos_token).strip() for doc in dataset["response"]]
    return Dataset.from_dict({"prompt": prompts, "response": responses})


def preprocess_data(examples, tokenizer, max_length=128):
    model_inputs = tokenizer(examples["prompt"], max_length=max_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["response"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
