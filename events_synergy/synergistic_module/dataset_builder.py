import json

from datasets import concatenate_datasets, Dataset, DatasetDict
from jinja2 import Template
from pathlib import Path
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Callable, List, Tuple

from ..coreference.utils import get_mention_map
from ..task_constants import COREF_TEMPLATE, SUMMARIZATION_TEMPLATE, COREF_POSITIVE_LABEL, COREF_NEGATIVE_LABEL

import pathlib
import os

import numpy as np

import wandb

RESUMMARIZATION_SAVES_DIR = (pathlib.Path(__file__).parent.parent / "data/resummarization_saves").resolve()


def generate_multitask_dataset(datasets: List[DatasetDict], dataset_names: List[str]):
    """
    :return: DatasetDict of the form ds['train], ds['validation_{task}'], ds['test_{task}']
    """

    final_ds = DatasetDict()

    # Shuffle the training data from all tasks
    final_ds['train'] = concatenate_datasets([dataset['train'] for dataset in datasets]).shuffle()

    for dataset, name in zip(datasets, dataset_names):
        final_ds[f'validation_{name}'] = dataset['validation']
        final_ds[f'test_{name}'] = dataset['test']

    return final_ds


def generate_summarized_coreference_dataset(
        config_file: Path,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        mention_map: dict,
        mention_pairs_train: List[Tuple[str, str]],
        mention_pairs_eval: List[Tuple[str, str]],
        men_type: str = "evt",
        save_to_wandb: bool = False,
        epoch=0
):
    """

    :param mention_dataset_dict:
    :param men_type: can be "evt" or "ent" or "all"
    :param filterer:
    :param text_key:
    :return: DatasetDict
    """
    config = json.load(open(config_file))

    coref_template = Template(COREF_TEMPLATE)
    summarization_template = Template(SUMMARIZATION_TEMPLATE)

    # Get list of mention ids to summarize
    unique_mention_ids = list(set([id for tup in mention_pairs_train for id in tup] +
                                  [id for tup in mention_pairs_eval for id in tup]))

    # Reduced mention map so we only summarize train + eval data
    mention_map_temp = {k: mention_map[k] for k in unique_mention_ids}

    resum_data = {
        'document': [summarization_template.render(document=mention_map_temp[k]['marked_doc']) for k in
                     mention_map_temp.keys()],
        'id': [k for k in mention_map.keys()]
    }
    batch_size = config['batch_size']

    summaries = []

    # Generate outputs in batches
    print("Re-summarizing ECB documents...")
    for i in tqdm(range(0, len(resum_data['document']), batch_size)):
        batch_input = resum_data['document'][i:i + batch_size]

        # Tokenize batch input
        inputs = tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # Generate output
        outputs = model.generate(**inputs, **config['generation'])

        # Decode output
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Append to output strings list
        summaries.extend(output_texts)

    for i, k in enumerate(mention_map.keys()):
        mention_map[k]['summary'] = summaries[i]

    if save_to_wandb:
        os.makedirs(RESUMMARIZATION_SAVES_DIR, exist_ok=True)
        with open(RESUMMARIZATION_SAVES_DIR / f"mention_map_epoch_{epoch}.json", "w") as f:
            json.dump(mention_map, f, cls=json_serialize)

        wandb.save(RESUMMARIZATION_SAVES_DIR / f"mention_map_epoch_{epoch}.json")

    train_prompt_response_pairs = []
    eval_prompt_response_pairs = []
    for m1, m2 in mention_pairs_train:
        mention_1 = mention_map[m1]
        mention_2 = mention_map[m2]

        prompt = coref_template.render(
            mention_text_1=mention_1["mention_text"],
            mention1_context=mention_1['summary'],
            mention_text_2=mention_2["mention_text"],
            mention2_context=mention_2['summary'],
        )

        response = (
            COREF_POSITIVE_LABEL
            if mention_1["gold_cluster"] == mention_2["gold_cluster"]
            else COREF_NEGATIVE_LABEL
        )

        train_prompt_response_pairs.append({"prompt": prompt, "response": response})

    for m1, m2 in mention_pairs_eval:
        mention_1 = mention_map[m1]
        mention_2 = mention_map[m2]

        prompt = coref_template.render(
            mention_text_1=mention_1["mention_text"],
            mention1_context=mention_1['summary'],
            mention_text_2=mention_2["mention_text"],
            mention2_context=mention_2['summary'],
        )

        response = (
            COREF_POSITIVE_LABEL
            if mention_1["gold_cluster"] == mention_2["gold_cluster"]
            else COREF_NEGATIVE_LABEL
        )

        eval_prompt_response_pairs.append({"prompt": prompt, "response": response})

    split2dataset = {"train": Dataset.from_list(train_prompt_response_pairs),
                     "dev": Dataset.from_list(eval_prompt_response_pairs)}

    return DatasetDict(split2dataset)


# Custom JSON encoder for handling NP objects in the mention map
class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def pre_process_eos(dataset, eos_token):
    prompts = [doc for doc in dataset["prompt"]]
    responses = [(doc + " " + eos_token).strip() for doc in dataset["response"]]
    return Dataset.from_dict({"prompt": prompts, "response": responses})


def preprocess_data(examples, tokenizer: T5Tokenizer, config: dict):
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=config["max_input_length"],
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["response"],
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_datasets(
        datasets: dict,
        tokenizer: T5Tokenizer
):
    train_dataset = concatenate_datasets(
        [
            pre_process_eos(dataset["train"], tokenizer.eos_token)
            for dataset in datasets.values()
        ]
    )

    eval_datasets = {
        dataset_name: (
            pre_process_eos(dataset["dev"], tokenizer.eos_token)
            if "dev" in dataset.keys()
            else pre_process_eos(dataset["validation"], tokenizer.eos_token)
        )
        for dataset_name, dataset in datasets.items()
    }

    train_tokenized = train_dataset.map(
        preprocess_data, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    evals_tokenized = {
        dataset_name: eval_dataset.map(
            preprocess_data, batched=True, fn_kwargs={"tokenizer": tokenizer}
        )
        for dataset_name, eval_dataset in eval_datasets.items()
    }

    return train_tokenized, evals_tokenized
