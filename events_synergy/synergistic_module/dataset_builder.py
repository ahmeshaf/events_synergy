import json

from datasets import concatenate_datasets, Dataset, DatasetDict
from jinja2 import Template
from pathlib import Path
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Callable, List

from ..coreference.utils import get_mention_map
from ..task_constants import COREF_TEMPLATE, SUMMARIZATION_TEMPLATE

import pathlib

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
        mention_dataset_dict: DatasetDict,
        filterer: Callable,
        text_key="marked_document",
        men_type: str = "evt",
        save_to_wandb: bool = False,
        epoch = 0
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
    splits = list(mention_dataset_dict)
    split2dataset = {}
    for split in splits:
        mention_map = get_mention_map(mention_dataset_dict[split], men_type)

        resum_data = {
            'document': [summarization_template.render(document=mention_map[k]['marked_doc']) for k in mention_map.keys()],
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
            with open(RESUMMARIZATION_SAVES_DIR / f"mention_map_epoch_{epoch}.json", "w") as f:
                json.dump(mention_map.to_json(), f)

            wandb.save(RESUMMARIZATION_SAVES_DIR / f"mention_map_epoch_{epoch}.json")

        mention_pairs_dataset = filterer(mention_map)
        prompt_responses = []
        for m1, m2 in mention_pairs_dataset:
            mention_1 = mention_map[m1]
            mention_2 = mention_map[m2]

            prompt = coref_template.render(
                mention_text_1=mention_1["mention_text"],
                mention1_context=mention_1['summary'],
                mention_text_2=mention_2["mention_text"],
                mention2_context=mention_2['summary'],
            )

            response = (
                "Yes"
                if mention_1["gold_cluster"] == mention_2["gold_cluster"]
                else "No"
            )

            prompt_responses.append({"prompt": prompt, "response": response})

        split2dataset[split] = Dataset.from_list(prompt_responses)

    return DatasetDict(split2dataset)
