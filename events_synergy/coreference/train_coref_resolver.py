from typing import List

from datasets import load_dataset, Dataset
from pathlib import Path
from typer import Typer

from .dataset_builder import CoreferenceDataset
from .filtering.lemma_heuristic import save_lh_pairs
from ..trainers.multi_task_trainer import trainer_seq2seq_multi

app = Typer()


def get_mention_map(dataset_name):
    dataset = load_dataset(dataset_name)
    splits = list(dataset)
    mention_map = {}
    for split in splits:
        dataset[split] = dataset[split].to_pandas()
        split_dict_array = dataset[split].to_dict("records")
        for record in split_dict_array:
            mention_id = record["mention_id"]
            mention_map[mention_id] = record
            mention_map[mention_id]["topic"] = mention_id.split("_")[0]
    return mention_map


def make_coref_dataset(mention_dataset_name, men_type="evt"):
    dataset = load_dataset(mention_dataset_name)
    mention_map = get_mention_map(mention_dataset_name)
    dataset_dict = {}
    splits = list(dataset)
    for split in splits:
        coref_split_dateset = CoreferenceDataset(mention_map, split, save_lh_pairs)
        dataset_dict[split] = coref_split_dateset
    return dataset_dict


@app.command()
def train(config_file: Path, mention_dataset: List[str]):
    dataset_names = list(set(mention_dataset))
    tagger_datasets = {
        ds_name: make_coref_dataset(ds_name) for ds_name in dataset_names
    }
    trainer_seq2seq_multi(
        config_file,
        tagger_datasets,
    )


if __name__ == "__main__":
    app()
