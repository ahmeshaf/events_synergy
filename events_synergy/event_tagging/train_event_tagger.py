from pathlib import Path
from typer import Typer
from typing import List

from .dataset_builder import make_tagger_dataset
from ..trainers.multi_task_trainer import trainer_seq2seq_multi

app = Typer()


@app.command()
def train(config_file: Path, mention_dataset: List[str]):
    dataset_names = list(set(mention_dataset))
    tagger_datasets = {
        ds_name: make_tagger_dataset(ds_name) for ds_name in dataset_names
    }
    trainer_seq2seq_multi(
        config_file,
        tagger_datasets,
    )


if __name__ == "__main__":
    app()
