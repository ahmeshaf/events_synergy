from ..trainers.multi_task_trainer import trainer_seq2seq_multi

from typer import Typer
from pathlib import Path
from typing import List

from .dataset_builder import get_xsum

app = Typer()


@app.command()
def train(config_file: Path, dataset_name: List[str]):
    dataset_names = list(set(dataset_name))
    tagger_datasets = {
        ds_name: get_xsum() for ds_name in dataset_names
    }
    trainer_seq2seq_multi(
        config_file,
        tagger_datasets,
    )


if __name__ == "__main__":
    app()