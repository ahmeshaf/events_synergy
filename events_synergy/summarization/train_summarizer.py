from pathlib import Path
from typing import List

from typer import Typer

from ..trainers.multi_task_trainer import trainer_seq2seq_multi
from .dataset_builder import get_entsum, get_xsum

app = Typer()


@app.command()
def train(config_file: Path, dataset_names: List[str]):
    dataset_names = list(set(dataset_names))
    summ_dataset_dict = {}

    for ds_name in dataset_names:
        if ds_name == "xsum":
            summ_dataset_dict[ds_name] = get_xsum()
        elif ds_name == "entsum":
            summ_dataset_dict[ds_name] = get_entsum()

    trainer_seq2seq_multi(
        config_file,
        summ_dataset_dict,
    )


if __name__ == "__main__":
    app()
