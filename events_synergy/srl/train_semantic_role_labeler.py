from pathlib import Path

from datasets import load_dataset
from typer import Typer
from typing import List

from ..trainers.multi_task_trainer import trainer_seq2seq_multi


app = Typer()


@app.command()
def train(config_file: Path, dataset_names: List[str]):
    """
    Train datasets of the form:
        {
            "prompt": "SRL for [predicate]: sentence with [predicate]",
            "response": ARG-0: [arg0] | ARG-1: [arg1] | ... | ARG-N: [argn]
        }
    :param config_file:
    :param dataset_names:
    :return:
    """
    dataset_names = list(set(dataset_names))
    srl_dataset_dict = {}

    for ds_name in dataset_names:
        srl_dataset_dict[ds_name] = load_dataset(ds_name)

    trainer_seq2seq_multi(
        config_file,
        srl_dataset_dict,
    )


if __name__ == "__main__":
    app()
