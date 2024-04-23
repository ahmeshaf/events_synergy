from datasets import load_dataset
from pathlib import Path
from typer import Typer
from typing import List, Optional

from .dataset_builder import generate_coref_dataset
from .filtering.lemma_heuristic import LHFilterer
from ..trainers.multi_task_trainer import trainer_seq2seq_multi

app = Typer()


@app.command()
def train_lh(
    config_file: Path, mention_dataset: List[str], men_type: Optional[str] = "evt"
):
    dataset_names = list(set(mention_dataset))

    coref_datasets = {}

    for ds_name in dataset_names:
        dataset_dict = load_dataset(ds_name)
        lh_filterer = LHFilterer(dataset_dict["train"])
        coref_datasets[ds_name] = generate_coref_dataset(
            dataset_dict, lh_filterer, men_type=men_type
        )

    trainer_seq2seq_multi(
        config_file,
        coref_datasets,
    )


if __name__ == "__main__":
    app()
