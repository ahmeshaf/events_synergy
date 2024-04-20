from typing import List, Optional

from pathlib import Path
from typer import Typer

from .dataset_builder import generate_coref_dataset
from .filtering.lemma_heuristic import save_lh_pairs
from ..trainers.multi_task_trainer import trainer_seq2seq_multi

app = Typer()


@app.command()
def train(config_file: Path, mention_dataset: List[str], men_type: Optional[str] = "evt"):
    dataset_names = list(set(mention_dataset))
    tagger_datasets = {
        ds_name: generate_coref_dataset(ds_name, men_type, save_lh_pairs) for ds_name in dataset_names
    }
    trainer_seq2seq_multi(
        config_file,
        tagger_datasets,
    )


if __name__ == "__main__":
    app()
