from pathlib import Path
from typer import Typer
from typing import List

from ..trainers.multi_task_trainer import trainer_seq2seq_multi
from .dataset_builder import generate_multitask_dataset
from ..summarization.dataset_builder import get_xsum
from ..coreference.dataset_builder import generate_coref_dataset

from datasets import load_dataset
from ..coreference.filtering.lemma_heuristic import LHFilterer

app = Typer()


@app.command()
def train(config_file: Path, dataset_names: List[str]):
    dataset_names = list(set(dataset_names))
    datasetdicts = {}

    for ds_name in dataset_names:
        if ds_name == "xsum":
            datasetdicts['xsum'] = get_xsum()
        elif ds_name == "ecb":
            dataset_dict = load_dataset('ahmeshaf/ecb_plus_mentions')
            lh_filterer = LHFilterer(dataset_dict["train"])
            coref_dataset = generate_coref_dataset(
                dataset_dict, lh_filterer, men_type="evt"
            )
            datasetdicts['ecb'] = coref_dataset

    # dataset_dict = generate_multitask_dataset(datasetdicts, dataset_names)

    trainer_seq2seq_multi(
        config_file,
        datasetdicts,
    )


if __name__ == "__main__":
    app()
