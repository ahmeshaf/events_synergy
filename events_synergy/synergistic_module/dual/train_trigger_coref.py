
from datasets import load_dataset
from pathlib import Path
from typer import Typer
from typing import Optional, List

from ...coreference.dataset_builder import generate_coref_dataset
from ...coreference.filtering.lemma_heuristic import LHFilterer
from ...event_tagging.dataset_builder import make_tagger_dataset
from ...trainers.multi_task_trainer import trainer_seq2seq_multi

app = Typer()


@app.command()
def train(
    config_file: Path,
    tagger_mention_datasets: Optional[List[str]] = None,
    coref_mention_datasets: Optional[List[str]] = None,
    filterer: Optional[str] = "lh",
    men_type: Optional[str] = "evt",
):
    dataset_dicts = {}

    if tagger_mention_datasets:
        for ds_name in tagger_mention_datasets:
            dataset_dicts[ds_name] = make_tagger_dataset(ds_name)
    if coref_mention_datasets:
        if filterer == "lh":
            for ds in coref_mention_datasets:
                dataset_dict = load_dataset(ds)
                lh_filterer = LHFilterer(dataset_dict["train"])
                coref_dataset = generate_coref_dataset(
                    dataset_dict, lh_filterer, men_type=men_type
                )
                dataset_dicts[ds] = coref_dataset

    trainer_seq2seq_multi(
        config_file,
        dataset_dicts,
    )


if __name__ == "__main__":
    app()
