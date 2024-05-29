
from datasets import load_dataset
from pathlib import Path
from typer import Typer
from typing import Optional, List

from ...coreference.dataset_builder import generate_coref_dataset
from ...coreference.filtering.lemma_heuristic import LHFilterer
from ...event_tagging.dataset_builder import make_tagger_dataset_dict
from ...trainers.multi_task_trainer import trainer_seq2seq_multi

app = Typer()


@app.command()
def train_summarization_coref(
        config_file: Path, coref_mention_dataset: str, summarization_dataset: str, men_type: Optional[str] = "evt"
):
    # TODO:
    coref_mention_dict = load_dataset(coref_mention_dataset)
    lh_filterer = LHFilterer(coref_mention_dict["train"])

    coref_dataset_dict = generate_coref_dataset(coref_mention_dict, lh_filterer, men_type=men_type)
    summarization_dd = get_xsum()

    train_dataset = concatenate_datasets(coref_dataset_dict["train"], summarization_dd["train"])
    eval_datasets = concatenate_datasets(coref_dataset_dict["dev"], summarization_dd["validation"])


if __name__ == "__main__":
    app()