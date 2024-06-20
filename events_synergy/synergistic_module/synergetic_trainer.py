from datasets import load_dataset, DatasetDict, interleave_datasets, concatenate_datasets

import events_synergy
from events_synergy.coreference.filtering.lemma_heuristic import LHFilterer
from .. import coreference

from ..trainers.multi_task_trainer import *
from .dataset_builder import generate_summarized_coreference_dataset, tokenize_datasets
from .utils import get_tokenized_multitask_datasets
import events_synergy.coreference as coref

from ..summarization.dataset_builder import get_xsum
from ..coreference.dataset_builder import generate_coref_dataset

from typer import Typer

from typing import Tuple

app = Typer()


class DynamicMultiEvalTrainer(MultiEvalTrainer):
    def __init__(
            self,
            static_train_dataset: Dataset,
            static_eval_datasets: Dict[str, Dataset],
            **kwargs,
    ):
        self.static_train_dataset = static_train_dataset
        self.static_eval_datasets = static_eval_datasets

        super(DynamicMultiEvalTrainer, self).__init__(
            train_dataset=self.static_train_dataset,
            eval_datasets=self.static_eval_datasets,
            **kwargs
        )

    def evaluate(
            self,
            **kwargs,
    ) -> Union[Dict[str, float], Dict]:
        """
        Generate the dynamic datasets and evaluate the model on the evaluation datasets.
        """
        self.generate_static_dynamic_datasets()

        eval_scores = super(DynamicMultiEvalTrainer, self).evaluate(**kwargs)

        return eval_scores

    def generate_static_dynamic_datasets(self):
        raise NotImplementedError("This method should be implemented in the subclass.")


class SummaryCorefTrainer(DynamicMultiEvalTrainer):
    def __init__(
            self,
            coref_mention_map: dict,
            coref_mention_pairs_train: List[Tuple[str, str]],
            coref_mention_pairs_eval: List[Tuple[str, str]],
            summarization_config_file: Path,
            **kwargs,
    ):
        self.coref_mention_map = coref_mention_map
        self.coref_mention_pairs_train = coref_mention_pairs_train
        self.coref_mention_pairs_eval = coref_mention_pairs_eval
        self.summarization_config_file = summarization_config_file
        super(SummaryCorefTrainer, self).__init__(
            **kwargs
        )

    def generate_static_dynamic_datasets(self):
        summ_config = json.load(open(self.summarization_config_file))

        summarized_coref_dataset = generate_summarized_coreference_dataset(
            self.summarization_config_file,
            self.model,
            self.tokenizer,
            self.coref_mention_map,
            self.coref_mention_pairs_train,
            self.coref_mention_pairs_eval,
            men_type="evt",
            save_to_wandb=("report_to" in summ_config["trainer"].keys()) and (
                    summ_config["trainer"]["report_to"] == "wandb"),
            epoch=self.state.epoch / 100,
        )

        tokenized_summarized_coref_dataset = \
        tokenize_datasets({"ecb_summ": summarized_coref_dataset}, self.tokenizer)

        self.train_dataset = interleave_datasets(
            [self.static_train_dataset, tokenized_summarized_coref_dataset[0]]
        )

        # Combine static datasets with newly generated one
        self.eval_datasets = self.static_eval_datasets | tokenized_summarized_coref_dataset[1]

        print(self.eval_datasets)


def train_coref_summarizer(
        config_file: Path,
        datasets_dict: Dict[str, Dict[str, Dataset]],
        summarization_config_file: Path,
        men_type: str = "evt",
):
    config = json.load(open(config_file))

    model_name_or_path = config.pop("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Generate datasets
    train_tokenized, evals_tokenized = get_tokenized_multitask_datasets(datasets_dict, tokenizer, config)

    # In the future add support for other datasets/filterers
    coref_dataset = load_dataset('ahmeshaf/ecb_plus_mentions')
    filterer = LHFilterer(coref_dataset["train"])

    splitwise_mention_maps = {split: coref.utils.get_mention_map(coref_dataset[split], men_type) for split in
                              coref_dataset.keys()}

    splitwise_mention_pairs = {split: filterer(splitwise_mention_maps[split]) for split in
                               splitwise_mention_maps.keys()}

    mention_map = splitwise_mention_maps['train'] | splitwise_mention_maps['dev']

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")

    training_args = Seq2SeqTrainingArguments(**config["trainer"])

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.update(**config["generation"])

    training_args.generation_config = generation_config
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    t5_trainer = SummaryCorefTrainer(
        model=model,
        args=training_args,
        static_train_dataset=train_tokenized,
        static_eval_datasets=evals_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        coref_mention_map=mention_map,
        coref_mention_pairs_train=splitwise_mention_pairs['train'],
        coref_mention_pairs_eval=splitwise_mention_pairs['dev'],
        summarization_config_file=summarization_config_file,
    )
    t5_trainer.train()
    t5_trainer.save_model()


@app.command()
def train(
        config_file: Path,
        dataset_names: List[str],
        debug: bool = False,
        summarization_config_file: str = Option(None, "--summ-config"),
        kv: str = Option(
            None,
            "--kv",
            help="Key-value pairs, separated by commas, e.g., key1=value1,key2=value2",
        ),
):
    """
    Train datasets of the form:
        {
            "prompt": "SRL for [predicate]: sentence with [predicate]",
            "response": ARG-0: [arg0] | ARG-1: [arg1] | ... | ARG-N: [argn]
        }
    :param config_file:
    :param dataset_names:
    :param debug: If True, only train on a small subset of the data
    :param summarization_config_file: optional summarization specific config file
    :param kv: override config file parameters with this. e.g., "num_train_epochs=20,per_device_train_batch_size=8"
    :return:
    """
    dataset_names = list(set(dataset_names))
    dataset_dict = {}

    for ds_name in dataset_names:
        if ds_name == "xsum":
            dataset_dict['xsum'] = get_xsum()
        elif ds_name == "ecb":
            ecb_dataset_dict = load_dataset('ahmeshaf/ecb_plus_mentions')
            lh_filterer = LHFilterer(ecb_dataset_dict["train"])
            coref_dataset = generate_coref_dataset(
                ecb_dataset_dict, lh_filterer, men_type="evt"
            )
            dataset_dict['ecb'] = coref_dataset

    if debug:
        for dataset_key in dataset_dict.keys():
            for split_key in dataset_dict[dataset_key].keys():
                dataset_dict[dataset_key][split_key] = dataset_dict[dataset_key][split_key][:100]

    kv_dict = {}

    if kv:
        try:
            kv_dict = parse_kv(kv)
            echo("Received key-value arguments:")
            for key, value in kv_dict.items():
                echo(f"{key}: {value}")
        except ValueError as e:
            echo(f"Error: {e}")
    else:
        echo("No key-value arguments provided.")

    train_coref_summarizer(
        config_file,
        dataset_dict,
        summarization_config_file if summarization_config_file is not None else config_file,
        **kv_dict
    )


if __name__ == "__main__":
    app()
