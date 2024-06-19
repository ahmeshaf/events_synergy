from datasets import load_dataset, DatasetDict, interleave_datasets, concatenate_datasets
from events_synergy.coreference.filtering.lemma_heuristic import LHFilterer
from .. import coreference

from ..trainers.multi_task_trainer import *
from .dataset_builder import generate_summarized_coreference_dataset, tokenize_datasets
from .utils import get_tokenized_multitask_datasets
import events_synergy.coreference as coref

from typer import Typer

import pathlib
import pickle as pkl

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

        super(DynamicMultiEvalTrainer).__init__(
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
        eval_scores = super(DynamicMultiEvalTrainer).evaluate(**kwargs)

        return eval_scores

    def generate_static_dynamic_datasets(self):
        raise NotImplementedError("This method should be implemented in the subclass.")


class SummaryCorefTrainer(MultiEvalTrainer):
    def __init__(
            self,
            coref_mention_map: dict,
            coref_mention_pairs_train: List[Tuple[str, str]],
            coref_mention_pairs_eval: List[Tuple[str, str]],
            **kwargs,
    ):
        self.coref_mention_map = coref_mention_map
        self.coref_mention_pairs_train = coref_mention_pairs_train
        self.coref_mention_pairs_eval = coref_mention_pairs_eval

        super(SummaryCorefTrainer).__init__(
            **kwargs
        )

    def generate_static_dynamic_datasets(self, epoch):
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
            epoch=epoch,
        )

        self.train_dataset = interleave_datasets(
            [self.static_train_dataset, pre_process_eos(summarized_coref_dataset['train'], self.tokenizer.eos_token)]
        )

        self.eval_datasets = self.static_eval_datasets.update(
            {"ecb_summ": pre_process_eos(summarized_coref_dataset['dev'], self.tokenizer.eos_token)}
        )


@app.command()
def train_coref_summarizer(
        config_file: Path,
        datasets_dict: Dict[str, Dict[str, Dataset]],
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

    splitwise_mention_pairs = {split: filterer(splitwise_mention_maps) for split in splitwise_mention_maps.keys()}

    mention_map = dict(splitwise_mention_pairs['train'] + splitwise_mention_pairs['dev'])

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
        mention_map=mention_map,
        coref_mention_pairs_train=splitwise_mention_pairs['train'],
        coref_mention_pairs_eval=splitwise_mention_pairs['dev'],
    )
    t5_trainer.train()
    t5_trainer.save_model()
