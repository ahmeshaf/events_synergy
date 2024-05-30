from datasets import load_dataset
from events_synergy.coreference.filtering.lemma_heuristic import LHFilterer

from ..trainers.multi_task_trainer import *


class DynamicMultiEvalTrainer(MultiEvalTrainer):
    def __init__(
        self,
        static_train_dataset: Dataset,
        static_eval_datasets: Dict[str, Dataset],
        **kwargs,
    ):
        self.static_train_dataset = static_train_dataset
        self.static_eval_datasets = static_eval_datasets

        super(MultiEvalTrainer).__init__(
            **kwargs
        )

    def evaluate(
        self,
        **kwargs,
    ) -> Union[Dict[str, float], Dict]:
        """
        Generate the dynamic datasets and evaluate the model on the evaluation datasets.
        Evaluate the model on the evaluation datasets.
        """
        self.generate_static_dynamic_datasets()
        eval_scores = super(MultiEvalTrainer).evaluate(**kwargs)

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
        # TODO 1: changed resummarize_ecb_datasets to generate_static_dynamic_datasets()
        # TODO 2: Use self.coref_mention_map and self.coref_mention_pairs_train and coref_mention_pairs_eval etc
        dataset_dict = load_dataset("ahmeshaf/ecb_plus_mentions") # don't do this after every epoch
        lh_filterer = LHFilterer(dataset_dict["train"])
        summ_config = json.load(open(self.summarization_config_file))

        save_to_wandb = False
        if "report_to" in summ_config["trainer"].keys():
            if summ_config["trainer"]["report_to"] == "wandb":
                save_to_wandb = True

        summarized_coref_dataset = generate_summarized_coreference_dataset(
            self.summarization_config_file,
            self.model,
            self.tokenizer,
            dataset_dict,
            lh_filterer,
            men_type="evt",
            save_to_wandb=save_to_wandb,
            epoch=epoch,
        )
        self.backup_datasets["ecb"] = summarized_coref_dataset
        # Include sentence level coreference data
        train_dataset = concatenate_datasets(
            [
                pre_process_eos(dataset["train"], self.tokenizer.eos_token)
                for dataset in self.backup_datasets.values()
            ]
        )

        eval_datasets = {
            dataset_name: (
                pre_process_eos(dataset["dev"], self.tokenizer.eos_token)
                if "dev" in dataset.keys()
                else pre_process_eos(dataset["validation"], self.tokenizer.eos_token)
            )
            for dataset_name, dataset in self.backup_datasets.items()
        }

        train_tokenized = train_dataset.map(
            preprocess_data, batched=True, fn_kwargs={"tokenizer": self.tokenizer}
        )
        evals_tokenized = {
            dataset_name: eval_dataset.map(
                preprocess_data, batched=True, fn_kwargs={"tokenizer": self.tokenizer}
            )
            for dataset_name, eval_dataset in eval_datasets.items()
        }

        self.train_dataset = train_tokenized
        self.eval_datasets = evals_tokenized


# TODO 3: Refactor this
# def trainer_seq2seq_multi(
#     config_file: Path,
#     coref_dataset_name: str,
#     summarization_dataset_name: str,
#     datasets_dict: Dict[str, Dict[str, Dataset]],
#     summarization_config_file: Optional[
#         Path
#     ] = None,  # Config specifically for re-summarization
# ):
#     ## So instead of a dynamic trainer, that creates summaries after each eval
#     ## We can instead change the problem to generate new training dataset after each eval
#     ## so change the train_dataset after each eval: create a new train_dataset which includes
#     # the sentence level coref prompts, the xsum summaries, and the summary-level coref prompts
#
#     """
#
#     :param config_file:
#     :param datasets_dict: Dictionary of Dictionaries of datasets. Outer Dict = task, Inner Dict = split
#     :param summarization_config_file: optional config file for re-summarization after each epoch.
#         if no config is provided re-summarization is disabled.
#     :return:
#
#     Parameters
#     ----------
#     summarization_config_file
#     """
#     config = json.load(open(config_file))
#
#     summ_config = json.load(open(summarization_config_file))
#
#     model_name_or_path = config.pop("model_name_or_path")
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#
#     # Concatenate the train sets of each dataset
#
#     train_dataset = concatenate_datasets(
#         [
#             pre_process_eos(dataset["train"], tokenizer.eos_token)
#             for dataset in datasets_dict.values()
#         ]
#     )
#
#     eval_datasets = {
#         dataset_name: (
#             pre_process_eos(dataset["dev"], tokenizer.eos_token)
#             if "dev" in dataset.keys()
#             else pre_process_eos(dataset["validation"], tokenizer.eos_token)
#         )
#         for dataset_name, dataset in datasets_dict.items()
#     }
#
#     train_tokenized = train_dataset.map(
#         preprocess_data,
#         batched=True,
#         fn_kwargs={
#             "tokenizer": tokenizer,
#             "max_length": config["generation"]["max_length"],
#         },
#     )
#     evals_tokenized = {
#         dataset_name: eval_dataset.map(
#             preprocess_data,
#             batched=True,
#             fn_kwargs={
#                 "tokenizer": tokenizer,
#                 "max_length": config["generation"]["max_length"],
#             },
#         )
#         for dataset_name, eval_dataset in eval_datasets.items()
#     }
#
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")
#     # from peft import prepare_model_for_kbit_training
#     # from peft import LoraConfig, get_peft_model, TaskType
#     #
#     # model = prepare_model_for_kbit_training(model)
#     #
#     # lora_config = LoraConfig(eval_datasets, ignore_keys, metric_key_prefix, **gen_kwargs
#     #     r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
#     # )
#     #
#     # model = get_peft_model(model, lora_config)
#
#     training_args = Seq2SeqTrainingArguments(**config["trainer"])
#
#     generation_config = GenerationConfig.from_pretrained(model_name_or_path)
#     generation_config.eos_token_id = tokenizer.eos_token_id
#     generation_config.update(**config["generation"])
#
#     training_args.generation_config = generation_config
#     data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
#
#     t5_trainer = MultiEvalTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_tokenized,
#         eval_datasets=evals_tokenized,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         backup_datasets=datasets_dict,
#         summarization_config_file=summarization_config_file,
#     )
#     t5_trainer.train()
#     t5_trainer.save_model()
#
#
# def preprocess_data(examples, tokenizer, max_length=128):
#     model_inputs = tokenizer(examples["prompt"], max_length=max_length, truncation=True)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(examples["response"], max_length=max_length, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
