from datasets import load_dataset
from events_synergy.coreference.filtering.lemma_heuristic import LHFilterer

from ..trainers.multi_task_trainer import *

from ..coreference.dataset_builder import generate_coref_dataset
from ..summarization.dataset_builder import get_xsum

class SummaryCorefTrainer(MultiEvalTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_datasets: Optional[Dict[str, Union[Dataset, Dict[str, Dataset]]]] = None,
        dataset2_is_rouge: Optional[Dict[str, bool]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        backup_datasets: Dict[str, Dict[str, Dataset]] = None,
        summarization_config_file: Path = None,
    ):
        self.summarization_config_file = summarization_config_file
        self.summarize_after_eval = summarization_config_file is not None
        self.backup_datasets = backup_datasets

        super(MultiEvalTrainer).__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_datasets,
            dataset2_is_rouge,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def evaluate(
        self,
        eval_datasets: Optional[Dict[str, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Union[Dict[str, float], Dict]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_datasets Optional[Dict[str, Dataset]]:
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
            :param task_name:
        """
        eval_scores = super(MultiEvalTrainer).evaluate(eval_datasets, ignore_keys, metric_key_prefix, **gen_kwargs)

        self.resummarize_ecb_datasets(eval_scores["epoch"])

        return eval_scores

    def resummarize_ecb_datasets(self, epoch):
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





def trainer_seq2seq_multi(
    config_file: Path,
    coref_dataset_name: str,
    summarization_dataset_name: str,
    datasets_dict: Dict[str, Dict[str, Dataset]],
    summarization_config_file: Optional[
        Path
    ] = None,  # Config specifically for re-summarization
):
    ## So instead of a dynamic trainer, that creates summaries after each eval
    ## We can instead change the problem to generate new training dataset after each eval
    ## so change the train_dataset after each eval: create a new train_dataset which includes
    # the sentence level coref prompts, the xsum summaries, and the summary-level coref prompts

    """

    :param config_file:
    :param datasets_dict: Dictionary of Dictionaries of datasets. Outer Dict = task, Inner Dict = split
    :param summarization_config_file: optional config file for re-summarization after each epoch.
        if no config is provided re-summarization is disabled.
    :return:

    Parameters
    ----------
    summarization_config_file
    """
    config = json.load(open(config_file))

    summ_config = json.load(open(summarization_config_file))

    model_name_or_path = config.pop("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Concatenate the train sets of each dataset

    train_dataset = concatenate_datasets(
        [
            pre_process_eos(dataset["train"], tokenizer.eos_token)
            for dataset in datasets_dict.values()
        ]
    )

    eval_datasets = {
        dataset_name: (
            pre_process_eos(dataset["dev"], tokenizer.eos_token)
            if "dev" in dataset.keys()
            else pre_process_eos(dataset["validation"], tokenizer.eos_token)
        )
        for dataset_name, dataset in datasets_dict.items()
    }

    train_tokenized = train_dataset.map(
        preprocess_data,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config["generation"]["max_length"],
        },
    )
    evals_tokenized = {
        dataset_name: eval_dataset.map(
            preprocess_data,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": config["generation"]["max_length"],
            },
        )
        for dataset_name, eval_dataset in eval_datasets.items()
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")
    # from peft import prepare_model_for_kbit_training
    # from peft import LoraConfig, get_peft_model, TaskType
    #
    # model = prepare_model_for_kbit_training(model)
    #
    # lora_config = LoraConfig(
    #     r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    # )
    #
    # model = get_peft_model(model, lora_config)

    training_args = Seq2SeqTrainingArguments(**config["trainer"])

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.update(**config["generation"])

    training_args.generation_config = generation_config
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    t5_trainer = MultiEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_datasets=evals_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        backup_datasets=datasets_dict,
        summarization_config_file=summarization_config_file,
    )
    t5_trainer.train()
    t5_trainer.save_model()


def preprocess_data(examples, tokenizer, max_length=128):
    model_inputs = tokenizer(examples["prompt"], max_length=max_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["response"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
