# Description: Custom Trainer class for multi eval model training

import json
import numpy as np
import torch

from datasets import concatenate_datasets, load_dataset
from evaluate import load
from pathlib import Path
from torch import nn
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.utils import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from ..synergistic_module.dataset_builder import generate_summarized_coreference_dataset
from ..coreference.filtering.lemma_heuristic import LHFilterer

if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction
    from transformers.training_args import TrainingArguments

logger = logging.get_logger(__name__)


def pre_process_eos(dataset, eos_token):
    prompts = [doc for doc in dataset["prompt"]]
    responses = [(doc + " " + eos_token).strip() for doc in dataset["response"]]
    return Dataset.from_dict({"prompt": prompts, "response": responses})


class MultiEvalTrainer(Seq2SeqTrainer):
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
        self.summarize_after_epoch = (summarization_config_file is not None)
        self.eval_datasets = eval_datasets
        self.rouge = load("rouge")

        self.backup_datasets = backup_datasets

        self.eval_is_rouge = dataset2_is_rouge

        self.compute_metrics = compute_metrics
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
        eval_scores = {}
        for dataset_name, eval_dataset in self.eval_datasets.items():
            if self.eval_is_rouge and self.eval_is_rouge[dataset_name]:
                self.compute_metrics = self.compute_rouge
            else:
                self.compute_metrics = self.compute_prf

            print(f"Evaluating on {dataset_name}")
            eval_scores.update(
                super(MultiEvalTrainer, self).evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"eval_{dataset_name}",
                )
            )

        # Code to re-summarize after evaluation.
        if self.summarize_after_epoch:
            self.resummarize_ecb_datasets(eval_scores['epoch'])

        return eval_scores

    def compute_prf(self, eval_pred):
        predictions, labs = eval_pred
        decoded_preds = self.decode_predictions(predictions)
        decoded_labels = self.decode_predictions(labs)

        decoded_preds = [
            set([p.strip() for p in pred.split("|") if p.strip() != ""])
            for pred in decoded_preds
        ]
        decoded_labels = [
            set([p.strip() for p in pred.split("|") if p.strip() != ""])
            for pred in decoded_labels
        ]

        common_preds_lens = [
            len(set.intersection(p1, p2))
            for p1, p2 in zip(decoded_preds, decoded_labels)
        ]
        decoded_preds_lens = [len(p) for p in decoded_preds]
        decoded_labels_lens = [len(p) for p in decoded_labels]

        TP = sum(common_preds_lens)
        FP = sum(decoded_preds_lens) - TP
        FN = sum(decoded_labels_lens) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute_rouge(self, eval_pred):
        predictions, labs = eval_pred
        decoded_preds = self.decode_predictions(predictions)
        decoded_labels = self.decode_predictions(labs)

        return self.rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

    def decode_predictions(self, predictions):
        predictions = np.where(
            predictions != -100, predictions, self.tokenizer.pad_token_id
        )
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        return decoded_preds

    def resummarize_ecb_datasets(self, epoch):
        dataset_dict = load_dataset('ahmeshaf/ecb_plus_mentions')
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
            epoch=epoch
        )
        self.backup_datasets['ecb'] = summarized_coref_dataset

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

        train_tokenized = train_dataset.map(preprocess_data, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
        evals_tokenized = {
            dataset_name: eval_dataset.map(preprocess_data, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            for dataset_name, eval_dataset in eval_datasets.items()
        }

        self.train_dataset = train_tokenized
        self.eval_datasets = evals_tokenized


def trainer_seq2seq_multi(
        config_file: Path,
        datasets_dict: Dict[str, Dict[str, Dataset]],
        summarization_config_file: Optional[Path] = None  # Config specifically for re-summarization
):
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

    train_tokenized = train_dataset.map(preprocess_data, batched=True, fn_kwargs={"tokenizer": tokenizer,
                                                                                  "max_length":
                                                                                      config['generation'][
                                                                                          'max_length']})
    evals_tokenized = {
        dataset_name: eval_dataset.map(preprocess_data, batched=True, fn_kwargs={"tokenizer": tokenizer,
                                                                                 "max_length":
                                                                                     config['generation'][
                                                                                         'max_length']})
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
        summarization_config_file=summarization_config_file
    )
    t5_trainer.train()
    t5_trainer.save_model()


def preprocess_data(examples, tokenizer, max_length=128):
    model_inputs = tokenizer(examples["prompt"], max_length=max_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["response"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
