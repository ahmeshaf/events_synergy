import numpy as np

from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader as torch_DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForSeq2SeqLM, Pipeline, GenerationConfig, T5Tokenizer, T5ForConditionalGeneration,
)
from transformers.pipelines.base import PipelineException


def get_model_tokenizer_generation_config(model_name, is_peft=False):
    if is_peft:
        config = PeftConfig.from_pretrained(model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    return model, tokenizer, generation_config


class EventsPipeline(Pipeline):
    def __init__(self, model, tokenizer, task_prefix, **kwargs):
        self.preprocess_params, self.forward_params, self.postprocess_params = (
            self._sanitize_parameters(**kwargs)
        )
        self.task_prefix = task_prefix
        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        for param_name, param_value in pipeline_parameters.items():
            if "max_length" == param_name:
                preprocess_params[param_name] = param_value
            if "generation_config" == param_name:
                forward_params[param_name] = param_value

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, prompt):
        # Preprocessing: Add the 'triggers: ' prefix to each prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        if not isinstance(prompt, list):
            raise PipelineException(
                "The `prompt` argument needs to be of type `str` or `list`."
            )
        prefixed_prompt = [f"{self.task_prefix}: " + p for p in prompt]
        return self.tokenizer(
            prefixed_prompt,
            truncation=True,
            return_tensors="pt",
            **self.preprocess_params
        )

    def _forward(self, model_inputs, **forward_params):
        # This step is necessary if you need to adjust how the model is called, for example, to modify the forward pass
        return self.model.generate(**model_inputs, **forward_params)

    def postprocess(self, predictions, **postprocess_params):
        # Postprocess the model output if needed
        predictions = np.where(
            predictions != -100, predictions, self.tokenizer.pad_token_id
        )
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        return decoded_preds[0]

    def __call__(self, inputs, **kwargs):
        # Ensure inputs is a list for consistent preprocessing
        return super().__call__(inputs, **kwargs)


def pipe(pipeline: Pipeline, sentences, batch_size=8, desc="Tagging"):
    pipe_dataloader = torch_DataLoader(sentences, batch_size=batch_size)
    outputs = []
    for batch in tqdm(pipe_dataloader, total=len(sentences), desc=desc):
        outputs.extend(pipeline(batch))

    return outputs
