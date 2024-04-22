import numpy as np

from transformers import (
    Pipeline,
)
from transformers.pipelines.base import PipelineException


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
        decoded_pred = [
            s.strip() for s in decoded_preds[0].split("|") if s.strip() != ""
        ]
        return decoded_pred

    def __call__(self, inputs, **kwargs):
        # Ensure inputs is a list for consistent preprocessing
        return super().__call__(inputs, **kwargs)
