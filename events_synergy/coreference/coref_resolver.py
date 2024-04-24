# The coref resolver uses a custom pipeline to resolve cross-doc coreferences in a text documents.
# The pipeline is initialized with a pre-trained model and tokenizer.
from typing import Callable

import jinja2
from tqdm import tqdm
from transformers.pipelines import Text2TextGenerationPipeline

from .utils import cluster
from ..task_constants import COREF_TEMPLATE


class MentionsCorefResolver(Callable):
    def __init__(
        self,
        model,
        tokenizer,
        generation_config,
        filterer=None,
        context_key="marked_sentence",
    ):
        self.filterer = filterer
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.context_key = context_key
        self.template = jinja2.Template(COREF_TEMPLATE)
        self.coref_pipeline = Text2TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            generation_config=self.generation_config,
        )

    def __call__(self, mention_map, batch_size=32, **kwargs):
        # return self.coref_pipeline(inputs)
        # create mention map
        m_ids = list(mention_map.keys())
        mention_pairs_dataset = self.filterer(mention_map)

        mention_pair_prompts = []

        for m1, m2 in mention_pairs_dataset:
            mention_1 = mention_map[m1]
            mention_2 = mention_map[m2]

            prompt = self.template.render(
                mention_text_1=mention_1["mention_text"],
                mention1_context=mention_1[self.context_key],
                mention_text_2=mention_2["mention_text"],
                mention2_context=mention_2[self.context_key],
            )
            mention_pair_prompts.append(prompt)

        similarities = []

        for i in tqdm(range(0, len(mention_pair_prompts), batch_size)):
            prompt_batch = mention_pair_prompts[i : i + batch_size]
            responses = self.coref_pipeline(prompt_batch, **kwargs)
            for response in responses:
                similarities.append(response["generated_text"] == "Yes")

        clusters = cluster(m_ids, mention_pairs_dataset, similarities)

        return clusters
