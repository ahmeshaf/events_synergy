# The coref resolver uses a custom pipeline to resolve cross-doc coreferences in a text documents.
# The pipeline is initialized with a pre-trained model and tokenizer.
from typing import Callable, List, Dict

import jinja2
from tqdm import tqdm
from transformers.pipelines import Text2TextGenerationPipeline

from .utils import cluster
from ..event_tagging.event_tagger import event_tagger_pipeline
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
            prompt_batch = mention_pair_prompts[i: i + batch_size]
            responses = self.coref_pipeline(prompt_batch, **kwargs)
            for response in responses:
                similarities.append(response["generated_text"] == "Yes")

        clusters = cluster(m_ids, mention_pairs_dataset, similarities)

        return clusters


class DocumentCorefResolver(MentionsCorefResolver):
    def __init__(self, mention_tagger, **kwargs):
        self.mention_tagger = mention_tagger
        super().__init__(**kwargs)

    def __call__(self, docs, **kwargs):
        """

        :param docs: each document is of the form:
            {
                "topic": "topic_1",
                "doc_id": "doc_1",
                "sentences": [
                    {
                        "sentence_id": "s1",
                        "sentence": "This is a sentence",
                    },
                    {
                        "sentence_id": "s2",
                        "sentence": "This is another sentence",
                    },
                    ...
                ]
            }
        :param kwargs:
        :return:
        """
        topic_doc_sentence_ids = [
            (d["topic"], d["doc_id"], s["sentence_id"], s["sentence"])
            for d in docs
            for s in d["sentences"]
        ]
        sentences = [
            sentence["sentence"] for doc in docs for sentence in doc["sentences"]
        ]

        event_triggers = event_tagger_pipeline(
            self.mention_tagger, sentences, batch_size=8
        )

        mention_map = {}
        for (topic, doc_id, sentence_id, sentence), triggers in zip(
                topic_doc_sentence_ids, event_triggers
        ):

            for i, trigger in enumerate(triggers):
                (mention_txt, (start, end)) = trigger
                mention_id = "_".join([topic, doc_id, sentence_id, str(start), str(end)])
                mention_map[mention_id] = {
                    "topic": topic,
                    "doc_id": doc_id,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "mention_text": mention_txt,
                    "start_char": start,
                    "end_char": end,
                    "marked_sentence": sentence[: start]
                                       + " <m> " + mention_txt + " </m> " + sentence[end:],
                    "mention_id": mention_id,
                    "split": "predict",
                    "men_type": "evt",
                    "lemma": mention_txt,
                    "sentence_tokens": sentence.split(),
                }
        return super().__call__(mention_map, **kwargs)

