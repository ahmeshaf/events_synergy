import pytest
from transformers import GenerationConfig, T5ForConditionalGeneration, T5Tokenizer

import json
import pathlib

from events_synergy.coreference.filtering import lemma_heuristic
from events_synergy.synergistic_module.synergistic_analysis import (
    SummarizerDocumentCorefResolver,
)
from datasets import Dataset

from events_synergy.coreference.utils import get_mention_map
from events_synergy.events_pipeline import EventsPipeline
from events_synergy.task_constants import TRIGGERS_PREFIX

CONFIG_ROOT = (pathlib.Path(__file__).parent.parent / "configs/inference").resolve()

TRAIN_DATASET = [
    {
        "mention_id": "m1",
        "doc_id": "doc1",
        "sentence_id": "s1",
        "topic": 1,
        "split": "train",
        "lemma": "like",
        "mention_text": "I like this",
        "sentence_tokens": ["I", "like", "this"],
        "gold_cluster": "1",
        "men_type": "evt",
        "marked_doc": "I <m> like </m> this",
    },
    {
        "mention_id": "m2",
        "doc_id": "doc2",
        "sentence_id": "s2",
        "topic": 1,
        "split": "train",
        "lemma": "love",
        "mention_text": "I love this",
        "sentence_tokens": ["I", "love", "this"],
        "gold_cluster": "1",
        "men_type": "evt",
        "marked_doc": "I <m> love </m> this",
    },
    {
        "mention_id": "m3",
        "doc_id": "doc2",
        "sentence_id": "s2",
        "topic": 1,
        "split": "train",
        "lemma": "I",
        "mention_text": "I",
        "sentence_tokens": ["I", "love", "this"],
        "gold_cluster": "ent_1",
        "men_type": "ent",
        "marked_doc": "<m> I </m> love this",
    },
    {
        "mention_id": "m4",
        "doc_id": "doc1",
        "sentence_id": "s2",
        "topic": 1,
        "split": "train",
        "lemma": "this",
        "mention_text": "I",
        "sentence_tokens": ["I", "like", "this"],
        "gold_cluster": "ent_2",
        "men_type": "ent",
        "marked_doc": "I like <m> this </m>",
    },
]

TEST_DATASET = [
    {
        "mention_id": "m1",
        "doc_id": "doc1",
        "sentence_id": "s1",
        "topic": 1,
        "split": "test",
        "lemma": "like",
        "mention_text": "like",
        "sentence_tokens": ["I", "like", "this"],
        "men_type": "evt",
        "marked_doc": "I <m> like </m> this",
    },
    {
        "mention_id": "m2",
        "doc_id": "doc2",
        "sentence_id": "s2",
        "topic": 1,
        "split": "test",
        "lemma": "love",
        "mention_text": "love",
        "sentence_tokens": ["I", "love", "this"],
        "men_type": "evt",
        "marked_doc": "I <m> love </m> this",
    },
    {
        "mention_id": "m3",
        "doc_id": "doc2",
        "sentence_id": "s2",
        "topic": 1,
        "split": "train",
        "lemma": "I",
        "mention_text": "I",
        "sentence_tokens": ["I", "love", "this"],
        "gold_cluster": "ent_1",
        "men_type": "ent",
        "marked_doc": "<m> I </m> love this",
    },
    {
        "mention_id": "m4",
        "doc_id": "doc1",
        "sentence_id": "s2",
        "topic": 1,
        "split": "train",
        "lemma": "this",
        "mention_text": "I",
        "sentence_tokens": ["I", "like", "this"],
        "gold_cluster": "ent_2",
        "men_type": "ent",
        "marked_doc": "I like <m> this </m>",
    },
]


class TestSummarizationCoreference:
    def test_summarizer_coref_resolver(self):
        docs = [  # These should be changed to full docs instead of sentences
            {  # Note to self: reconstruct marked documents from marked sentences
                "topic": "1",
                "doc_id": "doc1",
                "sentences": [
                    {
                        "sentence_id": "s1",
                        "sentence": "I like this sentence.",
                    },
                    {
                        "sentence_id": "s2",
                        "sentence": "I love this sentence.",
                    },
                ],
                "doc": "I like this sentence. I love this sentence.",
            },
            {
                "topic": "1",
                "doc_id": "doc2",
                "sentences": [
                    {
                        "sentence_id": "s1",
                        "sentence": "I like this sentence.",
                    },
                    {
                        "sentence_id": "s2",
                        "sentence": "I love this sentence.",
                    },
                ],
                "doc": "I like this sentence. I love this sentence.",
            },
        ]

        model_name = "/media/rehan/big_disk/models/kairos/ecb/multi/"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        generation_config = GenerationConfig.from_pretrained(model_name)

        ecb_name = "ahmeshaf/ecb_tagger_seq2seq"
        ecb_model = T5ForConditionalGeneration.from_pretrained(ecb_name)
        ecb_tokenizer = T5Tokenizer.from_pretrained(ecb_name)
        ecb_generation_config = GenerationConfig.from_pretrained(ecb_name)

        triggers_pipeline = EventsPipeline(
            model=ecb_model,
            tokenizer=ecb_tokenizer,
            task_prefix=TRIGGERS_PREFIX,
            framework="pt",
            generation_config=ecb_generation_config,
        )

        coref_resolver = SummarizerDocumentCorefResolver(
            triggers_pipeline,
            summary_generation_config=json.load(open(CONFIG_ROOT / "summarization.json")),
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            filterer=lemma_heuristic.LHFilterer(Dataset.from_list(TRAIN_DATASET)),
        )
        mention_clusters = coref_resolver(docs)
        mentions, clusters = zip(*mention_clusters)

        assert mention_clusters is not None
        assert "1_doc1_s1_2_6" in mentions
        assert "1_doc2_s1_2_6" in mentions


# Run the test
if __name__ == "__main__":
    pytest.main(["-s", "test_synergistic_module.py"])
