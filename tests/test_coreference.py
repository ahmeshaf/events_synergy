# test cases for the coreference module at events_synergy.coreference
import pytest
from transformers import GenerationConfig, T5ForConditionalGeneration, T5Tokenizer

from events_synergy.coreference.filtering import lemma_heuristic
from events_synergy.coreference.coref_resolver import (
    MentionsCorefResolver,
    DocumentCorefResolver,
)
from datasets import Dataset

from events_synergy.coreference.utils import get_mention_map
from events_synergy.events_pipeline import EventsPipeline
from events_synergy.task_constants import TRIGGERS_PREFIX

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
        "marked_sentence": "I <m> like </m> this",
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
        "marked_sentence": "I <m> love </m> this",
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
        "marked_sentence": "<m> I </m> love this",
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
        "marked_sentence": "I like <m> this </m>",
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
        "marked_sentence": "I <m> like </m> this",
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
        "marked_sentence": "I <m> love </m> this",
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
        "marked_sentence": "<m> I </m> love this",
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
        "marked_sentence": "I like <m> this </m>",
    },
]


class TestCoreference:
    def test_get_synonymous_lemma_pairs(self):
        # Define the input
        train_mention_dataset = [
            {"gold_cluster": 1, "lemma": "like"},
            {"gold_cluster": 1, "lemma": "love"},
            {"gold_cluster": 2, "lemma": "hate"},
            {"gold_cluster": 2, "lemma": "dislike"},
            {"gold_cluster": 3, "lemma": "eat"},
            {"gold_cluster": 3, "lemma": "consume"},
        ]
        # Call the function to test
        result = lemma_heuristic.get_synonymous_lemma_pairs(train_mention_dataset)
        # Define the expected output
        expected_output = {("like", "love"), ("dislike", "hate"), ("consume", "eat")}
        # Compare the result with the expected output
        assert result == expected_output

    def test_lh_filterer(self):
        # Test the LHFilterer class
        # Define the input
        train_dataset = Dataset.from_list(TRAIN_DATASET)

        test_dataset = Dataset.from_list(TEST_DATASET)

        test_map_ent = get_mention_map(test_dataset, "ent")
        test_map_evt = get_mention_map(test_dataset, "evt")
        test_map_all = get_mention_map(test_dataset, "all")

        lh_filterer = lemma_heuristic.LHFilterer(train_dataset)
        # Call the function to test
        result_evt = lh_filterer(test_map_evt)
        result_ent = lh_filterer(test_map_ent)
        result_all = lh_filterer(test_map_all)
        # Define the expected output
        expected_output_evt = {("m1", "m2")}
        expected_output_ent = {("m3", "m4")}
        expected_output_all = {("m1", "m2"), ("m3", "m4")}
        # Compare the result with the expected output
        assert set(result_evt) == expected_output_evt
        assert set(result_ent) == expected_output_ent
        assert set(result_all) == expected_output_all

    def test_mention_coref_resolver(self):
        # Test the CorefResolver class
        # Define the input
        train_dataset = Dataset.from_list(TRAIN_DATASET)

        test_dataset = Dataset.from_list(TEST_DATASET)

        test_map_evt = get_mention_map(test_dataset, "evt")
        test_map_ent = get_mention_map(test_dataset, "ent")
        test_map_all = get_mention_map(test_dataset, "all")

        lh_filterer = lemma_heuristic.LHFilterer(train_dataset)

        model_name = "/media/rehan/big_disk/models/kairos/ecb/multi/"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        generation_config = GenerationConfig.from_pretrained(model_name)

        coref_resolver = MentionsCorefResolver(
            model, tokenizer, generation_config, lh_filterer
        )
        # Call the function to test
        result_evt = coref_resolver(test_map_evt)
        result_ent = coref_resolver(test_map_ent)
        result_all = coref_resolver(test_map_all)

        # Define the expected output
        expected_output = [("m1", 0), ("m2", 0)]
        expected_output_ent = [("m3", 0), ("m4", 0)]
        expected_output_all = [("m1", 0), ("m2", 0), ("m3", 1), ("m4", 1)]
        # Compare the result with the expected output
        assert result_evt == expected_output
        assert result_ent == expected_output_ent
        assert result_all == expected_output_all

    def test_docs_coref_resolver(self):
        docs = [
            {
                "topic": "1",
                "doc_id": "doc1",
                "sentences": [
                    {
                        "sentence_id": "s1",
                        "sentence": "I like this sentence",
                    },
                    {
                        "sentence_id": "s2",
                        "sentence": "I love this sentence",
                    },
                ],
            },
            {
                "topic": "1",
                "doc_id": "doc2",
                "sentences": [
                    {
                        "sentence_id": "s1",
                        "sentence": "I like this sentence",
                    },
                    {
                        "sentence_id": "s2",
                        "sentence": "I love this sentence",
                    },
                ],
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

        coref_resolver = DocumentCorefResolver(
            triggers_pipeline,
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
        assert len(set(clusters)) == 2




# Run the test
if __name__ == "__main__":
    pytest.main(["-s", "test_coreference.py"])
