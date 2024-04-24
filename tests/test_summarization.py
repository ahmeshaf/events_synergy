# Test cases for the module summarization at events_synergy.summarization
import pytest
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig

from events_synergy.summarization.summarizer import summarize


def test_summarizer_summarize():
    # test cases for the summarize function
    documents = [
        "I like this sentence and hate this sentence and I like this thing",
        "The earthquake took 10 lives .",
    ]

    model_name = "/media/rehan/big_disk/models/kairos/ecb/multi/"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)

    summary = summarize(documents, model=model, tokenizer=tokenizer, generation_config=generation_config)
    assert summary is not None
    assert len(summary) == 2
    assert isinstance(summary, list)
    assert isinstance(summary[0], str)
    assert isinstance(summary[1], str)
    assert len(summary[0]) > 0
    assert len(summary[1]) > 0
