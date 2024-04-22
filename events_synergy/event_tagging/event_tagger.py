from transformers import (
    GenerationConfig,
    T5ForConditionalGeneration,
    T5Tokenizer
)

from ..events_pipeline import EventsPipeline
from ..task_constants import TRIGGERS_PREFIX
from ..utils.helpers import find_word_offsets


def event_tagger(sentences, model_name="ahmeshaf/ecb_tagger_seq2seq"):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    # Initialize our custom pipeline
    triggers_pipeline = EventsPipeline(
        model=model,
        tokenizer=tokenizer,
        task_prefix=TRIGGERS_PREFIX,
        framework="pt",
        generation_config=generation_config,
    )
    sentence_triggers = triggers_pipeline(sentences)
    trigger_offsets = []
    for sentence, triggers in zip(sentences, sentence_triggers):
        offsets = find_word_offsets(sentence, triggers)
        trigger_offsets.append(list(zip(triggers, offsets)))
    return trigger_offsets


if __name__ == "__main__":
    print(
        event_tagger(
            [
                "I like this sentence and hate this sentence and I like this thing",
                "The earthquake took 10 lives ."
            ]
        )
    )
