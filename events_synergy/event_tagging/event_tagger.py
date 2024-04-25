from ..events_pipeline import EventsPipeline, pipe
from ..task_constants import TRIGGERS_PREFIX
from ..utils.helpers import find_word_offsets


def event_tagger(sentences, model=None, tokenizer=None, generation_config=None, batch_size=8):
    # Initialize our custom pipeline
    triggers_pipeline = EventsPipeline(
        model=model,
        tokenizer=tokenizer,
        task_prefix=TRIGGERS_PREFIX,
        framework="pt",
        generation_config=generation_config,
    )
    outputs = pipe(triggers_pipeline, sentences, batch_size, desc="Tagging")
    trigger_offsets = []

    for sentence, triggers in zip(sentences, outputs):
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
