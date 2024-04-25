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
    return event_tagger_pipeline(triggers_pipeline, sentences, batch_size)


def event_tagger_pipeline(triggers_pipeline, sentences, batch_size=8):
    outputs = pipe(triggers_pipeline, sentences, batch_size, desc="Tagging")
    trigger_offsets = []

    for sentence, triggers in zip(sentences, outputs):
        split_triggers = [t.strip() for t in triggers.split("|") if t.strip() != ""]
        offsets = find_word_offsets(sentence, split_triggers)
        trigger_offsets.append(list(zip(split_triggers, offsets)))
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
