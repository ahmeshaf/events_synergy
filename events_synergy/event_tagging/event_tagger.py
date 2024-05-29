from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import List

from .dataset_builder import get_tagger_dataset
from ..events_pipeline import EventsPipeline, pipe
from ..task_constants import TRIGGERS_PREFIX
from ..utils.helpers import find_word_offsets, find_phrase_offsets_fuzzy, get_prf


def event_tagger(
    sentences, model=None, tokenizer=None, generation_config=None, batch_size=8
):
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


def tag_with_prompts(tagger_model_name, prompts, batch_size=32):
    tagger_dataset = Dataset.from_dict({"prompt": prompts})
    tagger_pipe = pipeline("text2text-generation", tagger_model_name, device_map="auto")
    tagger_out = []
    for out in tqdm(
            tagger_pipe(KeyDataset(tagger_dataset, "prompt"), batch_size=batch_size),
            total=len(prompts),
            desc="Tagging",
    ):
        tagger_out.append(out[0]["generated_text"])
    return tagger_out


def tagger(
    tagger_model_name: str,
    sentences: List[str],
    men_type: str = "evt",
    batch_size: int = 32,
):
    if men_type == "ent":
        prompts = [f"entities: {sentence}" for sentence in sentences]
    else:
        prompts = [f"triggers: {sentence}" for sentence in sentences]

    tagger_out = tag_with_prompts(tagger_model_name, prompts, batch_size)

    predicted_mentions = []

    for i, (sentence, tags_str) in enumerate(zip(sentences, tagger_out)):
        split_tags = [t.strip() for t in tags_str.split("|") if t.strip() != ""]
        offsets = find_phrase_offsets_fuzzy(sentence, split_tags)
        predicted_mentions.append(list(zip(split_tags, offsets)))

    return predicted_mentions


def evaluate(
    tagger_model_name: str,
    mention_dataset_name: str,
    split: str = "test",
    men_type: str = "evt",
    batch_size: int = 32,
) -> dict:
    dataset_dict = load_dataset(mention_dataset_name, token="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")
    # each record is a prompt sentence with triggers or entities ("run | jump")
    split_dataset = dataset_dict[split]

    # each prompt is str and each gold_mention is a tuple of (mention_text, start_char, end_char)
    tagger_dataset = get_tagger_dataset(split_dataset, men_type)

    prompts = tagger_dataset["prompt"]
    sentences = tagger_dataset["sentence"]
    gold_mentions = tagger_dataset["gold_mentions"]

    gold_mentions_flattened = [
        f"({i})" + mention
        for i, mentions in enumerate(gold_mentions)
        for mention in mentions
    ]

    tagger_out = tag_with_prompts(tagger_model_name, prompts, batch_size)

    predicted_mentions_flattened = []

    for i, (sentence, tags_str) in enumerate(zip(sentences, tagger_out)):
        split_tags = [t.strip() for t in tags_str.split("|") if t.strip() != ""]
        offsets = find_phrase_offsets_fuzzy(sentence, split_tags)
        predicted_mentions_flattened.extend(
            [f"({i})" + str(offset) for offset in offsets]
        )

    prf = get_prf(gold_mentions_flattened, predicted_mentions_flattened)

    return prf


if __name__ == "__main__":
    evaluate("ahmeshaf/ecb_tagger_seq2seq", "ahmeshaf/conll_05_mentions", "validation")
