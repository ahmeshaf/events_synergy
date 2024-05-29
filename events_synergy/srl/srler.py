from typing import List

from datasets import Dataset

from events_synergy.utils.helpers import find_phrase_offsets_fuzzy
from transformers import pipeline

from ..event_tagging.event_tagger import tagger

from ..events_pipeline import get_model_tokenizer_generation_config


def generate_srl_prompts(sentences, sentences_triggers):
    for i, (sentence, triggers) in enumerate(zip(sentences, sentences_triggers)):
        for trigger in triggers:
            (_, (trigger_txt, start, end)) = trigger
            sentence_prompt = sentence[:start] + f"[{trigger_txt}]" + sentence[end:]
            srl_prompt = f"SRL for [{trigger[0]}]: {sentence_prompt}"
            yield i, trigger, srl_prompt, sentence


def srl_predicted_triggers(sentences, triggers, srl_model, batch_size):
    srl_prompts = list(generate_srl_prompts(sentences, triggers))

    s_ids, flattened_triggers, srl_prompts, flattened_sentences = zip(*srl_prompts)

    model_s, tokenizer_s, generation_config_s = get_model_tokenizer_generation_config(
        srl_model
    )

    srl_pipe = pipeline(
        "text2text-generation",
        model=srl_model,
        tokenizer=tokenizer_s,
        generation_config=generation_config_s,
    )
    outputs = srl_pipe(list(srl_prompts), batch_size=batch_size)

    s_id2srl = {}

    for i, trigger, output, sentence in zip(
        s_ids, flattened_triggers, outputs, flattened_sentences
    ):
        if i not in s_id2srl:
            s_id2srl[i] = []

        s_id2srl[i].append((trigger, sentence, output["generated_text"]))

    s_id2srl = sorted(s_id2srl.items(), key=lambda x: x[0])
    sentence_srls = [srl for _, srl in s_id2srl]

    triggers_srls_offsets = []

    for sentence_srl in sentence_srls:
        curr_triggers_srls_phrases = []
        for trigger, sentence, str_srl in sentence_srl:
            arg_srls = [tuple(arg.split(": ")) for arg in str_srl.split(" | ")]
            arg_labels, arg_phrases = zip(*arg_srls)
            phrase_offsets = find_phrase_offsets_fuzzy(sentence, arg_phrases)
            arg_srls = list(zip(arg_labels, arg_phrases, phrase_offsets))
            curr_triggers_srls_phrases.append((trigger, arg_srls))
        triggers_srls_offsets.append(curr_triggers_srls_phrases)

    return triggers_srls_offsets


def semantic_role_labeler(
    sentences: List[str],
    trigger_model_name: str = "ahmeshaf/ecb_tagger_seq2seq",
    srl_model: str = "cu-kairos/propbank_srl_seq2seq_t5_large",
    batch_size: int = 32,
):
    triggers = tagger(
        trigger_model_name, sentences, batch_size=batch_size, men_type="evt"
    )
    return srl_predicted_triggers(sentences, triggers, srl_model, batch_size)
