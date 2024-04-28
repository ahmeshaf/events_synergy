from typing import List

from transformers import pipeline

from ..event_tagging.event_tagger import event_tagger

from ..events_pipeline import get_model_tokenizer_generation_config


def generate_srl_prompts(sentences, sentences_triggers):
    for i, (sentence, triggers) in enumerate(zip(sentences, sentences_triggers)):
        for trigger in triggers:
            sentence = (
                sentence[: trigger[1][0]]
                + f"[{trigger[0]}]"
                + sentence[trigger[1][1]:]
            )
            srl_prompt = f"SRL for [{trigger[0]}]: {sentence}"
            yield i, trigger, srl_prompt


def semantic_role_labeler(
    sentences: List[str],
    trigger_model: str = "ahmeshaf/ecb_tagger_seq2seq",
    srl_model: str = "cu-kairos/propbank_srl_seq2seq_t5_large",
    batch_size: int = 8,
):
    model_t, tokenizer_t, generation_config_t = get_model_tokenizer_generation_config(
        trigger_model
    )

    triggers = event_tagger(
        sentences, model_t, tokenizer_t, generation_config_t, batch_size
    )

    srl_prompts = list(generate_srl_prompts(sentences, triggers))

    s_ids, flattened_triggers, srl_prompts = zip(*srl_prompts)

    model_s, tokenizer_s, generation_config_s = get_model_tokenizer_generation_config(
        srl_model
    )

    # srl_pipeline = pipeline("text2text-generation", model=model_s, tokenizer=tokenizer_s, framework="pt", device="cuda")

    tokenized_inputs = tokenizer_s(list(srl_prompts), truncation=True, padding=True, return_tensors="pt")
    outputs = model_s.generate(**tokenized_inputs, generation_config=generation_config_s)

    s_id2srl = {}

    for i, trigger, output in zip(s_ids, flattened_triggers, outputs):
        if i not in s_id2srl:
            s_id2srl[i] = []

        s_id2srl[i].append((trigger, tokenizer_s.decode(output, skip_special_tokens=True)))

    s_id2srl = sorted(s_id2srl.items(), key=lambda x: x[0])
    sentence_srls = [srl for _, srl in s_id2srl]

    triggers_srls_offsets = []

    for sentence_srl in sentence_srls:
        curr_triggers_srls_offsets = []
        for trigger, str_srl in sentence_srl:
            arg_srls = [tuple(arg.split(": ")) for arg in str_srl.split(" | ")]
            curr_triggers_srls_offsets.append((trigger, arg_srls))
        triggers_srls_offsets.append(curr_triggers_srls_offsets)

    return triggers_srls_offsets
