from datasets import Dataset, load_dataset

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from typer import Typer
from typing import List

from .dataset_builder import get_srl_dataset

from ..event_tagging.event_tagger import tagger
from ..event_tagging.dataset_builder import get_tagger_dataset
from ..events_pipeline import get_model_tokenizer_generation_config
from ..utils.helpers import find_phrase_offsets_fuzzy, get_prf


app = Typer()


def generate_srl_prompts(sentences, sentences_triggers):
    for i, (sentence, triggers) in enumerate(zip(sentences, sentences_triggers)):
        for trigger in triggers:
            (_, (trigger_txt, start, end)) = trigger
            sentence_prompt = sentence[:start] + f"[{trigger_txt}]" + sentence[end:]
            srl_prompt = f"SRL for [{trigger[0]}]: {sentence_prompt}"
            yield i, trigger, srl_prompt, sentence


def run_srl_pipe(srl_model, is_srl_peft, srl_prompts, batch_size):
    model, tokenizer, generation_config = get_model_tokenizer_generation_config(
        srl_model, is_srl_peft
    )
    srl_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    outputs = []
    for out in tqdm(
        srl_pipe(KeyDataset(srl_prompts, "prompt"), batch_size=batch_size),
        total=len(srl_prompts),
        desc="SRL",
    ):
        outputs.append(out[0]["generated_text"])
    return outputs


def srl_predicted_triggers(
    sentences, triggers, srl_model, batch_size, is_srl_peft=False
):
    srl_prompts = list(generate_srl_prompts(sentences, triggers))

    s_ids, flattened_triggers, srl_prompts, flattened_sentences = zip(*srl_prompts)

    srl_dataset = Dataset.from_dict({"prompt": srl_prompts})

    outputs = run_srl_pipe(srl_model, is_srl_peft, srl_dataset, batch_size)

    s_id2srl = {}

    for i, trigger, output, sentence in zip(
        s_ids, flattened_triggers, outputs, flattened_sentences
    ):
        if i not in s_id2srl:
            s_id2srl[i] = []

        s_id2srl[i].append((trigger, sentence, output))

    s_id2srl = sorted(s_id2srl.items(), key=lambda x: x[0])
    sentence_srls = [srl for _, srl in s_id2srl]

    triggers_srls_offsets = []

    for sentence_srl in sentence_srls:
        curr_triggers_srls_phrases = []
        for trigger, sentence, str_srl in sentence_srl:
            if str_srl != "":
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
    is_trigger_peft: bool = False,
    srl_model: str = "cu-kairos/propbank_srl_seq2seq_t5_large",
    is_srl_peft: bool = False,
    batch_size: int = 32,
):
    triggers = tagger(
        trigger_model_name,
        sentences,
        batch_size=batch_size,
        men_type="evt",
        is_peft=is_trigger_peft,
    )
    return srl_predicted_triggers(
        sentences, triggers, srl_model, batch_size, is_srl_peft=is_srl_peft
    )


def get_arg_srls(sentence, srl_response):
    arg_srls = [tuple(arg.split(": ")) for arg in srl_response.split(" | ")]
    arg_labels, arg_phrases = zip(*arg_srls)
    phrase_offsets = find_phrase_offsets_fuzzy(sentence, arg_phrases)
    return list(zip(arg_labels, arg_phrases, phrase_offsets))


@app.command()
def evaluate_end_to_end(
    mention_dataset_name: str,
    split: str = "validation",
    trigger_model_name: str = "ahmeshaf/ecb_tagger_seq2seq",
    srl_model: str = "cu-kairos/propbank_srl_seq2seq_t5_small",
    batch_size: int = 32,
    is_trigger_peft: bool = False,
    is_srl_peft: bool = False,
):
    mention_dataset = load_dataset(mention_dataset_name)
    split_dataset = mention_dataset[split]
    dataset_records = split_dataset.to_pandas().to_dict(orient="records")

    sentence_id2sentence = {
        (record["doc_id"], record["sentence_id"]): record["sentence"] for record in dataset_records
    }
    sentence_ids, sentences = zip(*sentence_id2sentence.items())

    srls = semantic_role_labeler(
        sentences,
        trigger_model_name,
        is_trigger_peft,
        srl_model,
        is_srl_peft,
        batch_size,
    )
    gold_triggers = set()
    gold_srls = []
    for record in dataset_records:
        curr_rec = (
            record["doc_id"],
            record["sentence_id"],
            record["mention_text"],
            record["start_char"],
            record["end_char"],
        )
        gold_triggers.add(curr_rec)
        arg_srls = get_arg_srls(record["sentence"], record["srl_response"])
        curr_recs = [curr_rec + s[:1] + s[-1] for s in arg_srls]
        gold_srls.extend(curr_recs)

    predicted_triggers = set()
    predicted_srls = []
    for sentence_id, curr_srl in zip(sentence_ids, srls):
        for trigger, arg_srls in curr_srl:
            for arg_srl in arg_srls:
                predicted_triggers.add(sentence_id + trigger[-1])
                predicted_srls.append(sentence_id + trigger[-1] + arg_srl[:1] + arg_srl[-1])

    triggers_prf = get_prf(gold_triggers, predicted_triggers)
    overall_prf = get_prf(gold_srls, predicted_srls)
    prf = {"triggers": triggers_prf, "overall": overall_prf}
    print(prf)
    return prf


@app.command()
def evaluate_gold_triggers(
    mention_dataset_name: str,
    split: str = "validation",
    srl_model: str = "cu-kairos/propbank_srl_seq2seq_t5_large",
    is_srl_peft: bool = False,
    batch_size: int = 32,
):
    mention_dataset = load_dataset(mention_dataset_name)
    split_dataset = mention_dataset[split]
    srl_dataset = get_srl_dataset(split_dataset)
    outputs = run_srl_pipe(srl_model, is_srl_peft, srl_dataset, batch_size)

    dataset_records = srl_dataset.to_pandas().to_dict(orient="records")

    gold_srls = []
    predicted_srls = []

    for record, output in zip(dataset_records, outputs):
        # record is a dictionary with keys: prompt, response, sentence, gold_mentions
        # output is the generated text of the form ARG-0: The quick brown fox | ARG-1: over the lazy dog
        # get the offsets by splitting the output string and finding the phrase offsets
        gold_arg_srls = get_arg_srls(record["sentence"], record["response"])
        gold_srls.extend(
            [(record["sentence_id"],) + s[:1] + s[-1] for s in gold_arg_srls]
        )

        predicted_arg_srls = get_arg_srls(record["sentence"], output)
        predicted_srls.extend(
            [(record["sentence_id"],) + s[:1] + s[-1] for s in predicted_arg_srls]
        )

    prf = get_prf(gold_srls, predicted_srls)
    print(prf)
    return prf


if __name__ == "__main__":
    app()
