# Test suite for events_synergy/srl/semantic_role_labeler.py

import os

from events_synergy.srl.srler import (
    semantic_role_labeler,
    evaluate_gold_triggers,
    evaluate_end_to_end,
)


def test_semantic_role_labeler():

    sentences = [
        "The quick brown fox jumps over the lazy dog and then hunt the rabbit .",
        "The quick brown fox jumps over the lazy dog.",
    ]

    srl_ = semantic_role_labeler(sentences)

    assert len(srl_) == 2  # two sentences
    assert len(srl_[0]) == 2  # "jumps" and "hunts"
    assert len(srl_[1]) == 1  # "jumps"
    assert srl_[0][0][0][-1] == ("jumps", 20, 25)
    assert srl_[0][1][0][-1] == ("hunt", 53, 57)

    # check the arguments
    assert len(srl_[0][0][1]) == 2
    assert len(srl_[0][1][1]) == 2
    assert srl_[0][0][1][0] == (
        "ARG-0",
        "The quick brown fox",
        ("The quick brown fox", 0, 19),
    )


def test_srl_peft():
    srl_trigger_model_name = "cu-kairos/flan-srl-large-peft"
    sentences = [
        "The quick brown fox jumps over the lazy dog and then hunts the rabbit .",
        "The quick brown fox jumps over the lazy dog.",
    ]
    srl_ = semantic_role_labeler(
        sentences,
        trigger_model_name=srl_trigger_model_name,
        is_trigger_peft=True,
        srl_model=srl_trigger_model_name,
        is_srl_peft=True,
    )

    assert len(srl_) == 2  # two sentences
    assert len(srl_[0]) == 2  # "jumps" and "hunts"
    assert len(srl_[1]) == 1  # "jumps"
    assert srl_[0][0][0][-1] == ("jumps", 20, 25)
    assert srl_[0][1][0][-1] == ("hunts", 53, 58)

    # check the arguments
    # assert len(srl_[0][0][1]) == 2
    # assert len(srl_[0][1][1]) == 2
    assert srl_[0][0][1][0] == (
        "ARG-0",
        "The quick brown fox",
        ("The quick brown fox", 0, 19),
    )


def test_evaluate():
    dataset_name = "cu-kairos/conll05_mentions_sample"
    split = "validation"
    srl_model = "cu-kairos/propbank_srl_seq2seq_t5_small"
    is_srl_peft = False
    batch_size = 32
    evaluate_gold_triggers(dataset_name, split, srl_model, is_srl_peft, batch_size)

    evaluate_gold_triggers(
        dataset_name, split, "cu-kairos/flan-srl-large-peft", True, batch_size
    )


def test_evaluate_end_to_end():
    dataset_name = "cu-kairos/conll05_mentions_sample"
    split = "validation"

    evaluate_end_to_end(dataset_name, split)

    evaluate_end_to_end(
        dataset_name,
        split,
        "cu-kairos/flan-srl-large-peft",
        "cu-kairos/flan-srl-large-peft",
        batch_size=2,
        is_trigger_peft=True,
        is_srl_peft=True,
    )
