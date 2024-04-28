# Test suite for events_synergy/srl/semantic_role_labeler.py

import os

from events_synergy.srl.semantic_role_labeler import semantic_role_labeler


def test_semantic_role_labeler():

    sentences = [
        "The quick brown fox jumps over the lazy dog and then hunt the rabbit .",
        "The quick brown fox jumps over the lazy dog.",
    ]

    srl_ = semantic_role_labeler(sentences)

    print(srl_[0][1][0])

    assert len(srl_) == 2     # two sentences
    assert len(srl_[0]) == 2  # "jumps" and "hunts"
    assert len(srl_[1]) == 1  # "jumps"
    assert srl_[0][0][0] == ("jumps", (20, 25))
    assert srl_[0][1][0] == ("hunt", (53, 57))
