# Description: This file contains the implementation of the lemma heuristic filterer.
from typing import Callable, List, Set, Tuple

from ..utils import *


def get_synonymous_lemma_pairs(train_mention_dataset):
    cluster2lemmas = {}
    for record in train_mention_dataset:
        cluster_id = record["gold_cluster"]
        lemma = record["lemma"]
        if cluster_id not in cluster2lemmas:
            cluster2lemmas[cluster_id] = set()
        cluster2lemmas[cluster_id].add(lemma)
    lemma_pairs = set()
    for lemmas in cluster2lemmas.values():
        for lemma1 in lemmas:
            for lemma2 in lemmas:
                if lemma1 != lemma2:
                    lemma_pairs.add(tuple(sorted((lemma1, lemma2))))

    return lemma_pairs


def lh(mention_map: Dict, syn_lemma_pairs: Set[Tuple[str, str]], threshold=0.05):
    """
    Returns
    -------
    List[Tuple[str, str]]
    """
    mention_pairs = generate_topic_mention_pairs(mention_map)

    filtered_pairs = set()

    for m1, m2 in mention_pairs:
        m1_lemma = remove_puncts(mention_map[m1]["lemma"].lower())
        m2_lemma = remove_puncts(mention_map[m2]["lemma"].lower())

        m1_sentence_tokens = mention_map[m1]["sentence_tokens"]
        m2_sentence_tokens = mention_map[m2]["sentence_tokens"]

        m1_mention_text = remove_puncts(mention_map[m1]["mention_text"].lower())
        m2_mention_text = remove_puncts(mention_map[m2]["mention_text"].lower())

        # check if the lemmas are synonymous
        is_syn_lemma = False
        if (m1_lemma, m2_lemma) in syn_lemma_pairs or (
            m2_lemma,
            m1_lemma,
        ) in syn_lemma_pairs:
            is_syn_lemma = True

        # check if the lemmas are contained by the mention_texts
        is_contained_mention = False
        if m1_lemma in m2_mention_text or m2_lemma in m1_mention_text:
            is_contained_mention = True

        # calculate the sentence tokens similarity
        sentence_tokens_sim = jaccard_similarity(
            set(m1_sentence_tokens), set(m2_sentence_tokens)
        )

        if (is_syn_lemma or is_contained_mention) and sentence_tokens_sim > threshold:
            filtered_pairs.add(tuple(sorted((m1, m2))))

    return list(filtered_pairs)


class LHFilterer(Callable):
    def __init__(
        self,
        train_mention_dataset: Dataset = None,
        syn_lemma_pairs: List[Tuple[str, str]] = None,
        threshold: float = 0.05,
    ):
        if syn_lemma_pairs is not None:
            self.synonymous_lemma_pairs = syn_lemma_pairs
        elif train_mention_dataset is not None:
            self.synonymous_lemma_pairs = get_synonymous_lemma_pairs(
                train_mention_dataset
            )
        else:
            self.synonymous_lemma_pairs = set()
        self.threshold = threshold

    def __call__(self, mention_map):
        return lh(
            mention_map, self.synonymous_lemma_pairs, threshold=self.threshold
        )

