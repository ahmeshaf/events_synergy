from datasets import Dataset, DatasetDict, load_dataset
from typing import Callable

from jinja2 import Template

from ..task_constants import COREF_TEMPLATE
from .utils import get_mention_map


def generate_coref_dataset(
    mention_dataset_dict: DatasetDict,
    filterer: Callable,
    text_key="marked_sentence",
    men_type: str = "evt",
):
    """

    :param mention_dataset_dict:
    :param men_type: can be "evt" or "ent" or "all"
    :param filterer:
    :param text_key:
    :return: DatasetDict
    """
    template = Template(COREF_TEMPLATE)
    splits = list(mention_dataset_dict)
    split2dataset = {}
    for split in splits:
        mention_map = get_mention_map(mention_dataset_dict[split], men_type)
        mention_pairs_dataset = filterer(mention_map)
        prompt_responses = []
        for m1, m2 in mention_pairs_dataset:
            mention_1 = mention_map[m1]
            mention_2 = mention_map[m2]

            prompt = template.render(
                mention_text_1=mention_1["mention_text"],
                mention1_context=mention_1[text_key],
                mention_text_2=mention_2["mention_text"],
                mention2_context=mention_2[text_key],
            )

            response = (
                "Yes"
                if mention_1["gold_cluster"] == mention_2["gold_cluster"]
                else "No"
            )

            prompt_responses.append({"prompt": prompt, "response": response})

        split2dataset[split] = Dataset.from_list(prompt_responses)

    return DatasetDict(split2dataset)
