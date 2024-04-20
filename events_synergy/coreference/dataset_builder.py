from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Callable

from .filtering.lemma_heuristic import save_lh_pairs


class CoreferenceDataset(Dataset):
    def __init__(self, mention_map, split, filterer: Callable, text_key="marked_sentence"):
        self.mention_map = mention_map
        self.mention_pairs_dataset = filterer(self.mention_map, split)
        self.text_key = text_key
        self.dataset = self.generate_dataset()

    def __len__(self):
        return len(self.mention_pairs_dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def generate_dataset(self):
        prompts = []
        responses = []
        for m1, m2 in self.mention_pairs_dataset:
            mention_1 = self.mention_map[m1]
            mention_2 = self.mention_map[m2]

            prompt = f"Coreference: \
            <m> {mention_1['mention_text']} </m> in {mention_1[self.text_key]} </s> \
            <m> {mention_2['mention_text']} </m> in {mention_2[self.text_key]}"
            response = (
                "Yes" if mention_1["gold_cluster"] == mention_2["gold_cluster"] else "No"
            )

            prompts.append(prompt)
            responses.append(response)

        return {"prompt": prompts, "response": responses}


if __name__ == "__main__":
    mention_dataset_ = "ahmeshaf/ecb_plus_mentions"
    coref_dataset = CoreferenceDataset(mention_dataset_, "dev", save_lh_pairs)

    my_loader = DataLoader(coref_dataset, batch_size=2, shuffle=True)
    for batch in my_loader:
        print(batch)
        break


