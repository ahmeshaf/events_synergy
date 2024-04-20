from datasets import load_dataset
from torch.utils.data import Dataset
import pathlib

DATA_ROOT = (pathlib.Path(__file__).parent.parent / "data").resolve()
CNN_URL = "https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"



def get_xsum():
    xsum_dataset = load_dataset("EdinburghNLP/xsum")

    return {'train' : SummarizationDataset(xsum_dataset['train']),
            'dev' : SummarizationDataset(xsum_dataset['validation']),
            'test' : SummarizationDataset(xsum_dataset['test'])}


def generate_dataset(data):
    prompts = []
    responses = []
    for doc in data:
        prompt = f"Summarize the following article:\n\n{doc['document']}"
        summary = doc["summary"]

        prompts.append(prompt)
        responses.append(summary)

    return {"prompt": prompts, "response": responses}


class SummarizationDataset:
    def __init__(self, data):
        self.data = generate_dataset(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
