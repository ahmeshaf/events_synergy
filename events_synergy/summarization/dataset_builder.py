from datasets import load_dataset
from torch.utils.data import Dataset
import pathlib

DATA_ROOT = (pathlib.Path(__file__).parent.parent / "data").resolve()
CNN_URL = "https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"





def get_hf_dataset(dataset_name: str,
                   summary_colname: str = "summary",
                   doc_colname: str = "document"):
    dataset = load_dataset(dataset_name, summary_colname, doc_colname)

    return {'train': SummarizationDataset(dataset['train']),
            'dev': SummarizationDataset(dataset['validation']),
            'test': SummarizationDataset(dataset['test'])}

def generate_dataset(data, summary_colname, doc_colname):
    prompts = []
    responses = []
    for doc in data:
        prompt = f"Summarize the following article:\n\n{doc[doc_colname]}"
        summary = doc[summary_colname]

        prompts.append(prompt)
        responses.append(summary)

    return {"prompt": prompts, "response": responses}


class SummarizationDataset:
    def __init__(self, data, summary_colname, doc_colname):
        self.data = generate_dataset(data, summary_colname, doc_colname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
