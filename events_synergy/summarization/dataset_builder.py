from datasets import Dataset, DatasetDict, load_dataset
import pathlib

DATA_ROOT = (pathlib.Path(__file__).parent.parent / "data").resolve()
CNN_URL = "https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"


def get_hf_dataset(dataset: Dataset, doc_col_name, summary_col_name):
    records = []
    for doc in dataset.to_pandas().to_dict("records"):
        prompt = f"Summarize the following article:\n\n{doc[doc_col_name]}"
        summary = doc[summary_col_name]

        records.append({"prompt": prompt, "response": summary})

    return Dataset.from_list(records)


def generate_summ_dataset(
    summ_dataset_dict: DatasetDict, doc_col_name, summary_col_name
):
    dataset_dict = {}
    for split in summ_dataset_dict:
        dataset_dict[split] = get_hf_dataset(
            summ_dataset_dict[split], doc_col_name, summary_col_name
        )

    return DatasetDict(dataset_dict)


def get_xsum():
    XSUM_DS = "EdinburghNLP/xsum"
    dataset = load_dataset(XSUM_DS)
    return generate_summ_dataset(dataset, "document", "summary")
