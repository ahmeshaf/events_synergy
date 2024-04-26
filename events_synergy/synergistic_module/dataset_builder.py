from datasets import Dataset, DatasetDict, concatenate_datasets
from typing import List


def generate_multitask_dataset(datasets: List[DatasetDict], dataset_names: List[str]):
    """
    :return: DatasetDict of the form ds['train], ds['validation_{task}'], ds['test_{task}']
    """

    final_ds = DatasetDict()

    # Shuffle the training data from all tasks
    final_ds['train'] = concatenate_datasets([dataset['train'] for dataset in datasets]).shuffle()

    for dataset, name in zip(datasets, dataset_names):
        final_ds[f'validation_{name}'] = dataset['validation']
        final_ds[f'test_{name}'] = dataset['test']

    return final_ds
