"""
Use the data from https://github.com/ahmeshaf/SECURE/tree/dev/data/ecb%2B/gpt4
"""

import json
import pickle

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from ..coreference.utils import get_mention_map


def modify_elb_content(content, trigger_pos):
    start_pos, end_pos = trigger_pos

    substring = content[start_pos:end_pos]
    clean_substring = content[start_pos+1:end_pos-1]

    # Replace the first # with <m> and the second # with </m>
    modified_substring = substring.replace('#', '<m> ', 1).replace('#', ' </m>', 1)

    # Replace the original substring in the content with the modified one
    modified_content = content[:start_pos] + modified_substring + content[end_pos:]

    return modified_content, clean_substring

def get_doc_id(id):
    return id.split("ecb")[0]

def save_to_json(data_list, file_name):
    """
    Saves a list of dictionaries to a JSON file.
    """
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

def elaboration_map(split_mention_map, secure_split_path):
    result_list = []
    error_list = []

    with open(secure_split_path, 'r') as file:
        elb_json = json.load(file)    
    SECUREID2mention_id = { "_".join(str(men[k]) for k in ["doc_id", "sentence_id", "token_start", "token_end"]).replace(".xml", "") : m_id for m_id, men in split_mention_map.items()}
    for s_id, s_data in elb_json.items():
        if s_id in SECUREID2mention_id:
            m_id = SECUREID2mention_id.get(s_id)
            men = split_mention_map.get(m_id)

            mention_text = men.get("mention_text")
            marked_doc = men.get("marked_doc")

            content = s_data['elaboration']['content']
            trigger_pos = s_data['elaboration']['trigger_pos']
            elaboration_text, elaboration_ent = modify_elb_content(content, trigger_pos)

            if elaboration_ent != mention_text:
                error_list.append({
                    "doc_id":men.get("doc_id"),
                    "sentence_id": men.get("sentence_id"),
                    "token_start": men.get("token_start"),
                    "token_end": men.get("token_end"),
                    "s_id": s_id, 
                    "mention_text":mention_text,
                    "elaboration_ent":elaboration_ent,
                })
            prompt = f"Elaborate <m> {mention_text} </m> in the following article: {marked_doc}"

            result_list.append({
                "s_id": s_id,
                "m_id": m_id,
                "prompt": prompt,
                "response": elaboration_text
            })
        
    print(len(result_list), len(error_list))
    return result_list, error_list


def create_and_upload_elaboration_datasets(local_save_path="events_synergy/data/gpt4/"):
    """
    Creates elaboration datasets from the ECB dataset and uploads them to the specified hub.
    """
    ecb_dataset_name = "ahmeshaf/ecb_plus_mentions"
    ecb_dataset = load_dataset(ecb_dataset_name)

    elaboration_datasets = {}
    dataset_splits = ecb_dataset.keys()

    for split in dataset_splits:
        split_name = split
        if "dev" in split:
            split_name = "dev"
        elif "train" in split:
            split_name = "training"

        mention_map = get_mention_map(ecb_dataset[split], "all")
        elaboration_file_path = f"{local_save_path}elaboration_{split_name}_data.json"
        elaboration_results, elaboration_errors = elaboration_map(mention_map, elaboration_file_path)

        results_df = pd.DataFrame(elaboration_results)
        elaboration_dataset = Dataset.from_pandas(results_df)
        elaboration_datasets[split] = elaboration_dataset

        save_to_json(elaboration_results, f"{local_save_path}elaboration_{split}_data_complete.json")
        save_to_json(elaboration_errors, f"{local_save_path}elaboration_{split}_data_complete_error.json")

    elaboration_dataset_dict = DatasetDict(elaboration_datasets)
    elaboration_dataset_dict.push_to_hub("events-synergy/ecb_plus_elaboration")


if __name__ == "__main__":
    create_and_upload_elaboration_datasets()
