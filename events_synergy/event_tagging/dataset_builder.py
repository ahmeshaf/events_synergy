from datasets import load_dataset, Dataset, DatasetDict
from ..task_constants import TRIGGERS_PREFIX, ENTITIES_PREFIX


def make_tagger_dataset(mention_dataset_name):
    dataset_dict = {}
    dataset = load_dataset(mention_dataset_name)
    splits = list(dataset)
    for split in splits:
        dataset[split] = dataset[split].to_pandas()
        split_dict_array = dataset[split].to_dict("records")
        sent_id2evtrecords = {}
        sent_id2entrecords = {}
        sent_id2sentence = {}
        for record in split_dict_array:
            sent_id = (record["doc_id"], record["sentence_id"])
            sent_id2sentence[sent_id] = record["sentence"]
            if sent_id not in sent_id2evtrecords:
                sent_id2evtrecords[sent_id] = []
            if sent_id not in sent_id2entrecords:
                sent_id2entrecords[sent_id] = []

            if record["men_type"] == "evt":
                sent_id2evtrecords[sent_id].append(record)
            else:
                sent_id2entrecords[sent_id].append(record)

        sent_id2evtrecords = {
            sent_id: sorted(evt_records, key=lambda x: x["start_char"])
            for sent_id, evt_records in sent_id2evtrecords.items()
        }
        sent_id2entrecords = {
            sent_id: sorted(ent_records, key=lambda x: x["start_char"])
            for sent_id, ent_records in sent_id2entrecords.items()
        }

        trigger_records = [
            {
                "prompt": f"{TRIGGERS_PREFIX}: " + sent_id2sentence[sent_id],
                "response": " | ".join([record["mention_text"] for record in records]),
            }
            for sent_id, records in sent_id2evtrecords.items()
        ]

        entity_records = [
            {
                "prompt": f"{ENTITIES_PREFIX}: " + sent_id2sentence[sent_id],
                "response": " | ".join([record["mention_text"] for record in records]),
            }
            for sent_id, records in sent_id2entrecords.items()
        ]

        dataset_dict[split] = trigger_records + entity_records

    return DatasetDict(
        {split: Dataset.from_list(records) for split, records in dataset_dict.items()}
    )
