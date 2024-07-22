from datasets import load_dataset, Dataset, DatasetDict
from ..task_constants import TRIGGERS_PREFIX, ENTITIES_PREFIX


def get_tagger_dataset(dataset: Dataset, men_type="evt") -> Dataset:
    split_dict_array = dataset.to_pandas().to_dict("records")
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
            "response": str(
                " | ".join(
                    [
                        record["mention_text"] if record["mention_text"] else "<unk>"
                        for record in records
                    ]
                )
            ),
            "sentence": sent_id2sentence[sent_id],
            "gold_mentions": [
                str((record["mention_text"], record["start_char"], record["end_char"]))
                for record in records
            ],
        }
        for sent_id, records in sent_id2evtrecords.items()
    ]

    entity_records = [
        {
            "prompt": f"{ENTITIES_PREFIX}: " + sent_id2sentence[sent_id],
            "response": str(" | ".join([record["mention_text"] for record in records])),
            "sentence": sent_id2sentence[sent_id],
            "gold_mentions": [
                str((record["mention_text"], record["start_char"], record["end_char"]))
                for record in records
            ],
        }
        for sent_id, records in sent_id2entrecords.items()
    ]

    if men_type == "evt":
        return Dataset.from_list(trigger_records)
    elif men_type == "ent":
        return Dataset.from_list(entity_records)
    else:
        return Dataset.from_list(trigger_records + entity_records)


def make_tagger_dataset_dict(mention_dataset_name):
    dataset_dict = load_dataset(mention_dataset_name)
    splits = list(dataset_dict)

    return DatasetDict(
        {split: get_tagger_dataset(dataset_dict[split]) for split in splits}
    )
