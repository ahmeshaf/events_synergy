from datasets import Dataset


def get_srl_dataset(dataset: Dataset) -> Dataset:
    split_dict_array = dataset.to_pandas().to_dict("records")

    srl_records = [
        {
            "sentence_id": "<sep>".join((str(record["doc_id"]), str(record["sentence_id"]))),
            "prompt": f"SRL for [{record['mention_text']}]: " + record["marked_sentence"],
            "response": record["srl_response"],
            "sentence": record["sentence"],
            "gold_mention": "<sep>".join(
                (
                    str(record["doc_id"]),
                    str(record["sentence_id"]),
                    record["mention_text"],
                    str(record["start_char"]),
                    str(record["end_char"]),
                )
            ),
        }
        for record in split_dict_array
    ]

    return Dataset.from_list(srl_records)


if __name__ == "__main__":
    pass
