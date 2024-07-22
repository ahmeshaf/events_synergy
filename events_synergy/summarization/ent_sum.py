import csv
import logging

import typer
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

app = typer.Typer()


# Configure logging
logging.basicConfig(
    filename="error_log_entsum.txt", level=logging.ERROR, format="%(message)s"
)


@app.command()
def check_charoffset():
    ent_sum = load_dataset("events-synergy/entsum")
    dataset = ent_sum["train"]
    with open("error_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Example Index", "Substring", "Surfaces", "Summary", "Context"]
        )  # Write header

        for i, example in enumerate(tqdm(dataset)):
            entity = example["entity"]
            start_char_offsets = entity["startCharOffset"]
            end_char_offsets = entity["endCharOffset"]
            surfaces = set(entity["surface"])

            doc = example["context"]
            summary = example["summary"]

            for start_char, end_char in zip(start_char_offsets, end_char_offsets):
                substring = doc[int(start_char) : int(end_char) + 1]
                if substring not in surfaces:
                    writer.writerow([i, substring, list(surfaces), summary, doc])


@app.command()
def create_marked_summary_dataset():
    dataset = load_dataset("events-synergy/entsum")["train"]
    datapoints = []
    for i, example in enumerate(tqdm(dataset)):
        entity = example["entity"]
        start_char_offsets = entity["startCharOffset"]
        end_char_offsets = entity["endCharOffset"]
        surfaces = set(entity["surface"])
        doc = example["context"]
        summary = example["summary"]
        for start_char, end_char in zip(start_char_offsets, end_char_offsets):
            substring = doc[int(start_char) : int(end_char)].strip()
            substring.replace("\n", " ")
            substring_plus_one = (
                doc[int(start_char) : int(end_char) + 1].strip().replace("\n", " ")
            )
            substring_plus_one_start = (
                doc[int(start_char) + 1 : int(end_char)].strip().replace("\n", " ")
            )
            substring_plus_one_plus_one = (
                doc[int(start_char) + 1 : int(end_char) + 1].strip().replace("\n", " ")
            )
            if substring in surfaces:
                pass
            elif substring_plus_one in surfaces:
                end_char += 1
            elif substring_plus_one_start in surfaces:
                start_char += 1
            elif substring_plus_one_plus_one in surfaces:
                start_char += 1
                end_char += 1
            else:
                print(f"Error in example {i}: {substring} not found in {surfaces}")
                # assert (
                #     False
                # ), f"Error in example {i}: {substring} not found in {surfaces}"

            marked_doc = (
                doc[:start_char]
                + "<m> "
                + doc[start_char:end_char]
                + " </m>"
                + doc[end_char:]
            )
            prompt = (
                f"Summarize <m> {substring} </m> in the following"
                f" article:\n\n{marked_doc}"
            )
            response = summary[0]
            datapoints.append({"prompt": prompt, "response": response})
    dataset_dict = DatasetDict({"train": Dataset.from_list(datapoints[:5])})
    # push to huggingface hub repo "events-synergy/entsum_processed"
    dataset_dict.push_to_hub("events-synergy/entsum_processed_sample")


if __name__ == "__main__":
    app()
