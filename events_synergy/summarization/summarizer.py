from tqdm import tqdm
from transformers import Text2TextGenerationPipeline
from typing import List

from ..events_pipeline import EventsPipeline


def summarize(
    documents: List[str],
    model=None,
    tokenizer=None,
    generation_config=None,
    batch_size: int = 8,
):
    summaries = []
    summary_pipeline = EventsPipeline(
        model=model,
        tokenizer=tokenizer,
        task_prefix="summarize",
        generation_config=generation_config,
        framework="pt",
    )

    for i in tqdm(
        range(0, len(documents), batch_size), total=len(documents), desc="Summarizing"
    ):
        batch = documents[i: i + batch_size]
        batch_summaries = summary_pipeline(batch)
        summaries.extend([summary[0] for summary in batch_summaries])
    print(summaries)
    return summaries
