from tqdm import tqdm
from transformers import SummarizationPipeline
from typing import List

from ..events_pipeline import EventsPipeline, pipe


def summarize(
    documents: List[str],
    model=None,
    tokenizer=None,
    generation_config=None,
    batch_size: int = 8,
):
    summary_pipeline = EventsPipeline(
        model=model,
        tokenizer=tokenizer,
        task_prefix="summarize",
        generation_config=generation_config,
        framework="pt",
    )

    summaries = pipe(summary_pipeline, documents, batch_size, desc="Summarizing")

    summaries = [summary for summary in summaries]

    return summaries

