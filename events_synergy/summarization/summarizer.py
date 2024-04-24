from tqdm import tqdm
from transformers import Text2TextGenerationPipeline
from typing import List


def summarize(
    documents: List[str],
    model=None,
    tokenizer=None,
    generation_config=None,
    batch_size: int = 8,
):
    summaries = []
    text2text_pipeline = Text2TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        framework="pt",
    )

    for i in tqdm(
        range(0, len(documents), batch_size), total=len(documents), desc="Summarizing"
    ):
        batch = documents[i: i + batch_size]
        batch_summaries = text2text_pipeline(batch)
        summaries.extend([summary["generated_text"] for summary in batch_summaries])

    return summaries
