import jinja2
from tqdm import tqdm
from transformers.pipelines import Text2TextGenerationPipeline
from transformers import GenerationConfig

from ...coreference.utils import cluster
from ...coreference.coref_resolver import MentionsCorefResolver
from ...event_tagging.event_tagger import event_tagger_pipeline
from ...task_constants import COREF_TEMPLATE, SUMMARIZATION_TEMPLATE

from ...events_pipeline import get_model_tokenizer_generation_config


class SummarizerMentionCorefResolver(MentionsCorefResolver):
    def __init__(
        self, model_name, summary_generation_config, coref_generation_config, **kwargs
    ):
        self.model, self.tokenizer, _ = get_model_tokenizer_generation_config(
            model_name
        )
        self.summary_generation_config = summary_generation_config
        self.coref_generation_config = coref_generation_config
        super().__init__(
            self.model,
            self.tokenizer,
            self.generation_config,
            context_key="summarized_doc",
            **kwargs
        )

    def __call__(self, mention_map, batch_size=32, **kwargs):
        # add summaries for the mentions in mention_map
        # mention_map[mention_id]["summarized_doc"] = summary

        summarization_pipeline = Text2TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            generation_config=GenerationConfig(
                **self.summary_generation_config["generation"]
            ))

        prompts = [
            jinja2.Template(SUMMARIZATION_TEMPLATE).render(
                document=mention_map[mention_id]["marked_doc"]
            )
            for mention_id in mention_map.keys()
        ]

        summaries = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Summarizing..."):
            prompt_batch = prompts[i: i + batch_size]
            summaries.extend(
                output["generated_text"]
                for output in summarization_pipeline(prompt_batch, **kwargs)
            )

        for i, mention_id in enumerate(mention_map.keys()):
            mention_map[mention_id]["summarized_doc"] = summaries[i]

        return super().__call__(mention_map, batch_size=batch_size, **kwargs)


class SummarizerDocumentCorefResolver(MentionsCorefResolver):
    def __init__(self, mention_tagger, summary_generation_config, **kwargs):
        self.mention_tagger = mention_tagger
        self.batch_size = summary_generation_config["batch_size"]

        super().__init__(context_key="summarized_doc", **kwargs)

        self.summarization_pipeline = Text2TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            generation_config=GenerationConfig(
                **summary_generation_config["generation"]
            ),
        )

    def __call__(self, docs, **kwargs):
        """
        :param docs: each document is of the form:
            {
                "topic": "topic_1",
                "doc_id": "doc_1",
                "sentences": [
                    {
                        "sentence_id": "s1",
                        "sentence": "This is a sentence.",
                    },
                    {
                        "sentence_id": "s2",
                        "sentence": "This is another sentence.",
                    },
                    ...
                ],
                "doc" : "This is a sentence. This is another sentence.
            }
        :param kwargs:
        :return:
        """
        topic_doc_sentence_ids = [
            (d["topic"], d["doc_id"], s["sentence_id"], s["sentence"], d["doc"])
            for d in docs
            for s in d["sentences"]
        ]
        sentences = [
            sentence["sentence"] for doc in docs for sentence in doc["sentences"]
        ]

        event_triggers = event_tagger_pipeline(
            self.mention_tagger, sentences, batch_size=8
        )

        mention_map = {}
        for (topic, doc_id, sentence_id, sentence, doc), triggers in zip(
            topic_doc_sentence_ids, event_triggers
        ):

            for i, trigger in enumerate(triggers):
                (mention_txt, (start, end)) = trigger
                mention_id = "_".join(
                    [topic, doc_id, sentence_id, str(start), str(end)]
                )
                mention_map[mention_id] = {
                    "topic": topic,
                    "doc_id": doc_id,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "mention_text": mention_txt,
                    "start_char": start,
                    "end_char": end,
                    "marked_sentence": sentence[:start]
                    + " <m> "
                    + mention_txt
                    + " </m> "
                    + sentence[end:],
                    "mention_id": mention_id,
                    "split": "predict",
                    "men_type": "evt",
                    "lemma": mention_txt,
                    "sentence_tokens": sentence.split(),
                    "marked_doc": doc.replace(
                        sentence,
                        sentence[:start]
                        + " <m> "
                        + mention_txt
                        + " </m> "
                        + sentence[end:],
                    ),
                }

        # Summarize mentions
        prompts = [
            jinja2.Template(SUMMARIZATION_TEMPLATE).render(
                document=mention_map[mention_id]["marked_doc"]
            )
            for mention_id in mention_map.keys()
        ]
        summaries = []

        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Summarizing"):
            prompt_batch = prompts[i : i + self.batch_size]
            summaries.extend(
                output["generated_text"]
                for output in self.summarization_pipeline(prompt_batch, **kwargs)
            )

        for i, mention_id in enumerate(mention_map.keys()):
            mention_map[mention_id]["summarized_doc"] = summaries[i]

        return super().__call__(mention_map, **kwargs)
