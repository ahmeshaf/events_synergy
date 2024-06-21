
TRIGGERS_PREFIX = "triggers"
ENTITIES_PREFIX = "entities"
SRL_PREFIX = "SRL"
COREFERENCE_PREFIX = "Coreference"

COREF_TEMPLATE = """
Coreference:
<m> {{ mention_text_1 }} </m> in {{ mention1_context }}
</s>
<m> {{ mention_text_2 }} </m> in {{ mention2_context }}
"""

COREF_POSITIVE_LABEL = "Yes"
COREF_NEGATIVE_LABEL = "No"

SUMMARIZATION_TEMPLATE = r"Summarize the following article:\n\n{{document}}"
