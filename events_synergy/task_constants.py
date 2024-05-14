
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

SUMMARIZATION_TEMPLATE = "Summarize the following article:\n\n{{document}}"
