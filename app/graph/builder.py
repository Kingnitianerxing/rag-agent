import json
import logging
import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_spacy_cache = None

EXTRACTION_PROMPT = """Extract entity-relationship triples from the following text.
Return ONLY a JSON array of objects with keys: head, relation, tail.

Text: {text}

JSON:"""


class GraphBuilder:
    def __init__(self, extractor_type: str = "llm", llm: BaseChatModel | None = None):
        self.extractor_type = extractor_type
        self.llm = llm
        self._nlp = None
        if extractor_type == "nlp":
            self._nlp = self._load_spacy()

    def extract(self, documents: list[Document]) -> list[dict]:
        if self.extractor_type == "llm":
            return self._extract_llm(documents)
        elif self.extractor_type == "nlp":
            return self._extract_nlp(documents)
        return []

    def _extract_llm(self, documents: list[Document]) -> list[dict]:
        if not self.llm:
            return []
        all_triples = []
        for doc in documents:
            prompt = EXTRACTION_PROMPT.format(text=doc.page_content)
            response = self.llm.invoke(prompt)
            from app.core.factories import message_text

            triples = self._parse_response(message_text(response.content))
            all_triples.extend(triples)
        return all_triples

    def _extract_nlp(self, documents: list[Document]) -> list[dict]:
        if self._nlp is None:
            return []
        triples = []
        for doc in documents:
            spacy_doc = self._nlp(doc.page_content)
            entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
            for i, (text_a, _) in enumerate(entities):
                for text_b, _ in entities[i + 1:]:
                    triples.append({"head": text_a, "relation": "related_to", "tail": text_b})
        return triples

    @staticmethod
    def _load_spacy():
        global _spacy_cache
        if _spacy_cache is not None:
            return _spacy_cache
        import spacy
        try:
            _spacy_cache = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found, using blank model")
            _spacy_cache = spacy.blank("en")
        return _spacy_cache

    def _parse_response(self, content: str) -> list[dict]:
        try:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if not match:
                logger.warning("No JSON array found in LLM response: %s...", content[:200])
                return []
            triples = json.loads(match.group())
            if not isinstance(triples, list):
                logger.warning("Expected a JSON array of triples, got %s", type(triples).__name__)
                return []
            valid = []
            for triple in triples:
                if not isinstance(triple, dict):
                    logger.warning("Skipping non-object triple: %s", triple)
                    continue
                head, relation, tail = triple.get("head"), triple.get("relation"), triple.get("tail")
                # Require non-empty head/relation/tail: a null or blank head/tail
                # cannot be a graph node (networkx raises "None cannot be a node").
                if head and relation and tail:
                    valid.append({"head": str(head), "relation": str(relation), "tail": str(tail)})
                else:
                    logger.warning("Skipping triple with missing/empty fields: %s", triple)
            logger.info("Parsed %d triples from LLM response", len(valid))
            return valid
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.error("Failed to parse LLM response: %s — content: %s...", exc, content[:200])
            return []
