"""Heuristic scoring engine for GEO/AEO Content Optimizer.

This module implements a deterministic, LLM-free multi-dimension scoring engine
that evaluates content across six GEO/AEO signals using spaCy for NLP analysis.

Dimensions scored:
    - Question-Answer Alignment: Does the content directly answer questions?
    - Entity Density: Are key named entities present and well-distributed?
    - Structured Formatting: Does content use headings, lists, and structure?
    - Citation Cues: Does content reference sources, statistics, or authorities?
    - Semantic Clarity: Is language precise and free of filler?
    - Content Depth: Is content substantive enough to serve as a reference?

All scoring functions are pure (no side effects) and return float scores in
the range [0, 100], making them fully unit-testable without external API calls.

Typical usage::

    from geo_aeo_optimizer.scorer import ContentScorer

    scorer = ContentScorer()
    result = scorer.score(content="Your article text here...", target_query="What is GEO?")
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from geo_aeo_optimizer.models import (
    AnalysisResult,
    ContentInput,
    DimensionKey,
    DimensionScore,
    Settings,
    get_dimension_display_name,
    get_settings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# spaCy model loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str) -> Language:
    """Load and cache a spaCy language model by name.

    Uses ``lru_cache`` so the model is only loaded once per process, regardless
    of how many requests are processed.  Subsequent calls return the cached
    instance instantly.

    Args:
        model_name: The spaCy model identifier (e.g. ``"en_core_web_sm"``).

    Returns:
        Language: The loaded spaCy ``Language`` pipeline.

    Raises:
        OSError: If the model has not been downloaded.  Run
            ``python -m spacy download <model_name>`` to fix this.
    """
    try:
        nlp = spacy.load(model_name)
        logger.info("spaCy model '%s' loaded successfully.", model_name)
        return nlp
    except OSError as exc:
        logger.error(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            model_name,
            model_name,
        )
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal analysis data container
# ---------------------------------------------------------------------------


@dataclass
class _ContentFeatures:
    """Intermediate feature container extracted from content during scoring.

    Holds all pre-computed text features so each dimension scorer can access
    shared data without re-parsing the document.

    Attributes:
        text: The original content string.
        target_query: Optional target query string.
        doc: The spaCy ``Doc`` object for the content.
        sentences: List of sentence strings extracted from the doc.
        word_count: Number of tokens that are not punctuation or whitespace.
        char_count: Number of characters in the text.
        lines: Raw lines of the content (split on newlines).
    """

    text: str
    target_query: Optional[str]
    doc: Doc
    sentences: list[str] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sentences = [sent.text.strip() for sent in self.doc.sents if sent.text.strip()]
        self.word_count = sum(1 for token in self.doc if not token.is_punct and not token.is_space)
        self.char_count = len(self.text)
        self.lines = self.text.splitlines()


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------


def score_qa_alignment(features: _ContentFeatures) -> tuple[float, str]:
    """Score Question-Answer Alignment (0-100).

    Measures whether the content directly addresses questions a user might ask.
    Looks for:
    - Explicit question marks (Q&A patterns)
    - Interrogative sentence starters (What, How, Why, When, Where, Who, Which,
      Can, Does, Is, Are, Should, Would, Could)
    - Direct answer indicators ("The answer is", "In short", "To summarize", etc.)
    - If a target query is provided, checks whether content contains keywords
      from the query

    Args:
        features: Pre-computed content features.

    Returns:
        tuple[float, str]: The raw score (0-100) and a human-readable explanation.
    """
    text = features.text
    sentences = features.sentences
    word_count = features.word_count

    if not sentences or word_count == 0:
        return 0.0, "Content is too short to evaluate question-answer alignment."

    score = 0.0
    signals: list[str] = []

    # --- Signal 1: Explicit question sentences (up to 30 points) ---
    question_pattern = re.compile(r"\?", re.MULTILINE)
    question_sentences = [s for s in sentences if question_pattern.search(s)]
    question_count = len(question_sentences)

    if question_count >= 3:
        score += 30.0
        signals.append(f"{question_count} question sentences detected.")
    elif question_count >= 1:
        score += min(30.0, question_count * 10.0)
        signals.append(f"{question_count} question sentence(s) detected.")
    else:
        signals.append("No question sentences detected.")

    # --- Signal 2: Interrogative starters (up to 25 points) ---
    interrogatives = (
        r"^(what|how|why|when|where|who|which|can|does|do|is|are|should|would|could|will)"
    )
    interrogative_re = re.compile(interrogatives, re.IGNORECASE | re.MULTILINE)
    interrogative_sentences = [
        s for s in sentences if interrogative_re.match(s.strip())
    ]
    interrogative_count = len(interrogative_sentences)

    if interrogative_count >= 3:
        score += 25.0
        signals.append(f"{interrogative_count} interrogative-style sentences.")
    elif interrogative_count >= 1:
        score += min(25.0, interrogative_count * 8.0)
        signals.append(f"{interrogative_count} interrogative-style sentence(s).")

    # --- Signal 3: Direct answer indicators (up to 20 points) ---
    answer_indicators = [
        r"\bthe answer is\b",
        r"\bin short\b",
        r"\bto summarize\b",
        r"\bto sum up\b",
        r"\bin summary\b",
        r"\bin brief\b",
        r"\bsimply put\b",
        r"\bessentially\b",
        r"\bspecifically\b",
        r"\bfor example\b",
        r"\bfor instance\b",
        r"\bnamely\b",
        r"\bthat is\b",
        r"\bi\.e\.\b",
        r"\be\.g\.\b",
        r"\bdefined as\b",
        r"\brefers to\b",
        r"\bknown as\b",
        r"\bmeans that\b",
    ]
    answer_hits = sum(
        1 for pattern in answer_indicators
        if re.search(pattern, text, re.IGNORECASE)
    )
    answer_contribution = min(20.0, answer_hits * 4.0)
    score += answer_contribution
    if answer_hits > 0:
        signals.append(f"{answer_hits} direct-answer indicator phrase(s) found.")
    else:
        signals.append("No direct-answer indicator phrases found.")

    # --- Signal 4: Target query keyword overlap (up to 25 points) ---
    if features.target_query:
        query_words = {
            w.lower()
            for w in re.findall(r"\b\w+\b", features.target_query)
            if len(w) > 3  # ignore very short stop words
        }
        if query_words:
            content_lower = text.lower()
            matched_words = {w for w in query_words if w in content_lower}
            overlap_ratio = len(matched_words) / len(query_words)
            query_contribution = min(25.0, overlap_ratio * 25.0)
            score += query_contribution
            signals.append(
                f"Target query overlap: {len(matched_words)}/{len(query_words)} keywords found."
            )
        else:
            # Query was all short words; give partial credit
            score += 12.5
            signals.append("Target query provided but contains only short words.")
    else:
        # No target query: redistribute the 25 points to a content-pattern bonus
        # Check for FAQ-style structure
        faq_pattern = re.compile(r"^(q:|question:|a:|answer:)", re.IGNORECASE | re.MULTILINE)
        faq_matches = len(faq_pattern.findall(text))
        if faq_matches >= 2:
            score += 20.0
            signals.append(f"FAQ-style Q/A labels detected ({faq_matches} matches).")
        elif faq_matches == 1:
            score += 10.0
            signals.append("Partial FAQ-style structure detected.")
        else:
            # Partial credit for structured definitions
            definition_re = re.compile(
                r"\b(is defined as|is a|are a|can be described as)\b", re.IGNORECASE
            )
            def_count = len(definition_re.findall(text))
            if def_count >= 2:
                score += 12.0
                signals.append(f"{def_count} definition-style sentences found.")
            elif def_count == 1:
                score += 6.0
                signals.append("One definition-style sentence found.")

    score = min(100.0, score)
    explanation = " ".join(signals) if signals else "Low question-answer alignment detected."
    return score, explanation


def score_entity_density(features: _ContentFeatures) -> tuple[float, str]:
    """Score Entity Density (0-100).

    Measures the richness and distribution of named entities (people, places,
    organizations, products, dates, quantities, etc.) and noun chunks.

    Args:
        features: Pre-computed content features.

    Returns:
        tuple[float, str]: The raw score (0-100) and a human-readable explanation.
    """
    doc = features.doc
    word_count = features.word_count

    if word_count == 0:
        return 0.0, "Content is empty; no entities to evaluate."

    signals: list[str] = []
    score = 0.0

    # --- Signal 1: Named entity count and density (up to 40 points) ---
    entities = doc.ents
    entity_count = len(entities)
    unique_entity_texts = {ent.text.lower() for ent in entities}
    unique_entity_count = len(unique_entity_texts)

    # Entity density: unique entities per 100 words
    entity_density = (unique_entity_count / word_count) * 100 if word_count > 0 else 0.0

    if entity_density >= 5.0:
        score += 40.0
        signals.append(
            f"High entity density: {unique_entity_count} unique entities "
            f"({entity_density:.1f} per 100 words)."
        )
    elif entity_density >= 3.0:
        score += 30.0
        signals.append(
            f"Moderate entity density: {unique_entity_count} unique entities "
            f"({entity_density:.1f} per 100 words)."
        )
    elif entity_density >= 1.5:
        score += 20.0
        signals.append(
            f"Low entity density: {unique_entity_count} unique entities "
            f"({entity_density:.1f} per 100 words)."
        )
    elif entity_count > 0:
        score += 10.0
        signals.append(
            f"Very low entity density: {unique_entity_count} unique entities found."
        )
    else:
        signals.append("No named entities detected.")

    # --- Signal 2: Entity type diversity (up to 25 points) ---
    entity_types = {ent.label_ for ent in entities}
    type_count = len(entity_types)

    if type_count >= 5:
        score += 25.0
        signals.append(f"Rich entity type diversity: {type_count} different entity types.")
    elif type_count >= 3:
        score += 18.0
        signals.append(f"Good entity type diversity: {type_count} different entity types.")
    elif type_count >= 2:
        score += 10.0
        signals.append(f"Limited entity type diversity: {type_count} entity types.")
    elif type_count == 1:
        score += 5.0
        signals.append("Only one entity type present.")
    else:
        signals.append("No entity types found.")

    # --- Signal 3: Noun chunk richness as a proxy for concept density (up to 20 points) ---
    noun_chunks = list(doc.noun_chunks)
    unique_chunks = {chunk.text.lower() for chunk in noun_chunks}
    chunk_density = (len(unique_chunks) / word_count) * 100 if word_count > 0 else 0.0

    if chunk_density >= 15.0:
        score += 20.0
        signals.append(f"High noun-chunk density: {len(unique_chunks)} unique concepts.")
    elif chunk_density >= 8.0:
        score += 13.0
        signals.append(f"Moderate noun-chunk density: {len(unique_chunks)} unique concepts.")
    elif chunk_density >= 4.0:
        score += 7.0
        signals.append(f"Low noun-chunk density: {len(unique_chunks)} unique concepts.")

    # --- Signal 4: Presence of specific high-value entity types (up to 15 points) ---
    high_value_types = {"ORG", "PERSON", "GPE", "PRODUCT", "WORK_OF_ART", "LAW", "EVENT"}
    high_value_found = entity_types & high_value_types
    hv_contribution = min(15.0, len(high_value_found) * 3.0)
    score += hv_contribution
    if high_value_found:
        signals.append(
            f"High-value entity types present: {', '.join(sorted(high_value_found))}."
        )

    score = min(100.0, score)
    explanation = " ".join(signals) if signals else "Entity density could not be evaluated."
    return score, explanation


def score_structured_formatting(features: _ContentFeatures) -> tuple[float, str]:
    """Score Structured Formatting (0-100).

    Evaluates whether the content uses markdown or HTML structural signals
    that AI parsers prefer: headings, bullet/numbered lists, bold/italic
    emphasis, code blocks, tables, and logical paragraph breaks.

    Args:
        features: Pre-computed content features.

    Returns:
        tuple[float, str]: The raw score (0-100) and a human-readable explanation.
    """
    text = features.text
    lines = features.lines
    word_count = features.word_count

    if word_count == 0:
        return 0.0, "Content is empty; formatting cannot be evaluated."

    score = 0.0
    signals: list[str] = []

    # --- Signal 1: Headings (markdown # or HTML h1-h6) (up to 25 points) ---
    md_heading_re = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)
    html_heading_re = re.compile(r"<h[1-6][^>]*>.*?</h[1-6]>", re.IGNORECASE | re.DOTALL)
    md_headings = md_heading_re.findall(text)
    html_headings = html_heading_re.findall(text)
    heading_count = len(md_headings) + len(html_headings)

    if heading_count >= 4:
        score += 25.0
        signals.append(f"{heading_count} headings detected (strong structure).")
    elif heading_count >= 2:
        score += 18.0
        signals.append(f"{heading_count} headings detected.")
    elif heading_count == 1:
        score += 10.0
        signals.append("1 heading detected.")
    else:
        signals.append("No headings detected.")

    # --- Signal 2: Lists — bullet or numbered (up to 25 points) ---
    bullet_re = re.compile(r"^\s*[-*+]\s+.+", re.MULTILINE)
    numbered_re = re.compile(r"^\s*\d+\.\s+.+", re.MULTILINE)
    bullet_items = bullet_re.findall(text)
    numbered_items = numbered_re.findall(text)
    list_item_count = len(bullet_items) + len(numbered_items)

    if list_item_count >= 6:
        score += 25.0
        signals.append(f"{list_item_count} list items detected (comprehensive lists).")
    elif list_item_count >= 3:
        score += 18.0
        signals.append(f"{list_item_count} list items detected.")
    elif list_item_count >= 1:
        score += 10.0
        signals.append(f"{list_item_count} list item(s) detected.")
    else:
        signals.append("No bullet or numbered list items detected.")

    # --- Signal 3: Emphasis — bold or italic (up to 15 points) ---
    bold_re = re.compile(r"(\*\*|__).+?\1", re.DOTALL)
    italic_re = re.compile(r"(?<!\*)\*(?!\*).+?(?<!\*)\*(?!\*)", re.DOTALL)
    html_bold_re = re.compile(r"<(b|strong)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
    emphasis_count = (
        len(bold_re.findall(text))
        + len(italic_re.findall(text))
        + len(html_bold_re.findall(text))
    )
    emphasis_contribution = min(15.0, emphasis_count * 3.0)
    score += emphasis_contribution
    if emphasis_count > 0:
        signals.append(f"{emphasis_count} emphasis marker(s) (bold/italic) found.")
    else:
        signals.append("No bold or italic emphasis detected.")

    # --- Signal 4: Code blocks or inline code (up to 10 points) ---
    code_block_re = re.compile(r"```[\s\S]*?```", re.DOTALL)
    inline_code_re = re.compile(r"`[^`]+`")
    code_count = len(code_block_re.findall(text)) + len(inline_code_re.findall(text))
    if code_count >= 1:
        score += min(10.0, code_count * 5.0)
        signals.append(f"{code_count} code block(s)/inline code found.")

    # --- Signal 5: Paragraph structure — blank-line-separated paragraphs (up to 15 points) ---
    # Count non-empty paragraph blocks
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    para_count = len(paragraphs)

    if para_count >= 5:
        score += 15.0
        signals.append(f"{para_count} well-separated paragraphs detected.")
    elif para_count >= 3:
        score += 10.0
        signals.append(f"{para_count} paragraphs detected.")
    elif para_count >= 2:
        score += 5.0
        signals.append(f"{para_count} paragraphs detected.")
    else:
        signals.append("Content appears to be a single unbroken block of text.")

    # --- Signal 6: Tables (up to 10 points) ---
    table_re = re.compile(r"^\|.+\|\s*$", re.MULTILINE)
    table_rows = table_re.findall(text)
    if len(table_rows) >= 2:  # at least a header and one data row
        score += 10.0
        signals.append("Markdown table detected.")

    score = min(100.0, score)
    explanation = " ".join(signals) if signals else "No structural formatting signals detected."
    return score, explanation


def score_citation_cues(features: _ContentFeatures) -> tuple[float, str]:
    """Score Citation Cues (0-100).

    Detects signals that indicate authoritative, citable content: statistical
    references, attribution phrases, source mentions, numerical evidence,
    quotes, and research/study references.

    Args:
        features: Pre-computed content features.

    Returns:
        tuple[float, str]: The raw score (0-100) and a human-readable explanation.
    """
    text = features.text
    word_count = features.word_count

    if word_count == 0:
        return 0.0, "Content is empty; citation cues cannot be evaluated."

    score = 0.0
    signals: list[str] = []

    # --- Signal 1: Statistical references — percentages, numbers with context (up to 25 points) ---
    stat_patterns = [
        r"\b\d+(\.\d+)?\s*%",                          # percentages
        r"\b\d{4}\b",                                    # years as references
        r"\$\d+(\.\d+)?(\s*[BMK])?\b",                  # dollar amounts
        r"\b\d+(\.\d+)?\s*(million|billion|thousand)\b", # large numbers
        r"\b(\d+)\s+out\s+of\s+(\d+)\b",               # ratios
    ]
    stat_hits = sum(
        len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in stat_patterns
    )
    stat_contribution = min(25.0, stat_hits * 5.0)
    score += stat_contribution
    if stat_hits > 0:
        signals.append(f"{stat_hits} statistical reference(s) found.")
    else:
        signals.append("No statistical references detected.")

    # --- Signal 2: Attribution and source phrases (up to 25 points) ---
    attribution_patterns = [
        r"\baccording to\b",
        r"\bper\s+(a|the|\w+)\s+report\b",
        r"\bstudy\s+(found|shows|suggests|indicates|reveals)\b",
        r"\bresearch\s+(found|shows|suggests|indicates|reveals)\b",
        r"\b(researchers|scientists|experts|analysts)\s+(found|say|report|note|argue|suggest)\b",
        r"\bpublished\s+(in|by)\b",
        r"\bcited\s+(in|by)\b",
        r"\bsource[sd]?\b",
        r"\breference[sd]?\b",
        r"\bvia\b",
        r"\bquoting\b",
        r"\bin a\s+\d{4}\s+(study|report|survey|paper)\b",
        r"\bthe\s+\w+\s+(study|report|survey|journal)\b",
    ]
    attribution_hits = sum(
        1 for pattern in attribution_patterns
        if re.search(pattern, text, re.IGNORECASE)
    )
    attribution_contribution = min(25.0, attribution_hits * 5.0)
    score += attribution_contribution
    if attribution_hits > 0:
        signals.append(f"{attribution_hits} attribution phrase(s) detected.")
    else:
        signals.append("No attribution phrases detected.")

    # --- Signal 3: Quoted content (up to 20 points) ---
    # Double-quoted strings or blockquote markdown
    quote_re = re.compile(r'"[^"]{10,}"', re.DOTALL)
    blockquote_re = re.compile(r"^>\s+.+", re.MULTILINE)
    quote_count = len(quote_re.findall(text)) + len(blockquote_re.findall(text))
    quote_contribution = min(20.0, quote_count * 7.0)
    score += quote_contribution
    if quote_count > 0:
        signals.append(f"{quote_count} quotation(s)/blockquote(s) found.")
    else:
        signals.append("No direct quotations detected.")

    # --- Signal 4: Named source mentions — URLs, journals, organizations (up to 15 points) ---
    url_re = re.compile(r"https?://[^\s]+")
    url_count = len(url_re.findall(text))

    # Patterns for named sources
    named_source_patterns = [
        r"\b(Harvard|MIT|Stanford|Oxford|WHO|UN|IEEE|NASA|CDC|FDA)\b",
        r"\bJournal\s+of\b",
        r"\bProceedings\s+of\b",
        r"\bAnnals\s+of\b",
        r"\barXiv\b",
        r"\bPubMed\b",
        r"\bWikipedia\b",
    ]
    named_source_hits = sum(
        1 for p in named_source_patterns if re.search(p, text, re.IGNORECASE)
    )
    source_contribution = min(15.0, (url_count + named_source_hits) * 5.0)
    score += source_contribution
    if url_count + named_source_hits > 0:
        signals.append(
            f"{url_count} URL(s) and {named_source_hits} named source(s) found."
        )
    else:
        signals.append("No URLs or named sources detected.")

    # --- Signal 5: Ranking/list authority signals (up to 15 points) ---
    authority_patterns = [
        r"\b(ranked|rated|awarded|recognized|certified)\b",
        r"\b(industry leader|market leader|best-in-class|gold standard)\b",
        r"\b(as of \d{4}|in \d{4}|since \d{4})\b",
        r"\b(first|second|third|top)\s+(place|position|ranked)\b",
    ]
    authority_hits = sum(
        1 for p in authority_patterns if re.search(p, text, re.IGNORECASE)
    )
    authority_contribution = min(15.0, authority_hits * 5.0)
    score += authority_contribution
    if authority_hits > 0:
        signals.append(f"{authority_hits} authority/ranking signal(s) detected.")

    score = min(100.0, score)
    explanation = " ".join(signals) if signals else "No citation cues detected."
    return score, explanation


def score_semantic_clarity(features: _ContentFeatures) -> tuple[float, str]:
    """Score Semantic Clarity (0-100).

    Measures how precise and unambiguous the language is. Penalises filler
    phrases, vague language, excessive passive voice, long average sentence
    length, and high lexical repetition. Rewards precise vocabulary,
    transition words, and well-structured sentences.

    Args:
        features: Pre-computed content features.

    Returns:
        tuple[float, str]: The raw score (0-100) and a human-readable explanation.
    """
    doc = features.doc
    text = features.text
    sentences = features.sentences
    word_count = features.word_count

    if word_count == 0:
        return 0.0, "Content is empty; semantic clarity cannot be evaluated."
    if not sentences:
        return 0.0, "No sentences detected."

    score = 50.0  # Start from a neutral midpoint and adjust
    signals: list[str] = []

    # --- Penalty 1: Filler and hedge phrases (up to -25 points) ---
    filler_patterns = [
        r"\bvery\s+\w+\b",
        r"\breally\s+\w+\b",
        r"\bkind\s+of\b",
        r"\bsort\s+of\b",
        r"\bbasically\b",
        r"\bliterally\b",
        r"\bactually\b",
        r"\bjust\b",
        r"\bquite\b",
        r"\brather\b",
        r"\bsomewhat\b",
        r"\bin\s+order\s+to\b",
        r"\bdue\s+to\s+the\s+fact\s+that\b",
        r"\bit\s+is\s+important\s+to\s+note\s+that\b",
        r"\bit\s+should\s+be\s+noted\s+that\b",
        r"\bit\s+is\s+worth\s+mentioning\s+that\b",
        r"\bplease\s+note\s+that\b",
        r"\bneedless\s+to\s+say\b",
        r"\bwithout\s+further\s+ado\b",
        r"\bin\s+today.s\s+world\b",
        r"\bat\s+the\s+end\s+of\s+the\s+day\b",
        r"\bthe\s+fact\s+of\s+the\s+matter\s+is\b",
    ]
    filler_hits = sum(
        len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in filler_patterns
    )
    # Penalise proportionally; cap at 25 points
    filler_penalty = min(25.0, filler_hits * 2.5)
    score -= filler_penalty
    if filler_hits > 0:
        signals.append(f"{filler_hits} filler/hedge phrase(s) detected (−{filler_penalty:.0f} pts).")
    else:
        signals.append("No significant filler phrases detected.")

    # --- Penalty 2: Average sentence length (penalise very long or very short) ---
    sent_lengths = [len(s.split()) for s in sentences if s]
    if sent_lengths:
        avg_len = sum(sent_lengths) / len(sent_lengths)
        # Ideal range: 12–22 words per sentence
        if avg_len > 35:
            penalty = min(15.0, (avg_len - 35) * 0.5)
            score -= penalty
            signals.append(
                f"Average sentence length is very long ({avg_len:.0f} words; −{penalty:.0f} pts)."
            )
        elif avg_len > 25:
            penalty = min(8.0, (avg_len - 25) * 0.5)
            score -= penalty
            signals.append(
                f"Average sentence length is somewhat long ({avg_len:.0f} words; −{penalty:.0f} pts)."
            )
        elif avg_len < 8:
            penalty = 5.0
            score -= penalty
            signals.append(
                f"Average sentence length is very short ({avg_len:.0f} words; −{penalty:.0f} pts)."
            )
        else:
            signals.append(f"Average sentence length is reasonable ({avg_len:.0f} words).")

    # --- Bonus: Lexical diversity (type-token ratio) (up to +20 points) ---
    # Only consider meaningful tokens (not stopwords or punctuation)
    meaningful_tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    ]
    if meaningful_tokens:
        unique_lemmas = set(meaningful_tokens)
        ttr = len(unique_lemmas) / len(meaningful_tokens)
        # TTR of 0.7+ → excellent diversity; 0.4–0.7 → good; <0.4 → repetitive
        if ttr >= 0.7:
            bonus = 20.0
            signals.append(f"High lexical diversity (TTR={ttr:.2f}; +{bonus:.0f} pts).")
        elif ttr >= 0.5:
            bonus = 12.0
            signals.append(f"Good lexical diversity (TTR={ttr:.2f}; +{bonus:.0f} pts).")
        elif ttr >= 0.35:
            bonus = 5.0
            signals.append(f"Moderate lexical diversity (TTR={ttr:.2f}; +{bonus:.0f} pts).")
        else:
            bonus = 0.0
            signals.append(
                f"Low lexical diversity (TTR={ttr:.2f}); content may be repetitive."
            )
        score += bonus

    # --- Bonus: Transition/cohesion words (up to +10 points) ---
    transition_patterns = [
        r"\bhowever\b", r"\btherefore\b", r"\bfurthermore\b", r"\bmoreover\b",
        r"\bconsequently\b", r"\bnevertheless\b", r"\badditionally\b", r"\bin contrast\b",
        r"\bon the other hand\b", r"\bas a result\b", r"\bfor example\b", r"\bspecifically\b",
        r"\bin conclusion\b", r"\bto summarize\b", r"\bin summary\b", r"\bfirst(ly)?\b",
        r"\bsecond(ly)?\b", r"\bfinally\b", r"\bsubsequently\b", r"\baccordingly\b",
    ]
    transition_hits = sum(
        1 for p in transition_patterns if re.search(p, text, re.IGNORECASE)
    )
    transition_bonus = min(10.0, transition_hits * 1.5)
    score += transition_bonus
    if transition_hits > 0:
        signals.append(f"{transition_hits} cohesion/transition word(s) detected (+{transition_bonus:.0f} pts).")

    score = max(0.0, min(100.0, score))
    explanation = " ".join(signals) if signals else "Semantic clarity evaluation completed."
    return score, explanation


def score_content_depth(features: _ContentFeatures) -> tuple[float, str]:
    """Score Content Depth (0-100).

    Evaluates whether the content is substantive enough to be a useful
    reference for an AI answer. Considers word count, topical coverage
    (variety of unique concepts), sentence complexity, and the presence
    of explanatory content.

    Args:
        features: Pre-computed content features.

    Returns:
        tuple[float, str]: The raw score (0-100) and a human-readable explanation.
    """
    doc = features.doc
    text = features.text
    sentences = features.sentences
    word_count = features.word_count

    if word_count == 0:
        return 0.0, "Content is empty; depth cannot be evaluated."

    score = 0.0
    signals: list[str] = []

    # --- Signal 1: Word count thresholds (up to 30 points) ---
    if word_count >= 800:
        score += 30.0
        signals.append(f"Substantial content length: {word_count} words.")
    elif word_count >= 400:
        score += 22.0
        signals.append(f"Good content length: {word_count} words.")
    elif word_count >= 200:
        score += 15.0
        signals.append(f"Moderate content length: {word_count} words.")
    elif word_count >= 100:
        score += 8.0
        signals.append(f"Short content: {word_count} words.")
    else:
        score += 3.0
        signals.append(f"Very short content: {word_count} words (target ≥ 200 for depth).")

    # --- Signal 2: Unique concept coverage via unique lemmas (up to 25 points) ---
    content_lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
        and len(token.text) > 2
    ]
    unique_concepts = len(set(content_lemmas))

    if unique_concepts >= 100:
        score += 25.0
        signals.append(f"Rich concept coverage: {unique_concepts} unique content terms.")
    elif unique_concepts >= 60:
        score += 18.0
        signals.append(f"Good concept coverage: {unique_concepts} unique content terms.")
    elif unique_concepts >= 30:
        score += 12.0
        signals.append(f"Moderate concept coverage: {unique_concepts} unique content terms.")
    elif unique_concepts >= 10:
        score += 6.0
        signals.append(f"Limited concept coverage: {unique_concepts} unique content terms.")
    else:
        signals.append(f"Very limited concept coverage: {unique_concepts} unique content terms.")

    # --- Signal 3: Explanatory depth indicators (up to 20 points) ---
    explanatory_patterns = [
        r"\bbecause\b",
        r"\bsince\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bconsequently\b",
        r"\bas\s+a\s+result\b",
        r"\bthis\s+(means|implies|suggests|shows|demonstrates)\b",
        r"\bwhich\s+(means|implies|suggests|shows|demonstrates)\b",
        r"\bin\s+other\s+words\b",
        r"\bto\s+(illustrate|demonstrate|clarify|explain)\b",
        r"\bfor\s+example\b",
        r"\bfor\s+instance\b",
        r"\bsuch\s+as\b",
        r"\bincluding\b",
        r"\bnamely\b",
    ]
    explanatory_hits = sum(
        len(re.findall(p, text, re.IGNORECASE))
        for p in explanatory_patterns
    )
    explanatory_contribution = min(20.0, explanatory_hits * 2.0)
    score += explanatory_contribution
    if explanatory_hits > 0:
        signals.append(
            f"{explanatory_hits} explanatory/causal phrase(s) detected."
        )
    else:
        signals.append("Few explanatory phrases detected; content may lack analytical depth.")

    # --- Signal 4: Sentence count and variety (up to 15 points) ---
    sentence_count = len(sentences)
    if sentence_count >= 20:
        score += 15.0
        signals.append(f"{sentence_count} sentences detected (comprehensive coverage).")
    elif sentence_count >= 10:
        score += 10.0
        signals.append(f"{sentence_count} sentences detected.")
    elif sentence_count >= 5:
        score += 6.0
        signals.append(f"{sentence_count} sentences detected.")
    elif sentence_count >= 2:
        score += 3.0
        signals.append(f"{sentence_count} sentences detected.")
    else:
        signals.append("Very few sentences; content lacks depth.")

    # --- Signal 5: Comparative and evaluative language (up to 10 points) ---
    comparative_patterns = [
        r"\bcompared\s+(to|with)\b",
        r"\bin\s+contrast\b",
        r"\bon\s+the\s+other\s+hand\b",
        r"\bunlike\b",
        r"\bwhereas\b",
        r"\b(advantage|disadvantage|benefit|drawback|pro|con)s?\b",
        r"\b(better|worse|superior|inferior)\s+than\b",
        r"\bthe\s+(best|worst|most|least)\b",
        r"\bwhen\s+(compared|evaluated)\b",
    ]
    comparative_hits = sum(
        1 for p in comparative_patterns if re.search(p, text, re.IGNORECASE)
    )
    comparative_contribution = min(10.0, comparative_hits * 2.5)
    score += comparative_contribution
    if comparative_hits > 0:
        signals.append(
            f"{comparative_hits} comparative/evaluative pattern(s) detected."
        )

    score = min(100.0, score)
    explanation = " ".join(signals) if signals else "Content depth evaluation completed."
    return score, explanation


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------


class ContentScorer:
    """spaCy-based multi-dimension heuristic scoring engine.

    Evaluates content across six GEO/AEO dimensions and returns an
    ``AnalysisResult`` with a weighted composite score and per-dimension
    breakdowns.  All scoring is deterministic and LLM-free.

    Attributes:
        settings: The application ``Settings`` instance used for weights
            and configuration.
        nlp: The loaded spaCy ``Language`` pipeline.

    Example::

        scorer = ContentScorer()
        result = scorer.score(
            content="Python is a programming language...",
            target_query="What is Python?",
        )
        print(result.composite_score)  # e.g. 72.4
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the scorer with optional custom settings.

        Args:
            settings: Optional ``Settings`` instance.  If ``None``, the
                global ``get_settings()`` singleton is used.
        """
        self.settings: Settings = settings if settings is not None else get_settings()
        self.nlp: Language = _load_spacy_model(self.settings.spacy_model)

    def _build_features(self, content: str, target_query: str | None) -> _ContentFeatures:
        """Parse content with spaCy and build the shared feature container.

        Args:
            content: The raw content string to analyse.
            target_query: Optional target search query.

        Returns:
            _ContentFeatures: Pre-computed features ready for dimension scorers.
        """
        doc = self.nlp(content)
        return _ContentFeatures(
            text=content,
            target_query=target_query,
            doc=doc,
        )

    def _score_all_dimensions(
        self, features: _ContentFeatures
    ) -> list[DimensionScore]:
        """Run all six dimension scorers and return a list of ``DimensionScore`` objects.

        Args:
            features: Pre-computed content features shared by all scorers.

        Returns:
            list[DimensionScore]: One score object per dimension.
        """
        weights = self.settings.weights_by_dimension

        dimension_scorers = [
            (
                DimensionKey.QA_ALIGNMENT,
                score_qa_alignment,
                weights["qa_alignment"],
            ),
            (
                DimensionKey.ENTITY_DENSITY,
                score_entity_density,
                weights["entity_density"],
            ),
            (
                DimensionKey.STRUCTURED_FORMATTING,
                score_structured_formatting,
                weights["structured_formatting"],
            ),
            (
                DimensionKey.CITATION_CUES,
                score_citation_cues,
                weights["citation_cues"],
            ),
            (
                DimensionKey.SEMANTIC_CLARITY,
                score_semantic_clarity,
                weights["semantic_clarity"],
            ),
            (
                DimensionKey.CONTENT_DEPTH,
                score_content_depth,
                weights["content_depth"],
            ),
        ]

        dimension_scores: list[DimensionScore] = []
        for dimension_key, scorer_fn, weight in dimension_scorers:
            try:
                raw_score, explanation = scorer_fn(features)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Dimension scorer '%s' raised an unexpected error: %s",
                    dimension_key.value,
                    exc,
                    exc_info=True,
                )
                raw_score = 0.0
                explanation = f"Scoring error for this dimension: {exc}"

            dimension_scores.append(
                DimensionScore.create(
                    dimension=dimension_key,
                    display_name=get_dimension_display_name(dimension_key),
                    raw_score=raw_score,
                    weight=weight,
                    explanation=explanation,
                )
            )

        return dimension_scores

    def score(
        self,
        content: str,
        target_query: str | None = None,
    ) -> AnalysisResult:
        """Analyse content and return a complete ``AnalysisResult``.

        This is the primary public API of the scorer.  It parses the content
        with spaCy, runs all six dimension scorers, and builds the composite
        ``AnalysisResult`` with improvement priorities assigned.

        Args:
            content: The text content to analyse.  Should be at least 50
                characters long for meaningful results.
            target_query: Optional target search query or keyword phrase used
                to improve the accuracy of the QA Alignment dimension.

        Returns:
            AnalysisResult: Fully populated analysis result with composite
                score, per-dimension breakdowns, and improvement priorities.
                The ``suggestions`` list will always be empty — call the
                suggestions module separately to populate AI suggestions.

        Raises:
            OSError: If the spaCy model is not installed.
            ValueError: If ``content`` is empty after stripping.
        """
        stripped_content = content.strip()
        if not stripped_content:
            raise ValueError("Content must not be empty.")

        word_count = len(stripped_content.split())
        char_count = len(stripped_content)

        logger.debug(
            "Scoring content: %d words, %d chars, query=%r",
            word_count,
            char_count,
            target_query,
        )

        features = self._build_features(stripped_content, target_query)
        dimension_scores = self._score_all_dimensions(features)

        return AnalysisResult.build(
            dimensions=dimension_scores,
            settings=self.settings,
            suggestions=[],
            target_query=target_query,
            content_word_count=word_count,
            content_char_count=char_count,
            error_message=None,
        )

    def score_from_input(self, content_input: ContentInput) -> AnalysisResult:
        """Convenience wrapper that accepts a ``ContentInput`` model directly.

        Args:
            content_input: The validated request model from the API layer.

        Returns:
            AnalysisResult: Fully populated analysis result (without AI
                suggestions — those are added by the suggestions module).
        """
        return self.score(
            content=content_input.content,
            target_query=content_input.target_query,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def score_content(
    content: str,
    target_query: str | None = None,
    settings: Settings | None = None,
) -> AnalysisResult:
    """Module-level convenience function for one-shot content scoring.

    Creates a ``ContentScorer`` instance and immediately scores the provided
    content.  For repeated scoring (e.g. in a long-running server), prefer
    instantiating ``ContentScorer`` once and calling ``.score()`` on it
    to avoid repeated spaCy model lookups (even though the model itself is
    cached via ``lru_cache``).

    Args:
        content: The text content to analyse.
        target_query: Optional target search query.
        settings: Optional custom ``Settings``; defaults to the global
            singleton.

    Returns:
        AnalysisResult: Fully populated analysis result without AI suggestions.

    Raises:
        OSError: If the spaCy model is not installed.
        ValueError: If ``content`` is empty after stripping.
    """
    scorer = ContentScorer(settings=settings)
    return scorer.score(content=content, target_query=target_query)
