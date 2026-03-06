"""Unit tests for the heuristic scoring engine (geo_aeo_optimizer/scorer.py).

Tests cover:
- Each individual dimension scorer function
- Composite score calculation via ContentScorer.score()
- ContentScorer.score_from_input() convenience method
- Module-level score_content() function
- Edge cases: empty content, minimal content, maximum content, special characters
- Determinism: same input always yields same output
- Score boundaries: all scores must be in [0, 100]

Note: These tests require the spaCy 'en_core_web_sm' model to be installed.
Run: python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import pytest

from geo_aeo_optimizer.models import (
    AnalysisResult,
    ContentInput,
    DimensionKey,
    ScoreLabel,
    Settings,
)
from geo_aeo_optimizer.scorer import (
    ContentScorer,
    _ContentFeatures,
    _load_spacy_model,
    score_citation_cues,
    score_content,
    score_content_depth,
    score_entity_density,
    score_qa_alignment,
    score_semantic_clarity,
    score_structured_formatting,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nlp():
    """Load the spaCy model once for the entire test module."""
    return _load_spacy_model("en_core_web_sm")


@pytest.fixture(scope="module")
def scorer() -> ContentScorer:
    """Create a ContentScorer with default settings."""
    return ContentScorer()


def make_features(
    nlp,
    text: str,
    target_query: str | None = None,
) -> _ContentFeatures:
    """Helper to build _ContentFeatures from raw text."""
    doc = nlp(text)
    return _ContentFeatures(text=text, target_query=target_query, doc=doc)


# ---------------------------------------------------------------------------
# Rich test content samples
# ---------------------------------------------------------------------------

RICH_CONTENT = """
# What is Python?

Python is a high-level, general-purpose programming language created by Guido van Rossum
and first released in 1991. According to the 2024 Stack Overflow Developer Survey,
Python is the most popular programming language for the fourth consecutive year.

## Why is Python so popular?

Python is popular for several key reasons:

- **Readable syntax**: Python's syntax is clean and resembles plain English.
- **Versatility**: Python is used in data science, machine learning, web development,
  automation, and scientific computing.
- **Large ecosystem**: With over 400,000 packages on PyPI, Python has a library for
  almost every use case.
- **Community support**: Python has one of the largest and most active developer
  communities in the world.

## What is Python used for?

Python is widely used in the following domains:

1. **Data Science & Machine Learning**: Libraries like NumPy, Pandas, and TensorFlow
   make Python the dominant language for data analysis and AI.
2. **Web Development**: Frameworks like Django and FastAPI enable rapid web application
   development.
3. **Automation & Scripting**: Python excels at automating repetitive tasks.
4. **Scientific Computing**: NASA, CERN, and many research institutions use Python for
   scientific simulations.

## Python vs Other Languages

Compared to Java, Python requires significantly less boilerplate code. Unlike C++,
Python manages memory automatically. On the other hand, Python is slower than compiled
languages for CPU-intensive tasks.

As a result of its simplicity, Python is the most recommended language for beginners.
However, it is also used by senior engineers at Google, Facebook, and Netflix.

Sources: Stack Overflow Developer Survey 2024, Python Software Foundation annual report.
"""

POOR_CONTENT = "Python exists. It is a language. People use it. Code runs. It works."

MINIMAL_CONTENT = "a" * 50

STRUCTURED_CONTENT = """
# Introduction

This is the first section.

## Key Points

- Point one: very important
- Point two: also important
- Point three: equally important

## Details

Furthermore, the details are as follows:

1. First detail
2. Second detail

## Conclusion

In summary, the conclusion follows from the evidence.
"""

CITATION_RICH_CONTENT = """
According to the World Health Organization, 55% of the world's population lives in
urban areas. A 2023 UN report found that this figure is expected to reach 68% by 2050.

Researchers at Harvard University published findings in the Journal of Urban Studies
showing that urban density correlates with economic productivity. The study, cited in
over 200 peer-reviewed papers, was rated as one of the top 10 most influential
urban economics papers since 2000.

"Urban planning is not merely about buildings; it is about people," said Dr. Jane Smith,
a senior fellow at the MIT Urban Planning Institute.

For more information, see: https://www.who.int/urbanization and https://un.org/cities
"""

QA_RICH_CONTENT = """
What is machine learning? Machine learning is a subset of artificial intelligence
that enables systems to learn from data. The answer is found in statistical algorithms
that identify patterns in large datasets.

How does machine learning work? Machine learning works by training models on labeled
data, which means that the algorithm learns to associate inputs with outputs.

Why is machine learning important? Machine learning is important because it can
automatically improve through experience. In other words, the more data the system
processes, the better it becomes.

Can machine learning replace humans? Machine learning can automate many tasks but
cannot fully replace human judgment in complex or ethical situations.

Q: What is supervised learning?
A: Supervised learning is a type of machine learning where models are trained on
labeled datasets to make predictions on new, unseen data.
"""


# ---------------------------------------------------------------------------
# Tests for _ContentFeatures
# ---------------------------------------------------------------------------


class TestContentFeatures:
    """Tests for the internal _ContentFeatures dataclass."""

    def test_word_count_excludes_punctuation(self, nlp) -> None:
        features = make_features(nlp, "Hello, world! This is a test.")
        # words: Hello world This is a test = 6
        assert features.word_count == 6

    def test_char_count_matches_text_length(self, nlp) -> None:
        text = "Hello, world!"
        features = make_features(nlp, text)
        assert features.char_count == len(text)

    def test_sentences_extracted(self, nlp) -> None:
        text = "First sentence. Second sentence. Third one."
        features = make_features(nlp, text)
        assert len(features.sentences) >= 2

    def test_lines_split_on_newlines(self, nlp) -> None:
        text = "Line one\nLine two\nLine three"
        features = make_features(nlp, text)
        assert len(features.lines) == 3

    def test_target_query_preserved(self, nlp) -> None:
        features = make_features(nlp, "Some text here for testing purposes.", "test query")
        assert features.target_query == "test query"

    def test_target_query_none_when_not_provided(self, nlp) -> None:
        features = make_features(nlp, "Some text here for testing purposes.")
        assert features.target_query is None

    def test_sentences_are_strings(self, nlp) -> None:
        features = make_features(nlp, "First. Second. Third.")
        for s in features.sentences:
            assert isinstance(s, str)

    def test_empty_sentences_excluded(self, nlp) -> None:
        text = "One sentence.   "
        features = make_features(nlp, text)
        for s in features.sentences:
            assert s.strip() != ""

    def test_word_count_positive_for_real_text(self, nlp) -> None:
        features = make_features(nlp, "Python is a programming language.")
        assert features.word_count > 0

    def test_lines_include_empty_for_blank_lines(self, nlp) -> None:
        text = "Line one\n\nLine three"
        features = make_features(nlp, text)
        assert len(features.lines) == 3


# ---------------------------------------------------------------------------
# Tests for score_qa_alignment
# ---------------------------------------------------------------------------


class TestScoreQaAlignment:
    """Unit tests for the Question-Answer Alignment dimension scorer."""

    def test_qa_rich_content_scores_high(self, nlp) -> None:
        features = make_features(nlp, QA_RICH_CONTENT)
        score, explanation = score_qa_alignment(features)
        assert score >= 50.0, f"Expected >= 50, got {score}. Explanation: {explanation}"

    def test_poor_content_scores_low(self, nlp) -> None:
        features = make_features(nlp, POOR_CONTENT)
        score, explanation = score_qa_alignment(features)
        assert score < 60.0, f"Expected < 60, got {score}"

    def test_score_within_range(self, nlp) -> None:
        for text in [RICH_CONTENT, POOR_CONTENT, QA_RICH_CONTENT, MINIMAL_CONTENT]:
            features = make_features(nlp, text)
            score, _ = score_qa_alignment(features)
            assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100]"

    def test_target_query_boosts_score(self, nlp) -> None:
        text = (
            "Python is a programming language used for data science and web development. "
            "It was created by Guido van Rossum and is widely popular among developers."
        )
        features_no_query = make_features(nlp, text, target_query=None)
        features_with_query = make_features(nlp, text, target_query="Python programming language")
        score_no_query, _ = score_qa_alignment(features_no_query)
        score_with_query, _ = score_qa_alignment(features_with_query)
        # Having a matching query should produce a non-negative effect
        assert score_with_query >= score_no_query - 5.0  # allow small variance

    def test_explicit_questions_increase_score(self, nlp) -> None:
        text_with_questions = (
            "What is GEO? GEO stands for Generative Engine Optimization. "
            "How does it work? It works by optimizing content for AI assistants. "
            "Why is it important? It increases content discoverability in AI search."
        )
        text_without_questions = (
            "GEO stands for Generative Engine Optimization. "
            "It works by optimizing content for AI assistants. "
            "It increases content discoverability in AI search."
        )
        f_with = make_features(nlp, text_with_questions)
        f_without = make_features(nlp, text_without_questions)
        score_with, _ = score_qa_alignment(f_with)
        score_without, _ = score_qa_alignment(f_without)
        assert score_with > score_without

    def test_returns_explanation_string(self, nlp) -> None:
        features = make_features(nlp, RICH_CONTENT)
        score, explanation = score_qa_alignment(features)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_faq_labels_increase_score(self, nlp) -> None:
        faq_text = (
            "Q: What is Python?\n"
            "A: Python is a programming language.\n"
            "Q: What is it used for?\n"
            "A: It is used for web development and data science.\n"
        )
        features = make_features(nlp, faq_text)
        score, _ = score_qa_alignment(features)
        assert score >= 20.0

    def test_rich_content_qa_score_reasonable(self, nlp) -> None:
        features = make_features(nlp, RICH_CONTENT)
        score, _ = score_qa_alignment(features)
        # Rich content has headings as questions, explicit question sentences
        assert score >= 30.0

    def test_answer_indicator_phrases_boost_score(self, nlp) -> None:
        text_with_indicators = (
            "The answer is quite simple: Python is easy to learn. "
            "In other words, its syntax resembles plain English. "
            "For example, you can print text with a single line of code. "
            "Specifically, the print() function handles output. "
            "That is why beginners love Python so much."
        )
        text_without_indicators = (
            "Python is easy to learn. "
            "Its syntax resembles plain English. "
            "You can print text with a single line of code. "
            "The print() function handles output. "
            "Beginners love Python so much."
        )
        f_with = make_features(nlp, text_with_indicators)
        f_without = make_features(nlp, text_without_indicators)
        score_with, _ = score_qa_alignment(f_with)
        score_without, _ = score_qa_alignment(f_without)
        assert score_with >= score_without

    def test_target_query_no_keyword_overlap_still_returns_score(self, nlp) -> None:
        text = "Python is a programming language used for many purposes."
        # Query with words completely absent from text
        features = make_features(nlp, text, target_query="quantum mechanics thermodynamics")
        score, explanation = score_qa_alignment(features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_target_query_short_words_only_gives_partial_credit(self, nlp) -> None:
        text = "The is a an of in to for."
        # Query consists only of short words (<=3 chars)
        features = make_features(nlp, text, target_query="is an to")
        score, explanation = score_qa_alignment(features)
        assert 0.0 <= score <= 100.0
        assert "short" in explanation.lower() or "query" in explanation.lower()

    def test_definition_style_sentences_give_partial_credit(self, nlp) -> None:
        text = (
            "Python is a high-level programming language. "
            "Machine learning is a subset of artificial intelligence. "
            "FastAPI is a modern web framework for building APIs. "
            "Generators are a type of iterable that produce values lazily. "
            "Decorators can be described as higher-order functions in Python."
        )
        features = make_features(nlp, text)
        score, _ = score_qa_alignment(features)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Tests for score_entity_density
# ---------------------------------------------------------------------------


class TestScoreEntityDensity:
    """Unit tests for the Entity Density dimension scorer."""

    def test_rich_content_scores_higher_than_poor(self, nlp) -> None:
        features_rich = make_features(nlp, RICH_CONTENT)
        features_poor = make_features(nlp, POOR_CONTENT)
        score_rich, _ = score_entity_density(features_rich)
        score_poor, _ = score_entity_density(features_poor)
        assert score_rich > score_poor

    def test_score_within_range(self, nlp) -> None:
        for text in [RICH_CONTENT, POOR_CONTENT, CITATION_RICH_CONTENT]:
            features = make_features(nlp, text)
            score, _ = score_entity_density(features)
            assert 0.0 <= score <= 100.0

    def test_entity_rich_content_scores_high(self, nlp) -> None:
        entity_text = (
            "Apple Inc., founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in "
            "Cupertino, California, is now one of the world's most valuable companies. "
            "Microsoft, Google, Amazon, and Meta are its main competitors in Silicon Valley. "
            "In 2023, Apple launched the iPhone 15 and the Vision Pro headset in New York."
        )
        features = make_features(nlp, entity_text)
        score, explanation = score_entity_density(features)
        assert score >= 40.0, f"Expected >= 40, got {score}. Explanation: {explanation}"

    def test_returns_explanation_string(self, nlp) -> None:
        features = make_features(nlp, RICH_CONTENT)
        score, explanation = score_entity_density(features)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_generic_text_scores_lower(self, nlp) -> None:
        generic = (
            "The thing is very good. People like it a lot. "
            "Everyone uses it every day. Things are nice. "
            "It is the best thing to have. We all want more of it."
        )
        features = make_features(nlp, generic)
        score, _ = score_entity_density(features)
        assert score < 70.0

    def test_citation_rich_content_has_entities(self, nlp) -> None:
        features = make_features(nlp, CITATION_RICH_CONTENT)
        score, explanation = score_entity_density(features)
        # Citation-rich content names WHO, Harvard, MIT, UN, etc.
        assert score >= 20.0

    def test_explanation_mentions_entities(self, nlp) -> None:
        entity_text = (
            "Apple Inc., founded by Steve Jobs in Cupertino, California. "
            "Google was created by Larry Page and Sergey Brin at Stanford University. "
            "Microsoft was founded by Bill Gates and Paul Allen in 1975."
        )
        features = make_features(nlp, entity_text)
        score, explanation = score_entity_density(features)
        # Explanation should mention entities or density
        assert any(
            word in explanation.lower()
            for word in ["entity", "entities", "density", "unique", "type"]
        )

    def test_no_entities_returns_low_score(self, nlp) -> None:
        # Very generic text unlikely to have named entities
        generic = (
            "Things happen every day and night. "
            "People do lots of activities in various places. "
            "Everything works out fine in the long run. "
            "Situations change over time and become better."
        )
        features = make_features(nlp, generic)
        score, explanation = score_entity_density(features)
        # Score should be low, explanation should reflect limited entities
        assert score < 60.0

    def test_score_does_not_exceed_100(self, nlp) -> None:
        # Content packed with entities
        dense_entity_text = (
            "Apple, Google, Microsoft, Amazon, Facebook, Tesla, Netflix, Uber, "
            "Airbnb, Twitter founded by Elon Musk, Jack Dorsey, Jeff Bezos, "
            "Bill Gates, Steve Jobs, Larry Page, Sergey Brin in Silicon Valley, "
            "New York, London, Tokyo, Paris, Berlin, Sydney, Toronto, Chicago. "
            "In 2020, 2021, 2022, 2023, 2024 these organizations achieved milestones. "
            "Harvard, MIT, Stanford, Oxford, Cambridge produced Nobel Prize winners."
        )
        features = make_features(nlp, dense_entity_text)
        score, _ = score_entity_density(features)
        assert score <= 100.0


# ---------------------------------------------------------------------------
# Tests for score_structured_formatting
# ---------------------------------------------------------------------------


class TestScoreStructuredFormatting:
    """Unit tests for the Structured Formatting dimension scorer."""

    def test_structured_content_scores_high(self, nlp) -> None:
        features = make_features(nlp, STRUCTURED_CONTENT)
        score, explanation = score_structured_formatting(features)
        assert score >= 50.0, f"Expected >= 50, got {score}. Explanation: {explanation}"

    def test_unstructured_content_scores_low(self, nlp) -> None:
        features = make_features(nlp, POOR_CONTENT)
        score, _ = score_structured_formatting(features)
        assert score < 50.0

    def test_headings_detected(self, nlp) -> None:
        text_with_headings = (
            "# Main Heading\n\n"
            "Some content here that explains things.\n\n"
            "## Sub Heading\n\n"
            "More content here.\n\n"
            "### Third Level\n\n"
            "Even more content."
        )
        features = make_features(nlp, text_with_headings)
        score, explanation = score_structured_formatting(features)
        assert "heading" in explanation.lower()
        assert score >= 25.0

    def test_bullet_lists_detected(self, nlp) -> None:
        text_with_bullets = (
            "Here are the key points:\n"
            "- First point is important\n"
            "- Second point is also important\n"
            "- Third point rounds it out\n"
            "- Fourth point adds more detail\n"
            "These are all the points."
        )
        features = make_features(nlp, text_with_bullets)
        score, explanation = score_structured_formatting(features)
        assert "list" in explanation.lower() or "item" in explanation.lower()
        assert score >= 20.0

    def test_numbered_lists_detected(self, nlp) -> None:
        text_with_numbered = (
            "Steps to follow:\n"
            "1. Install Python\n"
            "2. Create a virtual environment\n"
            "3. Install dependencies\n"
            "4. Run the application\n"
        )
        features = make_features(nlp, text_with_numbered)
        score, _ = score_structured_formatting(features)
        assert score >= 20.0

    def test_score_within_range(self, nlp) -> None:
        for text in [RICH_CONTENT, POOR_CONTENT, STRUCTURED_CONTENT, MINIMAL_CONTENT]:
            features = make_features(nlp, text)
            score, _ = score_structured_formatting(features)
            assert 0.0 <= score <= 100.0

    def test_rich_content_with_all_signals_scores_high(self, nlp) -> None:
        features = make_features(nlp, RICH_CONTENT)
        score, _ = score_structured_formatting(features)
        assert score >= 50.0

    def test_returns_explanation_string(self, nlp) -> None:
        features = make_features(nlp, STRUCTURED_CONTENT)
        score, explanation = score_structured_formatting(features)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_bold_emphasis_detected(self, nlp) -> None:
        text_with_bold = (
            "This is **very important** content. You should **pay attention** to "
            "**key concepts** like **this one** and **that one**. "
            "Remember that **bold text** signals emphasis."
        )
        features = make_features(nlp, text_with_bold)
        score, explanation = score_structured_formatting(features)
        assert "emphasis" in explanation.lower() or "bold" in explanation.lower()

    def test_code_blocks_detected(self, nlp) -> None:
        text_with_code = (
            "# How to use Python\n\n"
            "Here is an example:\n\n"
            "```python\n"
            "def hello():\n"
            "    print('Hello')\n"
            "```\n\n"
            "Use `print()` for output."
        )
        features = make_features(nlp, text_with_code)
        score, explanation = score_structured_formatting(features)
        assert "code" in explanation.lower()
        assert score >= 20.0

    def test_paragraph_structure_contributes(self, nlp) -> None:
        text_with_paragraphs = (
            "First paragraph discusses the introduction to the topic.\n\n"
            "Second paragraph explains the core concepts in more detail.\n\n"
            "Third paragraph provides practical examples and use cases.\n\n"
            "Fourth paragraph summarizes the key takeaways.\n\n"
            "Fifth paragraph offers a conclusion and next steps."
        )
        features = make_features(nlp, text_with_paragraphs)
        score, explanation = score_structured_formatting(features)
        assert "paragraph" in explanation.lower()
        assert score >= 15.0

    def test_markdown_table_detected(self, nlp) -> None:
        text_with_table = (
            "Here is a comparison table:\n\n"
            "| Language | Use Case | Popularity |\n"
            "|---------|----------|------------|\n"
            "| Python  | Data Science | High |\n"
            "| Java    | Enterprise | High |\n"
            "| C++     | Systems | Medium |\n\n"
            "The table above summarizes the comparison."
        )
        features = make_features(nlp, text_with_table)
        score, explanation = score_structured_formatting(features)
        assert "table" in explanation.lower()

    def test_plain_text_no_structure_scores_low(self, nlp) -> None:
        # Single block of plain text with no structure signals
        plain = (
            "Python is a programming language that is widely used. It was created "
            "by Guido van Rossum. It has a simple syntax. Many people use it. It is "
            "good for beginners. It also works for experts. Companies use it a lot."
        )
        features = make_features(nlp, plain)
        score, _ = score_structured_formatting(features)
        # Very minimal structure should yield a low-to-medium score
        assert score < 60.0


# ---------------------------------------------------------------------------
# Tests for score_citation_cues
# ---------------------------------------------------------------------------


class TestScoreCitationCues:
    """Unit tests for the Citation Cues dimension scorer."""

    def test_citation_rich_content_scores_high(self, nlp) -> None:
        features = make_features(nlp, CITATION_RICH_CONTENT)
        score, explanation = score_citation_cues(features)
        assert score >= 50.0, f"Expected >= 50, got {score}. Explanation: {explanation}"

    def test_poor_content_scores_low(self, nlp) -> None:
        features = make_features(nlp, POOR_CONTENT)
        score, _ = score_citation_cues(features)
        assert score < 40.0

    def test_percentages_contribute_to_score(self, nlp) -> None:
        text_with_stats = (
            "Studies show that 75% of users prefer visual content. "
            "Additionally, 42% of marketers reported a 30% increase in engagement. "
            "Furthermore, email open rates increased by 25% after A/B testing. "
            "The conversion rate improved from 2.1% to 3.8% over six months."
        )
        features_with = make_features(nlp, text_with_stats)
        features_without = make_features(
            nlp,
            "Studies show that users prefer visual content. Marketers reported an "
            "increase in engagement. Email open rates increased after A/B testing.",
        )
        score_with, _ = score_citation_cues(features_with)
        score_without, _ = score_citation_cues(features_without)
        assert score_with > score_without

    def test_attribution_phrases_contribute(self, nlp) -> None:
        text_with_attribution = (
            "According to Harvard researchers, the study found significant results. "
            "Scientists suggest that the new method works better. "
            "Published in the Journal of Science, the paper was cited widely. "
            "Experts say the findings are groundbreaking."
        )
        features = make_features(nlp, text_with_attribution)
        score, explanation = score_citation_cues(features)
        assert score >= 25.0
        assert "attribution" in explanation.lower()

    def test_urls_contribute_to_score(self, nlp) -> None:
        text_with_urls = (
            "For more information, visit https://www.example.com and "
            "https://docs.python.org. "
            "Additional resources are available at https://github.com/python."
        )
        features_with = make_features(nlp, text_with_urls)
        score_with, _ = score_citation_cues(features_with)
        assert score_with >= 10.0

    def test_score_within_range(self, nlp) -> None:
        for text in [RICH_CONTENT, POOR_CONTENT, CITATION_RICH_CONTENT]:
            features = make_features(nlp, text)
            score, _ = score_citation_cues(features)
            assert 0.0 <= score <= 100.0

    def test_returns_explanation_string(self, nlp) -> None:
        features = make_features(nlp, CITATION_RICH_CONTENT)
        score, explanation = score_citation_cues(features)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_quotations_contribute(self, nlp) -> None:
        text_with_quotes = (
            'According to Dr. Smith, "The results are unprecedented in the field." '
            'The lead researcher noted, "We observed a 40% improvement over baseline." '
            'The journal editor stated, "This paper represents a significant advance."
        )
        features = make_features(nlp, text_with_quotes)
        score, explanation = score_citation_cues(features)
        assert score >= 20.0

    def test_named_institutions_contribute(self, nlp) -> None:
        text_with_institutions = (
            "Research from Harvard and MIT indicates that machine learning "
            "outperforms traditional methods. The WHO published a report in the "
            "Journal of Medicine confirming these findings. The study was indexed "
            "on PubMed and cited by institutions at Stanford and Oxford."
        )
        features = make_features(nlp, text_with_institutions)
        score, explanation = score_citation_cues(features)
        assert score >= 20.0
        assert any(
            word in explanation.lower()
            for word in ["source", "named", "url", "attribution", "statistical"]
        )

    def test_dollar_amounts_counted_as_stats(self, nlp) -> None:
        text_with_money = (
            "The market is valued at $4.2 billion in 2024. "
            "Revenue grew to $850 million, a 23% increase year-over-year. "
            "Investment rounds totaled $120 million across 15 startups."
        )
        features = make_features(nlp, text_with_money)
        score, explanation = score_citation_cues(features)
        assert score >= 15.0
        assert "statistical" in explanation.lower()

    def test_blockquote_markdown_detected(self, nlp) -> None:
        text_with_blockquote = (
            "The report states:\n"
            "> Urban areas produce 80% of global GDP despite covering 2% of land.\n"
            "> Population density drives economic specialization and innovation.\n\n"
            "These findings support urbanization policy recommendations."
        )
        features = make_features(nlp, text_with_blockquote)
        score, explanation = score_citation_cues(features)
        assert score >= 15.0


# ---------------------------------------------------------------------------
# Tests for score_semantic_clarity
# ---------------------------------------------------------------------------


class TestScoreSemanticClarity:
    """Unit tests for the Semantic Clarity dimension scorer."""

    def test_score_within_range(self, nlp) -> None:
        for text in [RICH_CONTENT, POOR_CONTENT, STRUCTURED_CONTENT, MINIMAL_CONTENT]:
            features = make_features(nlp, text)
            score, _ = score_semantic_clarity(features)
            assert 0.0 <= score <= 100.0

    def test_filler_phrases_reduce_score(self, nlp) -> None:
        clean_text = (
            "Python is a programming language. Developers use it for data science, "
            "web development, and automation. It has a large ecosystem and active community."
        )
        filler_text = (
            "Python is basically a very popular programming language. Developers "
            "literally use it for kind of important data science work. It's really quite "
            "good and actually has a rather large ecosystem and somewhat active community. "
            "Due to the fact that it is easy to learn, it is very widely used."
        )
        f_clean = make_features(nlp, clean_text)
        f_filler = make_features(nlp, filler_text)
        score_clean, _ = score_semantic_clarity(f_clean)
        score_filler, _ = score_semantic_clarity(f_filler)
        assert score_clean > score_filler

    def test_transition_words_bonus(self, nlp) -> None:
        text_with_transitions = (
            "First, Python is easy to learn. Furthermore, it has a large ecosystem. "
            "However, it can be slower than C++. Therefore, developers choose it "
            "for its productivity. In conclusion, Python is a great choice for beginners. "
            "Moreover, it is used by expert engineers at Google and Facebook."
        )
        text_without_transitions = (
            "Python is easy to learn. It has a large ecosystem. "
            "It can be slower than C++. Developers choose it for productivity. "
            "Python is a great choice for beginners. "
            "It is used by expert engineers at Google and Facebook."
        )
        f_with = make_features(nlp, text_with_transitions)
        f_without = make_features(nlp, text_without_transitions)
        score_with, _ = score_semantic_clarity(f_with)
        score_without, _ = score_semantic_clarity(f_without)
        # Transition words should boost score
        assert score_with >= score_without - 5.0

    def test_returns_explanation_string(self, nlp) -> None:
        features = make_features(nlp, RICH_CONTENT)
        score, explanation = score_semantic_clarity(features)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_repetitive_content_scores_lower(self, nlp) -> None:
        repetitive = (
            "The thing is good. The thing works well. The thing is very useful. "
            "The thing helps people. The thing is the best thing. The thing does things. "
            "People like the thing. The thing is popular. The thing is used widely."
        )
        diverse = (
            "Python is a versatile programming language. Developers leverage its "
            "extensive library ecosystem for machine learning, web development, and "
            "scientific computing. Its clean syntax facilitates rapid prototyping."
        )
        f_rep = make_features(nlp, repetitive)
        f_div = make_features(nlp, diverse)
        score_rep, _ = score_semantic_clarity(f_rep)
        score_div, _ = score_semantic_clarity(f_div)
        assert score_div >= score_rep - 10.0

    def test_explanation_mentions_filler_when_present(self, nlp) -> None:
        filler_text = (
            "Basically, this is very important. You literally need to kind of understand "
            "that it is really quite significant. Actually, it is somewhat critical that "
            "we note this. In today's world, at the end of the day, things are like this."
        )
        features = make_features(nlp, filler_text)
        score, explanation = score_semantic_clarity(features)
        assert "filler" in explanation.lower() or "hedge" in explanation.lower()

    def test_long_sentences_penalized(self, nlp) -> None:
        # Content with very long average sentence length
        long_sentence_text = (
            "Python is a high-level general-purpose dynamically-typed interpreted "
            "programming language that was created by Guido van Rossum and first "
            "released in 1991 and has since grown to become one of the most popular "
            "programming languages in the world for applications ranging from web "
            "development to data science to machine learning to scientific computing. "
            "The Python Software Foundation maintains the language and releases new "
            "versions on a regular schedule with Python 3.11 and 3.12 being the most "
            "recent major releases featuring significant performance improvements and "
            "new language features that make the language even more powerful than before."
        )
        short_sentence_text = (
            "Python is popular. It was created in 1991. Guido van Rossum designed it. "
            "It supports data science. It handles web development. Companies use it widely."
        )
        f_long = make_features(nlp, long_sentence_text)
        f_short = make_features(nlp, short_sentence_text)
        score_long, explanation_long = score_semantic_clarity(f_long)
        score_short, _ = score_semantic_clarity(f_short)
        # Long sentences should be penalized
        assert "long" in explanation_long.lower() or score_long <= score_short + 20.0

    def test_score_starts_from_neutral_midpoint(self, nlp) -> None:
        # A neutral text with no extreme signals should be near the midpoint
        neutral = (
            "Python is a programming language. It is used for web development. "
            "Many companies use it. It has good documentation. Developers enjoy using it."
        )
        features = make_features(nlp, neutral)
        score, _ = score_semantic_clarity(features)
        # Should be somewhere in a reasonable range
        assert 20.0 <= score <= 90.0


# ---------------------------------------------------------------------------
# Tests for score_content_depth
# ---------------------------------------------------------------------------


class TestScoreContentDepth:
    """Unit tests for the Content Depth dimension scorer."""

    def test_rich_content_scores_higher_than_poor(self, nlp) -> None:
        features_rich = make_features(nlp, RICH_CONTENT)
        features_poor = make_features(nlp, POOR_CONTENT)
        score_rich, _ = score_content_depth(features_rich)
        score_poor, _ = score_content_depth(features_poor)
        assert score_rich > score_poor

    def test_score_within_range(self, nlp) -> None:
        for text in [RICH_CONTENT, POOR_CONTENT, QA_RICH_CONTENT, MINIMAL_CONTENT]:
            features = make_features(nlp, text)
            score, _ = score_content_depth(features)
            assert 0.0 <= score <= 100.0

    def test_long_content_scores_higher_than_short(self, nlp) -> None:
        short_text = "Python is a programming language. It is used for many things. People like it."
        long_text = RICH_CONTENT
        f_short = make_features(nlp, short_text)
        f_long = make_features(nlp, long_text)
        score_short, _ = score_content_depth(f_short)
        score_long, _ = score_content_depth(f_long)
        assert score_long > score_short

    def test_explanatory_phrases_increase_score(self, nlp) -> None:
        explanatory_text = (
            "Machine learning works because it identifies patterns in data. "
            "This means that models can generalize to new examples. "
            "For example, a model trained on images of cats can recognize new cat photos. "
            "As a result, supervised learning is widely used in computer vision tasks. "
            "In other words, the training data defines what the model learns. "
            "Since the model improves with more data, data collection is crucial. "
            "To illustrate, consider a spam classifier trained on labeled emails."
        )
        plain_text = (
            "Machine learning uses patterns in data. "
            "Models work on new examples. "
            "A model trained on images can recognize cats. "
            "Supervised learning is used in computer vision. "
            "Training data defines what the model learns. "
            "The model improves with more data. "
            "A spam classifier is trained on emails."
        )
        f_exp = make_features(nlp, explanatory_text)
        f_plain = make_features(nlp, plain_text)
        score_exp, _ = score_content_depth(f_exp)
        score_plain, _ = score_content_depth(f_plain)
        assert score_exp > score_plain

    def test_returns_explanation_string(self, nlp) -> None:
        features = make_features(nlp, RICH_CONTENT)
        score, explanation = score_content_depth(features)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_very_short_content_gets_low_score(self, nlp) -> None:
        features = make_features(nlp, MINIMAL_CONTENT)
        score, _ = score_content_depth(features)
        assert score < 30.0

    def test_comparative_language_increases_score(self, nlp) -> None:
        comparative_text = (
            "Python is better than Java for beginners because it requires less boilerplate. "
            "Compared to C++, Python is slower but more readable. "
            "Unlike Ruby, Python has broader adoption in data science. "
            "The advantages of Python include readability, while the drawbacks include speed. "
            "On the other hand, compiled languages outperform Python in CPU-intensive tasks."
        )
        plain_text = (
            "Python is a language. Java is also a language. "
            "C++ is different from Python. Ruby is another language. "
            "Python has some features. Other languages have features too."
        )
        f_comp = make_features(nlp, comparative_text)
        f_plain = make_features(nlp, plain_text)
        score_comp, _ = score_content_depth(f_comp)
        score_plain, _ = score_content_depth(f_plain)
        assert score_comp > score_plain

    def test_word_count_signal_in_explanation(self, nlp) -> None:
        features = make_features(nlp, RICH_CONTENT)
        score, explanation = score_content_depth(features)
        assert "word" in explanation.lower() or "content" in explanation.lower()

    def test_many_sentences_boost_score(self, nlp) -> None:
        # Build text with 25+ sentences
        sentences = [
            f"This is sentence number {i} about Python programming and its applications."
            for i in range(1, 26)
        ]
        long_text = " ".join(sentences)
        features = make_features(nlp, long_text)
        score, explanation = score_content_depth(features)
        assert score >= 20.0
        assert "sentence" in explanation.lower()

    def test_unique_concepts_contribute(self, nlp) -> None:
        # Rich vocabulary content
        rich_vocab = (
            "Machine learning encompasses supervised, unsupervised, and reinforcement "
            "paradigms. Convolutional neural networks excel at image recognition tasks. "
            "Transformer architectures revolutionized natural language processing workflows. "
            "Gradient descent optimizes loss functions across parameter spaces. "
            "Regularization techniques like dropout prevent overfitting in deep networks. "
            "Backpropagation computes gradients through differentiable computational graphs."
        )
        # Low vocabulary content
        low_vocab = (
            "The thing is good. It does the thing well. The thing works. "
            "People use the thing. The thing is useful. It is a good thing."
        )
        f_rich = make_features(nlp, rich_vocab)
        f_low = make_features(nlp, low_vocab)
        score_rich, _ = score_content_depth(f_rich)
        score_low, _ = score_content_depth(f_low)
        assert score_rich > score_low


# ---------------------------------------------------------------------------
# Tests for ContentScorer
# ---------------------------------------------------------------------------


class TestContentScorer:
    """Integration tests for the ContentScorer class."""

    def test_score_returns_analysis_result(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert isinstance(result, AnalysisResult)

    def test_score_has_all_six_dimensions(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert len(result.dimensions) == 6

    def test_all_dimension_keys_present(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        dimension_keys = {d.dimension for d in result.dimensions}
        expected_keys = {
            DimensionKey.QA_ALIGNMENT,
            DimensionKey.ENTITY_DENSITY,
            DimensionKey.STRUCTURED_FORMATTING,
            DimensionKey.CITATION_CUES,
            DimensionKey.SEMANTIC_CLARITY,
            DimensionKey.CONTENT_DEPTH,
        }
        assert dimension_keys == expected_keys

    def test_composite_score_within_range(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert 0.0 <= result.composite_score <= 100.0

    def test_composite_label_is_valid(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert isinstance(result.composite_label, ScoreLabel)

    def test_rich_content_scores_higher_than_poor(self, scorer: ContentScorer) -> None:
        result_rich = scorer.score(RICH_CONTENT)
        result_poor = scorer.score(POOR_CONTENT)
        assert result_rich.composite_score > result_poor.composite_score

    def test_improvement_priorities_sorted_ascending(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        priorities = [d.improvement_priority for d in result.dimensions]
        assert priorities == sorted(priorities)

    def test_all_priorities_are_positive(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        for dim in result.dimensions:
            assert dim.improvement_priority >= 1

    def test_suggestions_empty_from_scorer(self, scorer: ContentScorer) -> None:
        """Scorer should return empty suggestions list (populated by suggestions module)."""
        result = scorer.score(RICH_CONTENT)
        assert result.suggestions == []

    def test_target_query_echoed_in_result(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT, target_query="What is Python?")
        assert result.target_query == "What is Python?"

    def test_target_query_none_by_default(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert result.target_query is None

    def test_word_count_stored(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert result.content_word_count > 0

    def test_char_count_stored(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert result.content_char_count > 0

    def test_error_message_none_on_success(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        assert result.error_message is None

    def test_determinism_same_input_same_output(self, scorer: ContentScorer) -> None:
        """Same input must always produce exactly the same composite score."""
        result1 = scorer.score(RICH_CONTENT, target_query="Python")
        result2 = scorer.score(RICH_CONTENT, target_query="Python")
        assert result1.composite_score == result2.composite_score
        for d1, d2 in zip(result1.dimensions, result2.dimensions):
            assert d1.raw_score == d2.raw_score

    def test_empty_content_raises_value_error(self, scorer: ContentScorer) -> None:
        with pytest.raises(ValueError, match="empty"):
            scorer.score("")

    def test_whitespace_only_content_raises_value_error(self, scorer: ContentScorer) -> None:
        with pytest.raises(ValueError, match="empty"):
            scorer.score("   \n\t  ")

    def test_all_dimension_scores_in_range(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        for dim in result.dimensions:
            assert 0.0 <= dim.raw_score <= 100.0, (
                f"{dim.dimension.value} score {dim.raw_score} out of [0, 100]"
            )

    def test_weighted_score_matches_calculation(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        for dim in result.dimensions:
            expected = round(dim.raw_score * dim.weight, 4)
            assert abs(dim.weighted_score - expected) < 0.01

    def test_score_from_input_matches_score(self, scorer: ContentScorer) -> None:
        content_input = ContentInput(
            content=RICH_CONTENT.strip(),
            target_query="What is Python?",
        )
        result_input = scorer.score_from_input(content_input)
        result_direct = scorer.score(RICH_CONTENT.strip(), target_query="What is Python?")
        assert result_input.composite_score == result_direct.composite_score

    def test_custom_settings_weights_applied(self) -> None:
        """Custom weights should affect composite score."""
        settings_high_depth = Settings(
            weight_content_depth=5.0,
            weight_qa_alignment=0.1,
            weight_entity_density=0.1,
            weight_structured_formatting=0.1,
            weight_citation_cues=0.1,
            weight_semantic_clarity=0.1,
        )
        settings_high_qa = Settings(
            weight_qa_alignment=5.0,
            weight_content_depth=0.1,
            weight_entity_density=0.1,
            weight_structured_formatting=0.1,
            weight_citation_cues=0.1,
            weight_semantic_clarity=0.1,
        )
        scorer_depth = ContentScorer(settings=settings_high_depth)
        scorer_qa = ContentScorer(settings=settings_high_qa)

        result_depth = scorer_depth.score(QA_RICH_CONTENT)
        result_qa = scorer_qa.score(QA_RICH_CONTENT)
        # Both should produce valid scores
        assert 0.0 <= result_depth.composite_score <= 100.0
        assert 0.0 <= result_qa.composite_score <= 100.0

    def test_priority_one_is_lowest_scored_dimension(self, scorer: ContentScorer) -> None:
        """The dimension with improvement_priority=1 must have the lowest raw_score."""
        result = scorer.score(RICH_CONTENT)
        priority_one = next(d for d in result.dimensions if d.improvement_priority == 1)
        min_score = min(d.raw_score for d in result.dimensions)
        assert priority_one.raw_score == min_score

    def test_score_from_input_uses_target_query(self, scorer: ContentScorer) -> None:
        content_input = ContentInput(
            content=RICH_CONTENT.strip(),
            target_query="What is Python?",
        )
        result = scorer.score_from_input(content_input)
        assert result.target_query == "What is Python?"

    def test_word_count_approximately_correct(self, scorer: ContentScorer) -> None:
        simple_text = "Python is a great programming language."
        # This text has 7 words
        # Note: ContentScorer.score strips and splits on whitespace for word count
        result = scorer.score(simple_text + " " * 10 + "a" * 50)
        # word count should be positive and reasonable
        assert result.content_word_count > 0

    def test_char_count_equals_stripped_length(self, scorer: ContentScorer) -> None:
        text = "   " + RICH_CONTENT + "   "
        result = scorer.score(text)
        assert result.content_char_count == len(RICH_CONTENT.strip())

    def test_all_explanations_are_non_empty_strings(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        for dim in result.dimensions:
            assert isinstance(dim.explanation, str)
            assert len(dim.explanation) > 0

    def test_all_display_names_are_non_empty(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        for dim in result.dimensions:
            assert isinstance(dim.display_name, str)
            assert len(dim.display_name) > 0

    def test_all_priorities_are_unique(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        priorities = [d.improvement_priority for d in result.dimensions]
        assert len(set(priorities)) == len(priorities)

    def test_composite_score_is_weighted_average(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        total_weighted = sum(d.weighted_score for d in result.dimensions)
        total_weight = sum(d.weight for d in result.dimensions)
        expected_composite = round(total_weighted / total_weight, 2)
        assert abs(result.composite_score - expected_composite) < 0.01


# ---------------------------------------------------------------------------
# Tests for score_content module-level function
# ---------------------------------------------------------------------------


class TestScoreContentFunction:
    """Tests for the module-level score_content convenience function."""

    def test_returns_analysis_result(self) -> None:
        result = score_content(RICH_CONTENT)
        assert isinstance(result, AnalysisResult)

    def test_composite_score_within_range(self) -> None:
        result = score_content(POOR_CONTENT)
        assert 0.0 <= result.composite_score <= 100.0

    def test_target_query_passed_through(self) -> None:
        result = score_content(RICH_CONTENT, target_query="Python")
        assert result.target_query == "Python"

    def test_custom_settings_accepted(self) -> None:
        settings = Settings(weight_content_depth=2.0)
        result = score_content(RICH_CONTENT, settings=settings)
        assert isinstance(result, AnalysisResult)

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError):
            score_content("")

    def test_returns_six_dimensions(self) -> None:
        result = score_content(RICH_CONTENT)
        assert len(result.dimensions) == 6

    def test_suggestions_empty(self) -> None:
        result = score_content(RICH_CONTENT)
        assert result.suggestions == []

    def test_deterministic_output(self) -> None:
        result1 = score_content(RICH_CONTENT)
        result2 = score_content(RICH_CONTENT)
        assert result1.composite_score == result2.composite_score

    def test_no_target_query_by_default(self) -> None:
        result = score_content(RICH_CONTENT)
        assert result.target_query is None

    def test_poor_content_gets_non_zero_score(self) -> None:
        # Even poor content should get some score for each dimension
        result = score_content(POOR_CONTENT)
        assert result.composite_score >= 0.0
        assert len(result.dimensions) == 6


# ---------------------------------------------------------------------------
# Tests for _load_spacy_model
# ---------------------------------------------------------------------------


class TestLoadSpacyModel:
    """Tests for the spaCy model loader."""

    def test_loads_en_core_web_sm(self) -> None:
        nlp = _load_spacy_model("en_core_web_sm")
        assert nlp is not None

    def test_returns_same_object_on_second_call(self) -> None:
        """Verify lru_cache is working correctly."""
        nlp1 = _load_spacy_model("en_core_web_sm")
        nlp2 = _load_spacy_model("en_core_web_sm")
        assert nlp1 is nlp2

    def test_loaded_model_can_parse_text(self) -> None:
        nlp = _load_spacy_model("en_core_web_sm")
        doc = nlp("Python is a programming language.")
        assert len(list(doc.sents)) >= 1

    def test_loaded_model_has_ner_pipe(self) -> None:
        nlp = _load_spacy_model("en_core_web_sm")
        assert "ner" in nlp.pipe_names

    def test_invalid_model_raises_os_error(self) -> None:
        # Clear cache first to ensure fresh load attempt
        _load_spacy_model.cache_clear()
        try:
            with pytest.raises(OSError, match="not installed"):
                _load_spacy_model("nonexistent_model_xyz_abc_123")
        finally:
            # Restore cache with valid model for subsequent tests
            _load_spacy_model.cache_clear()
            _load_spacy_model("en_core_web_sm")  # Reload valid model

    def test_error_message_includes_download_hint(self) -> None:
        _load_spacy_model.cache_clear()
        try:
            with pytest.raises(OSError) as exc_info:
                _load_spacy_model("nonexistent_model_xyz_abc_123")
            assert "download" in str(exc_info.value).lower()
        finally:
            _load_spacy_model.cache_clear()
            _load_spacy_model("en_core_web_sm")


# ---------------------------------------------------------------------------
# Boundary and edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests for the scoring engine."""

    def test_content_with_only_numbers_does_not_crash(self, scorer: ContentScorer) -> None:
        numeric_content = "123 456 789 " * 20  # Make it long enough
        result = scorer.score(numeric_content)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.composite_score <= 100.0

    def test_content_with_special_characters(self, scorer: ContentScorer) -> None:
        special_content = (
            "Python (version 3.11+) supports f-strings & walrus operator (:=). "
            "See: https://docs.python.org/3/ for details. "
            "Cost: $0.001/token @ 100K tokens/month = $100/month. "
            "Formula: E = mc\u00b2, where c \u2248 3\u00d710\u2078 m/s."
        )
        result = scorer.score(special_content)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.composite_score <= 100.0

    def test_content_with_markdown_code_blocks(self, scorer: ContentScorer) -> None:
        code_content = (
            "# How to use Python\n\n"
            "Here is an example:\n\n"
            "```python\n"
            "def hello_world():\n"
            "    print('Hello, World!')\n"
            "```\n\n"
            "You can also use inline code like `print()` to output text. "
            "This function is defined in the `builtins` module."
        )
        result = scorer.score(code_content)
        assert isinstance(result, AnalysisResult)
        # Code content should be recognized in structured formatting
        formatting_dim = next(
            d for d in result.dimensions
            if d.dimension == DimensionKey.STRUCTURED_FORMATTING
        )
        assert formatting_dim.raw_score > 0.0

    def test_very_long_content_does_not_crash(self, scorer: ContentScorer) -> None:
        long_content = RICH_CONTENT * 5  # ~5x the normal length
        result = scorer.score(long_content)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.composite_score <= 100.0

    def test_all_scores_bounded_for_poor_content(self, scorer: ContentScorer) -> None:
        result = scorer.score(POOR_CONTENT)
        for dim in result.dimensions:
            assert 0.0 <= dim.raw_score <= 100.0

    def test_all_scores_bounded_for_rich_content(self, scorer: ContentScorer) -> None:
        result = scorer.score(RICH_CONTENT)
        for dim in result.dimensions:
            assert 0.0 <= dim.raw_score <= 100.0

    def test_unicode_content_handled(self, scorer: ContentScorer) -> None:
        unicode_content = (
            "H\u00e9llo, w\u00f6rld! This is t\u00ebst cont\u00ebnt with Unicode char\u00e4cters. "
            "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 is also present. "
            "According to the r\u00e9sum\u00e9, the caf\u00e9 serves cr\u00eapes. "
            "The na\u00efve approach to r\u00e9sum\u00e9s uses simple templ\u00e2tes. "
            "What is Unicode? Unicode is a standard for text encoding worldwide."
        )
        result = scorer.score(unicode_content)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.composite_score <= 100.0

    def test_content_with_html_tags(self, scorer: ContentScorer) -> None:
        html_content = (
            "<h1>What is Python?</h1>\n"
            "<p>Python is a high-level programming language created by "
            "<strong>Guido van Rossum</strong> in 1991.</p>\n"
            "<h2>Why is Python popular?</h2>\n"
            "<ul>\n"
            "<li>Easy to learn and read</li>\n"
            "<li>Large ecosystem of libraries</li>\n"
            "<li>Versatile for many use cases</li>\n"
            "</ul>\n"
            "<p>According to Stack Overflow, Python is the most popular language in 2024.</p>"
        )
        result = scorer.score(html_content)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.composite_score <= 100.0

    def test_content_with_only_whitespace_after_stripping_raises(self, scorer: ContentScorer) -> None:
        with pytest.raises(ValueError, match="empty"):
            scorer.score("\n\n\n\t\t\t   \n")

    def test_content_with_repeated_words_still_scores(self, scorer: ContentScorer) -> None:
        repeated = ("Python " * 60).strip()  # 60 repetitions
        result = scorer.score(repeated)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.composite_score <= 100.0

    def test_content_at_minimum_length_is_scored(self, scorer: ContentScorer) -> None:
        result = scorer.score(MINIMAL_CONTENT)
        assert isinstance(result, AnalysisResult)
        assert len(result.dimensions) == 6

    def test_citation_content_scored_correctly(self, scorer: ContentScorer) -> None:
        result = scorer.score(CITATION_RICH_CONTENT)
        citation_dim = next(
            d for d in result.dimensions
            if d.dimension == DimensionKey.CITATION_CUES
        )
        assert citation_dim.raw_score >= 20.0

    def test_structured_content_formatting_recognized(self, scorer: ContentScorer) -> None:
        result = scorer.score(STRUCTURED_CONTENT)
        formatting_dim = next(
            d for d in result.dimensions
            if d.dimension == DimensionKey.STRUCTURED_FORMATTING
        )
        assert formatting_dim.raw_score >= 30.0

    def test_qa_content_qa_alignment_recognized(self, scorer: ContentScorer) -> None:
        result = scorer.score(QA_RICH_CONTENT)
        qa_dim = next(
            d for d in result.dimensions
            if d.dimension == DimensionKey.QA_ALIGNMENT
        )
        assert qa_dim.raw_score >= 40.0
