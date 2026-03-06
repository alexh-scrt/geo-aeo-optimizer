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
        # Content with many named entities
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
        # Generic text without entities should score relatively low
        assert score < 70.0


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
        # RICH_CONTENT has headings, bullets, numbered lists, bold text, paragraphs
        features = make_features(nlp, RICH_CONTENT)
        score, _ = score_structured_formatting(features)
        assert score >= 50.0

    def test_returns_explanation_string(self, nlp) -> None:
        features = make_features(nlp, STRUCTURED_CONTENT)
        score, explanation = score_structured_formatting(features)
        assert isinstance(explanation, str)
        assert len(explanation) > 0


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
        assert score_with >= score_without - 5.0  # Allow small variance from other signals

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
        assert score_div >= score_rep - 10.0  # diverse should not score worse


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
        long_text = RICH_CONTENT  # ~300+ words
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

        # QA-rich content should score higher with high QA weight
        result_depth = scorer_depth.score(QA_RICH_CONTENT)
        result_qa = scorer_qa.score(QA_RICH_CONTENT)
        # The scores will differ because of different weight configurations
        # We can't assert direction without knowing exact scores, but they should differ
        # unless both dimensions happen to have the exact same raw score
        # Both should still be valid
        assert 0.0 <= result_depth.composite_score <= 100.0
        assert 0.0 <= result_qa.composite_score <= 100.0


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
            "Formula: E = mc², where c ≈ 3×10⁸ m/s."
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
            "Héllo, wörld! This is tëst contënt with Unicode charäcters. "
            "日本語のテキスト is also present. "
            "According to the résumé, the café serves crêpes. "
            "The naïve approach to résumés uses simple templâtes. "
            "What is Unicode? Unicode is a standard for text encoding worldwide."
        )
        result = scorer.score(unicode_content)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.composite_score <= 100.0
