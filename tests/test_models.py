"""Unit tests for Pydantic model validation and edge case handling.

Tests cover:
- Settings loading and weight configuration
- ContentInput validation (stripping, length limits, type errors)
- ScoreLabel derivation from numeric scores
- DimensionScore creation and field derivation
- SuggestionItem construction
- AnalysisResult.build composite score calculation and priority ranking
- Edge cases: empty content, boundary scores, weight variations
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geo_aeo_optimizer.models import (
    AnalysisResult,
    ContentInput,
    DimensionKey,
    DimensionScore,
    ScoreLabel,
    Settings,
    SuggestionItem,
    get_dimension_display_name,
    get_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIN_CONTENT = "a" * 50  # exactly the minimum length


def make_dimension(dimension: DimensionKey, raw_score: float, weight: float = 1.0) -> DimensionScore:
    """Helper to create a DimensionScore with minimal boilerplate."""
    from geo_aeo_optimizer.models import get_dimension_display_name

    return DimensionScore.create(
        dimension=dimension,
        display_name=get_dimension_display_name(dimension),
        raw_score=raw_score,
        weight=weight,
        explanation=f"Test explanation for {dimension.value}",
    )


def make_all_dimensions(score: float = 75.0) -> list[DimensionScore]:
    """Return a full set of six DimensionScore objects all set to ``score``."""
    return [
        make_dimension(DimensionKey.QA_ALIGNMENT, score, 1.5),
        make_dimension(DimensionKey.ENTITY_DENSITY, score, 1.0),
        make_dimension(DimensionKey.STRUCTURED_FORMATTING, score, 1.0),
        make_dimension(DimensionKey.CITATION_CUES, score, 1.2),
        make_dimension(DimensionKey.SEMANTIC_CLARITY, score, 1.0),
        make_dimension(DimensionKey.CONTENT_DEPTH, score, 1.3),
    ]


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


class TestSettings:
    """Tests for the Settings configuration model."""

    def test_default_weights_are_positive(self) -> None:
        settings = Settings()
        for key, weight in settings.weights_by_dimension.items():
            assert weight > 0, f"Weight for {key!r} must be positive"

    def test_total_weight_equals_sum_of_individual_weights(self) -> None:
        settings = Settings()
        expected = (
            settings.weight_qa_alignment
            + settings.weight_entity_density
            + settings.weight_structured_formatting
            + settings.weight_citation_cues
            + settings.weight_semantic_clarity
            + settings.weight_content_depth
        )
        assert settings.total_weight == pytest.approx(expected)

    def test_weights_by_dimension_has_all_six_keys(self) -> None:
        settings = Settings()
        expected_keys = {
            "qa_alignment",
            "entity_density",
            "structured_formatting",
            "citation_cues",
            "semantic_clarity",
            "content_depth",
        }
        assert set(settings.weights_by_dimension.keys()) == expected_keys

    def test_custom_weights_via_constructor(self) -> None:
        settings = Settings(
            weight_qa_alignment=2.0,
            weight_entity_density=0.5,
            weight_structured_formatting=0.5,
            weight_citation_cues=1.0,
            weight_semantic_clarity=1.0,
            weight_content_depth=1.0,
        )
        assert settings.weight_qa_alignment == 2.0
        assert settings.weight_entity_density == 0.5

    def test_zero_weight_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            Settings(weight_qa_alignment=0.0)

    def test_negative_weight_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            Settings(weight_entity_density=-1.0)

    def test_default_openai_model(self) -> None:
        settings = Settings()
        assert settings.openai_model == "gpt-4o-mini"

    def test_default_spacy_model(self) -> None:
        settings = Settings()
        assert settings.spacy_model == "en_core_web_sm"

    def test_default_max_content_length(self) -> None:
        settings = Settings()
        assert settings.max_content_length == 50_000

    def test_enable_ai_suggestions_default_true(self) -> None:
        settings = Settings()
        assert settings.enable_ai_suggestions is True


# ---------------------------------------------------------------------------
# ScoreLabel tests
# ---------------------------------------------------------------------------


class TestScoreLabel:
    """Tests for ScoreLabel derivation from numeric scores."""

    @pytest.mark.parametrize(
        "score, expected",
        [
            (100.0, ScoreLabel.EXCELLENT),
            (80.0, ScoreLabel.EXCELLENT),
            (79.9, ScoreLabel.GOOD),
            (60.0, ScoreLabel.GOOD),
            (59.9, ScoreLabel.FAIR),
            (40.0, ScoreLabel.FAIR),
            (39.9, ScoreLabel.POOR),
            (20.0, ScoreLabel.POOR),
            (19.9, ScoreLabel.CRITICAL),
            (0.0, ScoreLabel.CRITICAL),
        ],
    )
    def test_from_score_boundaries(self, score: float, expected: ScoreLabel) -> None:
        assert ScoreLabel.from_score(score) == expected

    def test_label_values_are_strings(self) -> None:
        for label in ScoreLabel:
            assert isinstance(label.value, str)


# ---------------------------------------------------------------------------
# ContentInput tests
# ---------------------------------------------------------------------------


class TestContentInput:
    """Tests for ContentInput validation."""

    def test_valid_content_accepted(self) -> None:
        ci = ContentInput(content=MIN_CONTENT)
        assert ci.content == MIN_CONTENT

    def test_content_is_stripped(self) -> None:
        padded = "   " + MIN_CONTENT + "   "
        ci = ContentInput(content=padded)
        assert ci.content == MIN_CONTENT

    def test_content_too_short_raises_error(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ContentInput(content="short")
        assert "50" in str(exc_info.value)

    def test_empty_content_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content="")

    def test_content_at_exactly_minimum_length(self) -> None:
        ci = ContentInput(content="x" * 50)
        assert len(ci.content) == 50

    def test_content_exceeding_max_length_raises_error(self) -> None:
        # Use a settings instance with a small max to avoid building a 50k string
        oversized = "a" * 51
        # Patch max_content_length by using a small settings instance
        # We test the validator with a temporarily lowered limit via monkeypatching
        # For a simpler approach: build actual oversized content with real limit
        # The default limit is 50000 so we just check the error path exists
        # by constructing with max_content_length=100
        from unittest.mock import patch

        small_settings = Settings(max_content_length=100)
        with patch("geo_aeo_optimizer.models.get_settings", return_value=small_settings):
            with pytest.raises(ValidationError) as exc_info:
                ContentInput(content="a" * 101)
            assert "maximum" in str(exc_info.value).lower() or "100" in str(exc_info.value)

    def test_non_string_content_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content=12345)  # type: ignore[arg-type]

    def test_target_query_defaults_to_none(self) -> None:
        ci = ContentInput(content=MIN_CONTENT)
        assert ci.target_query is None

    def test_target_query_is_stripped(self) -> None:
        ci = ContentInput(content=MIN_CONTENT, target_query="  what is GEO?  ")
        assert ci.target_query == "what is GEO?"

    def test_empty_target_query_becomes_none(self) -> None:
        ci = ContentInput(content=MIN_CONTENT, target_query="   ")
        assert ci.target_query is None

    def test_target_query_none_explicit(self) -> None:
        ci = ContentInput(content=MIN_CONTENT, target_query=None)
        assert ci.target_query is None

    def test_include_suggestions_defaults_to_true(self) -> None:
        ci = ContentInput(content=MIN_CONTENT)
        assert ci.include_suggestions is True

    def test_include_suggestions_can_be_false(self) -> None:
        ci = ContentInput(content=MIN_CONTENT, include_suggestions=False)
        assert ci.include_suggestions is False

    def test_non_string_target_query_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content=MIN_CONTENT, target_query=999)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DimensionScore tests
# ---------------------------------------------------------------------------


class TestDimensionScore:
    """Tests for DimensionScore creation and field derivation."""

    def test_create_computes_weighted_score(self) -> None:
        ds = DimensionScore.create(
            dimension=DimensionKey.QA_ALIGNMENT,
            display_name="Question-Answer Alignment",
            raw_score=80.0,
            weight=1.5,
            explanation="Test",
        )
        assert ds.weighted_score == pytest.approx(80.0 * 1.5)

    def test_create_derives_label_excellent(self) -> None:
        ds = make_dimension(DimensionKey.ENTITY_DENSITY, 90.0)
        assert ds.label == ScoreLabel.EXCELLENT

    def test_create_derives_label_critical(self) -> None:
        ds = make_dimension(DimensionKey.CITATION_CUES, 10.0)
        assert ds.label == ScoreLabel.CRITICAL

    def test_raw_score_clamped_above_100(self) -> None:
        ds = make_dimension(DimensionKey.SEMANTIC_CLARITY, 150.0)
        assert ds.raw_score == 100.0

    def test_raw_score_clamped_below_0(self) -> None:
        ds = make_dimension(DimensionKey.CONTENT_DEPTH, -10.0)
        assert ds.raw_score == 0.0

    def test_raw_score_rounded_to_2_decimals(self) -> None:
        ds = make_dimension(DimensionKey.STRUCTURED_FORMATTING, 66.6666666)
        assert ds.raw_score == pytest.approx(66.67, abs=0.01)

    def test_improvement_priority_default_zero(self) -> None:
        ds = make_dimension(DimensionKey.QA_ALIGNMENT, 50.0)
        assert ds.improvement_priority == 0

    def test_invalid_raw_score_above_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=DimensionKey.QA_ALIGNMENT,
                display_name="Test",
                raw_score=101.0,
                weight=1.0,
                weighted_score=101.0,
                label=ScoreLabel.EXCELLENT,
                explanation="Bad",
            )

    def test_invalid_weight_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=DimensionKey.QA_ALIGNMENT,
                display_name="Test",
                raw_score=50.0,
                weight=0.0,
                weighted_score=0.0,
                label=ScoreLabel.FAIR,
                explanation="Bad",
            )


# ---------------------------------------------------------------------------
# SuggestionItem tests
# ---------------------------------------------------------------------------


class TestSuggestionItem:
    """Tests for SuggestionItem construction."""

    def test_minimal_suggestion_item(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.CITATION_CUES,
            display_name="Citation Cues",
            issue="No citations found.",
            suggestion="Add at least two statistical references.",
        )
        assert item.before_example is None
        assert item.after_example is None

    def test_full_suggestion_item(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.QA_ALIGNMENT,
            display_name="Question-Answer Alignment",
            issue="No question patterns detected.",
            suggestion="Add explicit Q&A pairs.",
            before_example="Python is fast.",
            after_example="What makes Python fast? Python's JIT and C extensions.",
        )
        assert item.before_example == "Python is fast."
        assert item.after_example is not None


# ---------------------------------------------------------------------------
# AnalysisResult tests
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    """Tests for AnalysisResult.build and composite score calculation."""

    def test_uniform_scores_composite_equals_raw(self) -> None:
        """When all dimensions have the same raw score, composite == raw."""
        dims = make_all_dimensions(score=70.0)
        result = AnalysisResult.build(
            dimensions=dims,
            content_word_count=100,
            content_char_count=500,
        )
        assert result.composite_score == pytest.approx(70.0, abs=0.01)

    def test_composite_score_within_range(self) -> None:
        dims = make_all_dimensions(score=55.0)
        result = AnalysisResult.build(dimensions=dims)
        assert 0.0 <= result.composite_score <= 100.0

    def test_composite_label_derived_correctly(self) -> None:
        dims = make_all_dimensions(score=85.0)
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_label == ScoreLabel.EXCELLENT

    def test_composite_label_poor_for_low_scores(self) -> None:
        dims = make_all_dimensions(score=25.0)
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_label == ScoreLabel.POOR

    def test_improvement_priority_assigned(self) -> None:
        dims = make_all_dimensions(score=50.0)
        # Give one dimension a lower score so it gets priority 1
        dims[0] = make_dimension(DimensionKey.QA_ALIGNMENT, 10.0, 1.5)
        result = AnalysisResult.build(dimensions=dims)
        # The lowest-scored dimension must have priority 1
        lowest = min(result.dimensions, key=lambda d: d.raw_score)
        assert lowest.improvement_priority == 1

    def test_dimensions_sorted_by_priority_ascending(self) -> None:
        dims = make_all_dimensions(score=50.0)
        result = AnalysisResult.build(dimensions=dims)
        priorities = [d.improvement_priority for d in result.dimensions]
        assert priorities == sorted(priorities)

    def test_weighted_composite_differs_from_simple_average(self) -> None:
        """Verify weights actually affect the composite score."""
        # One dimension with a high weight and a very different score
        dims = [
            make_dimension(DimensionKey.QA_ALIGNMENT, 100.0, 10.0),  # heavy weight, perfect
            make_dimension(DimensionKey.ENTITY_DENSITY, 0.0, 1.0),
            make_dimension(DimensionKey.STRUCTURED_FORMATTING, 0.0, 1.0),
            make_dimension(DimensionKey.CITATION_CUES, 0.0, 1.0),
            make_dimension(DimensionKey.SEMANTIC_CLARITY, 0.0, 1.0),
            make_dimension(DimensionKey.CONTENT_DEPTH, 0.0, 1.0),
        ]
        result = AnalysisResult.build(dimensions=dims)
        # Simple average would be ~16.7; weighted should be 10/15*100 ≈ 66.7
        simple_average = 100.0 / 6
        assert result.composite_score > simple_average

    def test_empty_suggestions_by_default(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims)
        assert result.suggestions == []

    def test_suggestions_passed_through(self) -> None:
        dims = make_all_dimensions()
        sugg = SuggestionItem(
            dimension=DimensionKey.CITATION_CUES,
            display_name="Citation Cues",
            issue="Missing citations.",
            suggestion="Add references.",
        )
        result = AnalysisResult.build(dimensions=dims, suggestions=[sugg])
        assert len(result.suggestions) == 1
        assert result.suggestions[0].dimension == DimensionKey.CITATION_CUES

    def test_target_query_echoed(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims, target_query="What is GEO?")
        assert result.target_query == "What is GEO?"

    def test_content_counts_stored(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(
            dimensions=dims,
            content_word_count=200,
            content_char_count=1200,
        )
        assert result.content_word_count == 200
        assert result.content_char_count == 1200

    def test_error_message_stored(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims, error_message="AI timeout")
        assert result.error_message == "AI timeout"

    def test_error_message_none_by_default(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims)
        assert result.error_message is None

    def test_empty_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AnalysisResult.build(dimensions=[])

    def test_more_than_six_dimensions_raises(self) -> None:
        dims = make_all_dimensions() + make_all_dimensions()
        with pytest.raises(ValueError, match="6"):
            AnalysisResult.build(dimensions=dims)

    def test_perfect_score(self) -> None:
        dims = make_all_dimensions(score=100.0)
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_score == pytest.approx(100.0, abs=0.01)
        assert result.composite_label == ScoreLabel.EXCELLENT

    def test_zero_score(self) -> None:
        dims = make_all_dimensions(score=0.0)
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_score == pytest.approx(0.0, abs=0.01)
        assert result.composite_label == ScoreLabel.CRITICAL


# ---------------------------------------------------------------------------
# Dimension display name registry tests
# ---------------------------------------------------------------------------


class TestDimensionDisplayNames:
    """Tests for the DIMENSION_DISPLAY_NAMES registry."""

    def test_all_dimension_keys_have_display_names(self) -> None:
        for key in DimensionKey:
            name = get_dimension_display_name(key)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_qa_alignment_display_name(self) -> None:
        assert get_dimension_display_name(DimensionKey.QA_ALIGNMENT) == "Question-Answer Alignment"

    def test_entity_density_display_name(self) -> None:
        assert get_dimension_display_name(DimensionKey.ENTITY_DENSITY) == "Entity Density"

    def test_structured_formatting_display_name(self) -> None:
        assert (
            get_dimension_display_name(DimensionKey.STRUCTURED_FORMATTING)
            == "Structured Formatting"
        )

    def test_citation_cues_display_name(self) -> None:
        assert get_dimension_display_name(DimensionKey.CITATION_CUES) == "Citation Cues"

    def test_semantic_clarity_display_name(self) -> None:
        assert get_dimension_display_name(DimensionKey.SEMANTIC_CLARITY) == "Semantic Clarity"

    def test_content_depth_display_name(self) -> None:
        assert get_dimension_display_name(DimensionKey.CONTENT_DEPTH) == "Content Depth"


# ---------------------------------------------------------------------------
# get_settings caching tests
# ---------------------------------------------------------------------------


class TestGetSettingsCache:
    """Tests for the get_settings LRU cache behaviour."""

    def test_get_settings_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_returns_same_object(self) -> None:
        """Verify the cache returns the same object on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
