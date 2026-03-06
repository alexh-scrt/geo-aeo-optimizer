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

    def test_default_openai_max_tokens(self) -> None:
        settings = Settings()
        assert settings.openai_max_tokens == 1024

    def test_default_openai_timeout(self) -> None:
        settings = Settings()
        assert settings.openai_timeout == 30

    def test_default_app_host(self) -> None:
        settings = Settings()
        assert settings.app_host == "0.0.0.0"

    def test_default_app_port(self) -> None:
        settings = Settings()
        assert settings.app_port == 8000

    def test_default_app_reload_false(self) -> None:
        settings = Settings()
        assert settings.app_reload is False

    def test_default_log_level(self) -> None:
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_openai_api_key_empty(self) -> None:
        settings = Settings()
        assert settings.openai_api_key == ""

    def test_openai_max_tokens_ge_64(self) -> None:
        with pytest.raises(ValidationError):
            Settings(openai_max_tokens=63)

    def test_openai_max_tokens_le_4096(self) -> None:
        with pytest.raises(ValidationError):
            Settings(openai_max_tokens=4097)

    def test_openai_timeout_ge_5(self) -> None:
        with pytest.raises(ValidationError):
            Settings(openai_timeout=4)

    def test_openai_timeout_le_120(self) -> None:
        with pytest.raises(ValidationError):
            Settings(openai_timeout=121)

    def test_app_port_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            Settings(app_port=0)

    def test_app_port_le_65535(self) -> None:
        with pytest.raises(ValidationError):
            Settings(app_port=65536)

    def test_max_content_length_ge_100(self) -> None:
        with pytest.raises(ValidationError):
            Settings(max_content_length=99)

    def test_all_weight_keys_match_dimension_keys(self) -> None:
        settings = Settings()
        for key in DimensionKey:
            assert key.value in settings.weights_by_dimension

    def test_total_weight_with_custom_values(self) -> None:
        settings = Settings(
            weight_qa_alignment=1.0,
            weight_entity_density=1.0,
            weight_structured_formatting=1.0,
            weight_citation_cues=1.0,
            weight_semantic_clarity=1.0,
            weight_content_depth=1.0,
        )
        assert settings.total_weight == pytest.approx(6.0)

    def test_negative_max_content_length_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(max_content_length=-1)

    def test_all_structured_formatting_weight_positive(self) -> None:
        settings = Settings()
        assert settings.weight_structured_formatting > 0

    def test_all_citation_cues_weight_positive(self) -> None:
        settings = Settings()
        assert settings.weight_citation_cues > 0

    def test_all_semantic_clarity_weight_positive(self) -> None:
        settings = Settings()
        assert settings.weight_semantic_clarity > 0

    def test_all_content_depth_weight_positive(self) -> None:
        settings = Settings()
        assert settings.weight_content_depth > 0


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

    def test_all_labels_have_non_empty_values(self) -> None:
        for label in ScoreLabel:
            assert len(label.value) > 0

    def test_excellent_value_string(self) -> None:
        assert ScoreLabel.EXCELLENT.value == "Excellent"

    def test_good_value_string(self) -> None:
        assert ScoreLabel.GOOD.value == "Good"

    def test_fair_value_string(self) -> None:
        assert ScoreLabel.FAIR.value == "Fair"

    def test_poor_value_string(self) -> None:
        assert ScoreLabel.POOR.value == "Poor"

    def test_critical_value_string(self) -> None:
        assert ScoreLabel.CRITICAL.value == "Critical"

    def test_score_50_is_fair(self) -> None:
        assert ScoreLabel.from_score(50.0) == ScoreLabel.FAIR

    def test_score_30_is_poor(self) -> None:
        assert ScoreLabel.from_score(30.0) == ScoreLabel.POOR

    def test_score_10_is_critical(self) -> None:
        assert ScoreLabel.from_score(10.0) == ScoreLabel.CRITICAL

    def test_score_70_is_good(self) -> None:
        assert ScoreLabel.from_score(70.0) == ScoreLabel.GOOD

    def test_score_90_is_excellent(self) -> None:
        assert ScoreLabel.from_score(90.0) == ScoreLabel.EXCELLENT

    def test_five_labels_exist(self) -> None:
        assert len(list(ScoreLabel)) == 5


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

    def test_content_with_only_whitespace_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content="   " * 20)

    def test_content_49_chars_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content="a" * 49)

    def test_content_51_chars_accepted(self) -> None:
        ci = ContentInput(content="a" * 51)
        assert len(ci.content) == 51

    def test_target_query_max_length_500(self) -> None:
        # Exactly 500 chars should be accepted
        ci = ContentInput(content=MIN_CONTENT, target_query="q" * 500)
        assert len(ci.target_query) == 500

    def test_target_query_over_500_raises(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content=MIN_CONTENT, target_query="q" * 501)

    def test_content_newlines_preserved(self) -> None:
        content_with_newlines = "Line one\nLine two\nLine three\n" + "a" * 30
        ci = ContentInput(content=content_with_newlines)
        assert "\n" in ci.content

    def test_content_list_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content=["a", "b", "c"])  # type: ignore[arg-type]

    def test_content_none_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            ContentInput(content=None)  # type: ignore[arg-type]

    def test_include_suggestions_true_explicit(self) -> None:
        ci = ContentInput(content=MIN_CONTENT, include_suggestions=True)
        assert ci.include_suggestions is True

    def test_target_query_single_word_accepted(self) -> None:
        ci = ContentInput(content=MIN_CONTENT, target_query="python")
        assert ci.target_query == "python"

    def test_target_query_with_special_chars_accepted(self) -> None:
        ci = ContentInput(content=MIN_CONTENT, target_query="What is Python?")
        assert ci.target_query == "What is Python?"


# ---------------------------------------------------------------------------
# DimensionKey tests
# ---------------------------------------------------------------------------


class TestDimensionKey:
    """Tests for the DimensionKey enumeration."""

    def test_six_dimension_keys_exist(self) -> None:
        assert len(list(DimensionKey)) == 6

    def test_qa_alignment_value(self) -> None:
        assert DimensionKey.QA_ALIGNMENT.value == "qa_alignment"

    def test_entity_density_value(self) -> None:
        assert DimensionKey.ENTITY_DENSITY.value == "entity_density"

    def test_structured_formatting_value(self) -> None:
        assert DimensionKey.STRUCTURED_FORMATTING.value == "structured_formatting"

    def test_citation_cues_value(self) -> None:
        assert DimensionKey.CITATION_CUES.value == "citation_cues"

    def test_semantic_clarity_value(self) -> None:
        assert DimensionKey.SEMANTIC_CLARITY.value == "semantic_clarity"

    def test_content_depth_value(self) -> None:
        assert DimensionKey.CONTENT_DEPTH.value == "content_depth"

    def test_dimension_keys_are_strings(self) -> None:
        for key in DimensionKey:
            assert isinstance(key.value, str)

    def test_can_construct_from_string(self) -> None:
        assert DimensionKey("qa_alignment") == DimensionKey.QA_ALIGNMENT

    def test_invalid_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            DimensionKey("nonexistent_dimension")


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

    def test_create_derives_label_good(self) -> None:
        ds = make_dimension(DimensionKey.SEMANTIC_CLARITY, 65.0)
        assert ds.label == ScoreLabel.GOOD

    def test_create_derives_label_fair(self) -> None:
        ds = make_dimension(DimensionKey.CONTENT_DEPTH, 45.0)
        assert ds.label == ScoreLabel.FAIR

    def test_create_derives_label_poor(self) -> None:
        ds = make_dimension(DimensionKey.STRUCTURED_FORMATTING, 25.0)
        assert ds.label == ScoreLabel.POOR

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

    def test_invalid_raw_score_below_0_raises(self) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=DimensionKey.QA_ALIGNMENT,
                display_name="Test",
                raw_score=-1.0,
                weight=1.0,
                weighted_score=-1.0,
                label=ScoreLabel.CRITICAL,
                explanation="Bad",
            )

    def test_weighted_score_at_boundary_100(self) -> None:
        ds = make_dimension(DimensionKey.QA_ALIGNMENT, 100.0, 2.0)
        assert ds.weighted_score == pytest.approx(200.0)

    def test_weighted_score_at_boundary_0(self) -> None:
        ds = make_dimension(DimensionKey.QA_ALIGNMENT, 0.0, 2.0)
        assert ds.weighted_score == pytest.approx(0.0)

    def test_dimension_field_stored(self) -> None:
        ds = make_dimension(DimensionKey.CITATION_CUES, 50.0)
        assert ds.dimension == DimensionKey.CITATION_CUES

    def test_display_name_stored(self) -> None:
        ds = make_dimension(DimensionKey.CITATION_CUES, 50.0)
        assert ds.display_name == "Citation Cues"

    def test_explanation_stored(self) -> None:
        ds = DimensionScore.create(
            dimension=DimensionKey.QA_ALIGNMENT,
            display_name="Q&A",
            raw_score=50.0,
            weight=1.0,
            explanation="Custom explanation text",
        )
        assert ds.explanation == "Custom explanation text"

    def test_weight_stored(self) -> None:
        ds = make_dimension(DimensionKey.QA_ALIGNMENT, 50.0, 1.7)
        assert ds.weight == pytest.approx(1.7)

    def test_raw_score_exactly_100_accepted(self) -> None:
        ds = DimensionScore(
            dimension=DimensionKey.QA_ALIGNMENT,
            display_name="Test",
            raw_score=100.0,
            weight=1.0,
            weighted_score=100.0,
            label=ScoreLabel.EXCELLENT,
            explanation="Max score",
        )
        assert ds.raw_score == 100.0

    def test_raw_score_exactly_0_accepted(self) -> None:
        ds = DimensionScore(
            dimension=DimensionKey.QA_ALIGNMENT,
            display_name="Test",
            raw_score=0.0,
            weight=1.0,
            weighted_score=0.0,
            label=ScoreLabel.CRITICAL,
            explanation="Min score",
        )
        assert ds.raw_score == 0.0

    def test_create_with_high_weight(self) -> None:
        ds = make_dimension(DimensionKey.QA_ALIGNMENT, 50.0, 10.0)
        assert ds.weighted_score == pytest.approx(500.0)

    def test_create_with_fractional_score(self) -> None:
        ds = make_dimension(DimensionKey.QA_ALIGNMENT, 33.333, 1.0)
        assert 0.0 <= ds.raw_score <= 100.0


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

    def test_dimension_stored(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.ENTITY_DENSITY,
            display_name="Entity Density",
            issue="Low entity count.",
            suggestion="Add named entities.",
        )
        assert item.dimension == DimensionKey.ENTITY_DENSITY

    def test_display_name_stored(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.ENTITY_DENSITY,
            display_name="Entity Density",
            issue="Low entity count.",
            suggestion="Add named entities.",
        )
        assert item.display_name == "Entity Density"

    def test_issue_stored(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.CONTENT_DEPTH,
            display_name="Content Depth",
            issue="Content is too short.",
            suggestion="Add more detail.",
        )
        assert item.issue == "Content is too short."

    def test_suggestion_stored(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.CONTENT_DEPTH,
            display_name="Content Depth",
            issue="Content is too short.",
            suggestion="Expand each section with examples.",
        )
        assert item.suggestion == "Expand each section with examples."

    def test_before_example_none_default(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.SEMANTIC_CLARITY,
            display_name="Semantic Clarity",
            issue="Filler phrases.",
            suggestion="Remove filler.",
        )
        assert item.before_example is None

    def test_after_example_none_default(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.SEMANTIC_CLARITY,
            display_name="Semantic Clarity",
            issue="Filler phrases.",
            suggestion="Remove filler.",
        )
        assert item.after_example is None

    def test_all_six_dimension_keys_accepted(self) -> None:
        for key in DimensionKey:
            item = SuggestionItem(
                dimension=key,
                display_name=get_dimension_display_name(key),
                issue="Test issue.",
                suggestion="Test suggestion.",
            )
            assert item.dimension == key

    def test_suggestion_item_with_multiline_examples(self) -> None:
        item = SuggestionItem(
            dimension=DimensionKey.STRUCTURED_FORMATTING,
            display_name="Structured Formatting",
            issue="No headings detected.",
            suggestion="Add ## headings to separate sections.",
            before_example="Python is great.\nIt has many features.\nDevelopers love it.",
            after_example="## Why Python Is Great\n\n- Clean syntax\n- Large ecosystem",
        )
        assert "\n" in item.before_example
        assert "\n" in item.after_example


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

    def test_composite_label_critical_for_zero_scores(self) -> None:
        dims = make_all_dimensions(score=0.0)
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_label == ScoreLabel.CRITICAL

    def test_composite_label_fair_for_mid_scores(self) -> None:
        dims = make_all_dimensions(score=50.0)
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_label == ScoreLabel.FAIR

    def test_composite_label_good_for_good_scores(self) -> None:
        dims = make_all_dimensions(score=65.0)
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_label == ScoreLabel.GOOD

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

    def test_target_query_none_default(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims)
        assert result.target_query is None

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

    def test_six_dimensions_returned(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims)
        assert len(result.dimensions) == 6

    def test_all_priorities_in_range_1_to_6(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims)
        priorities = {d.improvement_priority for d in result.dimensions}
        assert priorities == {1, 2, 3, 4, 5, 6}

    def test_single_dimension_accepted(self) -> None:
        dims = [make_dimension(DimensionKey.QA_ALIGNMENT, 75.0)]
        result = AnalysisResult.build(dimensions=dims)
        assert result.composite_score == pytest.approx(75.0, abs=0.01)

    def test_composite_score_rounded_to_2_decimals(self) -> None:
        # Use scores that would produce a non-integer composite
        dims = [
            make_dimension(DimensionKey.QA_ALIGNMENT, 33.33, 1.0),
            make_dimension(DimensionKey.ENTITY_DENSITY, 66.67, 1.0),
            make_dimension(DimensionKey.STRUCTURED_FORMATTING, 50.0, 1.0),
            make_dimension(DimensionKey.CITATION_CUES, 75.0, 1.0),
            make_dimension(DimensionKey.SEMANTIC_CLARITY, 42.5, 1.0),
            make_dimension(DimensionKey.CONTENT_DEPTH, 88.0, 1.0),
        ]
        result = AnalysisResult.build(dimensions=dims)
        # Score should have at most 2 decimal places
        assert result.composite_score == round(result.composite_score, 2)

    def test_multiple_suggestions_stored(self) -> None:
        dims = make_all_dimensions()
        suggestions = [
            SuggestionItem(
                dimension=DimensionKey.CITATION_CUES,
                display_name="Citation Cues",
                issue="No citations.",
                suggestion="Add references.",
            ),
            SuggestionItem(
                dimension=DimensionKey.QA_ALIGNMENT,
                display_name="Question-Answer Alignment",
                issue="No questions.",
                suggestion="Add Q&A pairs.",
            ),
        ]
        result = AnalysisResult.build(dimensions=dims, suggestions=suggestions)
        assert len(result.suggestions) == 2

    def test_custom_settings_weights_affect_composite(self) -> None:
        """Different weight settings should produce different composite scores
        when dimension raw scores differ."""
        dims_for_qa = [
            make_dimension(DimensionKey.QA_ALIGNMENT, 100.0, 5.0),
            make_dimension(DimensionKey.ENTITY_DENSITY, 0.0, 1.0),
            make_dimension(DimensionKey.STRUCTURED_FORMATTING, 0.0, 1.0),
            make_dimension(DimensionKey.CITATION_CUES, 0.0, 1.0),
            make_dimension(DimensionKey.SEMANTIC_CLARITY, 0.0, 1.0),
            make_dimension(DimensionKey.CONTENT_DEPTH, 0.0, 1.0),
        ]
        dims_for_depth = [
            make_dimension(DimensionKey.QA_ALIGNMENT, 0.0, 1.0),
            make_dimension(DimensionKey.ENTITY_DENSITY, 0.0, 1.0),
            make_dimension(DimensionKey.STRUCTURED_FORMATTING, 0.0, 1.0),
            make_dimension(DimensionKey.CITATION_CUES, 0.0, 1.0),
            make_dimension(DimensionKey.SEMANTIC_CLARITY, 0.0, 1.0),
            make_dimension(DimensionKey.CONTENT_DEPTH, 100.0, 5.0),
        ]
        result_qa = AnalysisResult.build(dimensions=dims_for_qa)
        result_depth = AnalysisResult.build(dimensions=dims_for_depth)
        # Both have one perfect dimension with weight 5 and 5 zero dimensions with weight 1
        # Composite = (100*5 + 0*5) / (5+5) = 50 for both
        assert result_qa.composite_score == pytest.approx(result_depth.composite_score, abs=0.01)

    def test_content_word_count_default_zero(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims)
        assert result.content_word_count == 0

    def test_content_char_count_default_zero(self) -> None:
        dims = make_all_dimensions()
        result = AnalysisResult.build(dimensions=dims)
        assert result.content_char_count == 0

    def test_build_with_all_parameters(self) -> None:
        dims = make_all_dimensions(score=60.0)
        sugg = SuggestionItem(
            dimension=DimensionKey.ENTITY_DENSITY,
            display_name="Entity Density",
            issue="Low entities.",
            suggestion="Add more entities.",
        )
        settings = Settings()
        result = AnalysisResult.build(
            dimensions=dims,
            settings=settings,
            suggestions=[sugg],
            target_query="test query",
            content_word_count=150,
            content_char_count=900,
            error_message="partial error",
        )
        assert result.composite_score == pytest.approx(60.0, abs=0.01)
        assert len(result.suggestions) == 1
        assert result.target_query == "test query"
        assert result.content_word_count == 150
        assert result.content_char_count == 900
        assert result.error_message == "partial error"

    def test_varying_scores_priority_order_correct(self) -> None:
        """Dimension with lowest score gets priority 1."""
        dims = [
            make_dimension(DimensionKey.QA_ALIGNMENT, 90.0, 1.5),
            make_dimension(DimensionKey.ENTITY_DENSITY, 10.0, 1.0),   # lowest -> priority 1
            make_dimension(DimensionKey.STRUCTURED_FORMATTING, 70.0, 1.0),
            make_dimension(DimensionKey.CITATION_CUES, 50.0, 1.2),
            make_dimension(DimensionKey.SEMANTIC_CLARITY, 60.0, 1.0),
            make_dimension(DimensionKey.CONTENT_DEPTH, 80.0, 1.3),
        ]
        result = AnalysisResult.build(dimensions=dims)
        priority_one = next(d for d in result.dimensions if d.improvement_priority == 1)
        assert priority_one.dimension == DimensionKey.ENTITY_DENSITY
        assert priority_one.raw_score == pytest.approx(10.0, abs=0.01)

    def test_composite_score_is_weighted_average(self) -> None:
        """Manually verify the composite score formula."""
        dims = [
            make_dimension(DimensionKey.QA_ALIGNMENT, 80.0, 2.0),
            make_dimension(DimensionKey.ENTITY_DENSITY, 40.0, 1.0),
            make_dimension(DimensionKey.STRUCTURED_FORMATTING, 60.0, 1.0),
            make_dimension(DimensionKey.CITATION_CUES, 20.0, 1.0),
            make_dimension(DimensionKey.SEMANTIC_CLARITY, 50.0, 1.0),
            make_dimension(DimensionKey.CONTENT_DEPTH, 70.0, 1.0),
        ]
        # total_weight = 2+1+1+1+1+1 = 7
        # total_weighted = 160 + 40 + 60 + 20 + 50 + 70 = 400
        # composite = 400 / 7 ≈ 57.14
        result = AnalysisResult.build(dimensions=dims)
        expected = round(400.0 / 7.0, 2)
        assert result.composite_score == pytest.approx(expected, abs=0.01)


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

    def test_display_names_are_title_case_or_readable(self) -> None:
        for key in DimensionKey:
            name = get_dimension_display_name(key)
            # Name should contain at least one uppercase letter (proper noun/title)
            assert any(c.isupper() for c in name), f"Name '{name}' has no uppercase letters"

    def test_no_duplicate_display_names(self) -> None:
        names = [get_dimension_display_name(key) for key in DimensionKey]
        assert len(names) == len(set(names)), "Duplicate display names found"


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

    def test_get_settings_cache_can_be_cleared(self) -> None:
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # After clearing, a new instance should be created
        # (they may be equal but not necessarily the same object)
        assert isinstance(s2, Settings)
        # Restore cache for subsequent tests
        get_settings.cache_clear()
        get_settings()

    def test_get_settings_default_values_correct(self) -> None:
        settings = get_settings()
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.spacy_model == "en_core_web_sm"
        assert settings.max_content_length == 50_000
