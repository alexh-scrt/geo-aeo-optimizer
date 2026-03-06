"""Pydantic data models for the GEO/AEO Content Optimizer.

This module defines all request and response schemas used throughout the
application, including:

- ``Settings``: Application configuration sourced from environment variables,
  including scoring weights for each GEO/AEO dimension.
- ``ContentInput``: Validated request schema for content submitted by users.
- ``DimensionScore``: Schema representing a single scored GEO/AEO dimension
  with its raw score, weighted contribution, label, and explanation.
- ``AnalysisResult``: Complete response schema aggregating all dimension
  scores into a composite score with AI-generated suggestions.

Scoring weights are configurable via environment variables so content teams
can tune which GEO/AEO signals matter most for their publishing domain.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Application settings (loaded from environment / .env file)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All fields have sensible defaults so the application can run without
    a ``.env`` file.  Override any value by setting the corresponding
    environment variable or by placing a ``key=value`` pair in ``.env``.

    Scoring weight attributes control the relative importance of each
    GEO/AEO dimension in the composite score calculation.  They must all
    be strictly positive floats.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # OpenAI configuration
    # ------------------------------------------------------------------
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key used for generating rewrite suggestions.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier for suggestions generation.",
    )
    openai_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum tokens in the OpenAI API response.",
    )
    openai_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout in seconds for OpenAI API requests.",
    )

    # ------------------------------------------------------------------
    # Application server configuration
    # ------------------------------------------------------------------
    app_host: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    app_port: int = Field(default=8000, ge=1, le=65535, description="Uvicorn bind port.")
    app_reload: bool = Field(default=False, description="Enable Uvicorn auto-reload.")
    log_level: str = Field(default="INFO", description="Application log level.")

    # ------------------------------------------------------------------
    # spaCy configuration
    # ------------------------------------------------------------------
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy language model name.  Must be downloaded separately.",
    )

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    enable_ai_suggestions: bool = Field(
        default=True,
        description="When False, skip OpenAI calls and return no suggestions.",
    )
    max_content_length: int = Field(
        default=50_000,
        ge=100,
        description="Maximum character length for submitted content.",
    )

    # ------------------------------------------------------------------
    # Scoring weights  (must be > 0)
    # ------------------------------------------------------------------
    weight_qa_alignment: float = Field(
        default=1.5,
        gt=0,
        description="Weight applied to the Question-Answer Alignment dimension.",
    )
    weight_entity_density: float = Field(
        default=1.0,
        gt=0,
        description="Weight applied to the Entity Density dimension.",
    )
    weight_structured_formatting: float = Field(
        default=1.0,
        gt=0,
        description="Weight applied to the Structured Formatting dimension.",
    )
    weight_citation_cues: float = Field(
        default=1.2,
        gt=0,
        description="Weight applied to the Citation Cues dimension.",
    )
    weight_semantic_clarity: float = Field(
        default=1.0,
        gt=0,
        description="Weight applied to the Semantic Clarity dimension.",
    )
    weight_content_depth: float = Field(
        default=1.3,
        gt=0,
        description="Weight applied to the Content Depth dimension.",
    )

    @property
    def total_weight(self) -> float:
        """Sum of all dimension weights — used for normalising the composite score."""
        return (
            self.weight_qa_alignment
            + self.weight_entity_density
            + self.weight_structured_formatting
            + self.weight_citation_cues
            + self.weight_semantic_clarity
            + self.weight_content_depth
        )

    @property
    def weights_by_dimension(self) -> dict[str, float]:
        """Return a mapping of dimension key → weight for convenient lookup."""
        return {
            "qa_alignment": self.weight_qa_alignment,
            "entity_density": self.weight_entity_density,
            "structured_formatting": self.weight_structured_formatting,
            "citation_cues": self.weight_citation_cues,
            "semantic_clarity": self.weight_semantic_clarity,
            "content_depth": self.weight_content_depth,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application ``Settings`` singleton.

    Using ``lru_cache`` ensures the ``.env`` file is only parsed once per
    process.  Tests that need to override settings should call
    ``get_settings.cache_clear()`` before and after patching environment
    variables.

    Returns:
        Settings: The loaded and validated application settings.
    """
    return Settings()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DimensionKey(str, Enum):
    """Canonical keys for the six GEO/AEO scoring dimensions.

    Using a string enum allows dimension keys to be serialised directly to
    JSON strings while still benefiting from enum-level type safety.
    """

    QA_ALIGNMENT = "qa_alignment"
    ENTITY_DENSITY = "entity_density"
    STRUCTURED_FORMATTING = "structured_formatting"
    CITATION_CUES = "citation_cues"
    SEMANTIC_CLARITY = "semantic_clarity"
    CONTENT_DEPTH = "content_depth"


class ScoreLabel(str, Enum):
    """Human-readable quality label derived from a dimension's raw score.

    Labels are mapped from the 0-100 raw score range:
    - EXCELLENT  ≥ 80
    - GOOD       ≥ 60
    - FAIR       ≥ 40
    - POOR       ≥ 20
    - CRITICAL   <  20
    """

    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    CRITICAL = "Critical"

    @classmethod
    def from_score(cls, score: float) -> "ScoreLabel":
        """Derive a ``ScoreLabel`` from a 0-100 numeric score.

        Args:
            score: Numeric score in the range [0, 100].

        Returns:
            ScoreLabel: The corresponding quality label.
        """
        if score >= 80:
            return cls.EXCELLENT
        if score >= 60:
            return cls.GOOD
        if score >= 40:
            return cls.FAIR
        if score >= 20:
            return cls.POOR
        return cls.CRITICAL


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ContentInput(BaseModel):
    """Request schema for content submitted for GEO/AEO analysis.

    Attributes:
        content: The main body text to be scored.  Must be between 50 and
            50,000 characters after stripping leading/trailing whitespace.
        target_query: An optional keyword or question that represents the
            user's intended search intent.  When provided, the scoring engine
            uses it to compute question-answer alignment more accurately.
        include_suggestions: Whether to request AI-powered rewrite suggestions
            from OpenAI in addition to the heuristic scores.  Defaults to
            ``True``.  Setting this to ``False`` reduces latency and cost.
    """

    content: Annotated[
        str,
        Field(
            min_length=50,
            description=(
                "The content text to analyse.  Must be at least 50 characters long."
            ),
        ),
    ]
    target_query: Annotated[
        str | None,
        Field(
            default=None,
            max_length=500,
            description=(
                "Optional target search query or keyword phrase used to focus the "
                "question-answer alignment dimension of the analysis."
            ),
        ),
    ] = None
    include_suggestions: bool = Field(
        default=True,
        description=(
            "When True (default), AI-powered rewrite suggestions are generated via "
            "OpenAI for the lowest-scoring dimensions."
        ),
    )

    @field_validator("content", mode="before")
    @classmethod
    def strip_and_validate_content(cls, value: object) -> str:
        """Strip whitespace and enforce the configured maximum content length.

        Args:
            value: Raw field value received from the request.

        Returns:
            str: Stripped content string.

        Raises:
            ValueError: If ``value`` is not a string or exceeds the maximum
                allowed length configured in ``Settings.max_content_length``.
        """
        if not isinstance(value, str):
            raise ValueError("Content must be a string.")
        stripped = value.strip()
        max_length = get_settings().max_content_length
        if len(stripped) > max_length:
            raise ValueError(
                f"Content exceeds the maximum allowed length of {max_length:,} characters. "
                f"Received {len(stripped):,} characters."
            )
        return stripped

    @field_validator("target_query", mode="before")
    @classmethod
    def strip_target_query(cls, value: object) -> str | None:
        """Strip whitespace from the optional target query field.

        Args:
            value: Raw field value received from the request.

        Returns:
            str | None: Stripped string, or ``None`` if the value was empty
            or ``None``.

        Raises:
            ValueError: If ``value`` is not a string or ``None``.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("target_query must be a string or null.")
        stripped = value.strip()
        return stripped if stripped else None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": (
                        "Python is a high-level, general-purpose programming language "
                        "known for its clear syntax and readability. According to the "
                        "2024 Stack Overflow Developer Survey, Python is the most popular "
                        "language for the third consecutive year. What is Python used for? "
                        "Python is widely used in data science, machine learning, web "
                        "development, automation, and scientific computing."
                    ),
                    "target_query": "What is Python used for?",
                    "include_suggestions": True,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score and metadata for a single GEO/AEO dimension.

    Attributes:
        dimension: The canonical key identifying which dimension was scored.
        display_name: A human-readable name for display in the UI.
        raw_score: The unweighted score for this dimension on a 0-100 scale.
        weight: The weight applied to this dimension in the composite score
            calculation, as configured in ``Settings``.
        weighted_score: The dimension's contribution to the composite score
            (``raw_score × weight``).  Used internally for aggregation.
        label: A human-readable quality label (Excellent / Good / Fair /
            Poor / Critical) derived from ``raw_score``.
        explanation: A brief, human-readable explanation of what this
            dimension measures and how this specific content performed.
        improvement_priority: An integer priority rank (1 = highest priority
            to address first) assigned during ``AnalysisResult`` construction.
            Lower values indicate dimensions where improvement will yield the
            greatest composite score gain.
    """

    dimension: DimensionKey = Field(description="Canonical dimension identifier.")
    display_name: str = Field(description="Human-readable dimension name for UI display.")
    raw_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Unweighted dimension score in the range [0, 100].",
    )
    weight: float = Field(
        gt=0.0,
        description="Configured weight for this dimension in composite score calculation.",
    )
    weighted_score: float = Field(
        ge=0.0,
        description="Product of raw_score and weight; used for composite score aggregation.",
    )
    label: ScoreLabel = Field(
        description="Quality label derived from raw_score (Excellent / Good / Fair / Poor / Critical)."
    )
    explanation: str = Field(
        description="Brief explanation of how this content performed on this dimension."
    )
    improvement_priority: int = Field(
        default=0,
        ge=0,
        description=(
            "Priority rank for addressing this dimension (1 = most important). "
            "Set by AnalysisResult after all dimensions are scored."
        ),
    )

    @classmethod
    def create(
        cls,
        dimension: DimensionKey,
        display_name: str,
        raw_score: float,
        weight: float,
        explanation: str,
    ) -> "DimensionScore":
        """Factory method to construct a ``DimensionScore`` with derived fields.

        Automatically computes ``weighted_score`` and ``label`` from the
        provided ``raw_score`` and ``weight``.

        Args:
            dimension: The canonical ``DimensionKey`` for this score.
            display_name: Human-readable name for UI display.
            raw_score: Unweighted score in [0, 100].
            weight: Positive float weight for composite score calculation.
            explanation: Human-readable explanation of the score.

        Returns:
            DimensionScore: A fully populated dimension score instance.
        """
        clamped = max(0.0, min(100.0, raw_score))
        return cls(
            dimension=dimension,
            display_name=display_name,
            raw_score=round(clamped, 2),
            weight=weight,
            weighted_score=round(clamped * weight, 4),
            label=ScoreLabel.from_score(clamped),
            explanation=explanation,
        )

    model_config = {"use_enum_values": False}


class SuggestionItem(BaseModel):
    """A single AI-generated rewrite suggestion for a specific dimension.

    Attributes:
        dimension: The dimension this suggestion targets.
        display_name: Human-readable dimension name for UI grouping.
        issue: A concise description of the problem identified.
        suggestion: Detailed, actionable advice for addressing the issue.
        before_example: Optional short excerpt from the original content
            that illustrates the problem.
        after_example: Optional rewritten version of ``before_example``
            demonstrating how to apply the suggestion.
    """

    dimension: DimensionKey = Field(
        description="The dimension this suggestion addresses."
    )
    display_name: str = Field(description="Human-readable dimension name.")
    issue: str = Field(
        description="Concise description of the identified content problem."
    )
    suggestion: str = Field(
        description="Actionable advice for improving the dimension score."
    )
    before_example: str | None = Field(
        default=None,
        description="Short excerpt from the original content illustrating the problem.",
    )
    after_example: str | None = Field(
        default=None,
        description="Rewritten version of before_example demonstrating the improvement.",
    )


class AnalysisResult(BaseModel):
    """Complete GEO/AEO analysis response returned to the client.

    Attributes:
        composite_score: The final weighted-average score across all six
            dimensions, normalised to the range [0, 100].  Higher is better.
        composite_label: Quality label derived from ``composite_score``.
        dimensions: Ordered list of all six ``DimensionScore`` objects,
            sorted by ``improvement_priority`` (most important first).
        suggestions: List of AI-generated ``SuggestionItem`` objects for the
            lowest-scoring dimensions.  Empty when suggestions were not
            requested or when ``enable_ai_suggestions`` is ``False``.
        target_query: Echo of the target query submitted with the request,
            or ``None`` if none was provided.
        content_word_count: The number of words in the analysed content.
        content_char_count: The number of characters in the analysed content.
        error_message: Non-fatal error message, e.g. if AI suggestions failed
            but heuristic scoring succeeded.  ``None`` when everything worked.
    """

    composite_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Weighted-average composite GEO/AEO score in [0, 100].",
    )
    composite_label: ScoreLabel = Field(
        description="Quality label derived from composite_score."
    )
    dimensions: list[DimensionScore] = Field(
        description=(
            "All six dimension scores, ordered by improvement_priority ascending "
            "(highest-priority improvement first)."
        )
    )
    suggestions: list[SuggestionItem] = Field(
        default_factory=list,
        description="AI-generated rewrite suggestions for the lowest-scoring dimensions.",
    )
    target_query: str | None = Field(
        default=None,
        description="Echo of the target query used during analysis, or None.",
    )
    content_word_count: int = Field(
        ge=0,
        description="Number of whitespace-delimited words in the analysed content.",
    )
    content_char_count: int = Field(
        ge=0,
        description="Number of characters in the analysed content.",
    )
    error_message: str | None = Field(
        default=None,
        description=(
            "Non-fatal error message if part of the analysis failed (e.g., AI "
            "suggestions unavailable).  None when everything succeeded."
        ),
    )

    @classmethod
    def build(
        cls,
        dimensions: list[DimensionScore],
        settings: Settings | None = None,
        suggestions: list[SuggestionItem] | None = None,
        target_query: str | None = None,
        content_word_count: int = 0,
        content_char_count: int = 0,
        error_message: str | None = None,
    ) -> "AnalysisResult":
        """Build an ``AnalysisResult`` from a list of ``DimensionScore`` objects.

        Computes the weighted composite score, derives the composite label,
        assigns ``improvement_priority`` to each dimension (lower raw score =
        higher priority), and sorts the dimension list accordingly.

        Args:
            dimensions: List of all six scored ``DimensionScore`` objects.
            settings: Application settings used for weight lookup.  Defaults
                to the global ``get_settings()`` singleton.
            suggestions: Optional list of AI-generated suggestions.  Defaults
                to an empty list.
            target_query: Target query echoed from the request.
            content_word_count: Word count of the analysed content.
            content_char_count: Character count of the analysed content.
            error_message: Optional non-fatal error message.

        Returns:
            AnalysisResult: Fully populated analysis result.

        Raises:
            ValueError: If ``dimensions`` is empty or contains more than six
                items.
        """
        if not dimensions:
            raise ValueError("AnalysisResult.build requires at least one DimensionScore.")
        if len(dimensions) > 6:
            raise ValueError(
                f"Expected at most 6 DimensionScore objects, got {len(dimensions)}."
            )

        cfg = settings if settings is not None else get_settings()

        # Compute weighted composite score
        total_weighted = sum(d.weighted_score for d in dimensions)
        total_weight = sum(d.weight for d in dimensions)
        raw_composite = total_weighted / total_weight if total_weight > 0 else 0.0
        composite = round(max(0.0, min(100.0, raw_composite)), 2)

        # Assign improvement priority: rank dimensions by raw_score ascending
        # (lowest score = priority 1, i.e. most urgent to fix).
        sorted_by_score = sorted(dimensions, key=lambda d: d.raw_score)
        for priority, dim in enumerate(sorted_by_score, start=1):
            dim.improvement_priority = priority

        # Re-sort the final list by improvement_priority
        sorted_dimensions = sorted(dimensions, key=lambda d: d.improvement_priority)

        return cls(
            composite_score=composite,
            composite_label=ScoreLabel.from_score(composite),
            dimensions=sorted_dimensions,
            suggestions=suggestions or [],
            target_query=target_query,
            content_word_count=content_word_count,
            content_char_count=content_char_count,
            error_message=error_message,
        )

    @model_validator(mode="after")
    def validate_dimensions_count(self) -> "AnalysisResult":
        """Ensure the dimensions list is not empty after construction.

        Returns:
            AnalysisResult: The validated instance.

        Raises:
            ValueError: If the dimensions list is empty.
        """
        if not self.dimensions:
            raise ValueError("AnalysisResult must contain at least one DimensionScore.")
        return self

    model_config = {"use_enum_values": False}


# ---------------------------------------------------------------------------
# Display name registry  (shared between scorer.py and templates)
# ---------------------------------------------------------------------------

DIMENSION_DISPLAY_NAMES: dict[DimensionKey, str] = {
    DimensionKey.QA_ALIGNMENT: "Question-Answer Alignment",
    DimensionKey.ENTITY_DENSITY: "Entity Density",
    DimensionKey.STRUCTURED_FORMATTING: "Structured Formatting",
    DimensionKey.CITATION_CUES: "Citation Cues",
    DimensionKey.SEMANTIC_CLARITY: "Semantic Clarity",
    DimensionKey.CONTENT_DEPTH: "Content Depth",
}


def get_dimension_display_name(dimension: DimensionKey) -> str:
    """Return the human-readable display name for a dimension key.

    Args:
        dimension: The ``DimensionKey`` to look up.

    Returns:
        str: The display name string.

    Raises:
        KeyError: If the dimension key is not found in ``DIMENSION_DISPLAY_NAMES``.
    """
    return DIMENSION_DISPLAY_NAMES[dimension]
