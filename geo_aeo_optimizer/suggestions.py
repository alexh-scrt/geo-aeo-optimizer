"""OpenAI-powered rewrite suggestions module for GEO/AEO Content Optimizer.

This module generates prioritized, actionable rewrite suggestions for the
lowest-scoring GEO/AEO dimensions by calling the OpenAI Chat Completions API.

The suggestions are structured as ``SuggestionItem`` objects containing:
- A concise description of the identified problem
- Actionable advice for improvement
- Before/after rewrite examples drawn from the actual content

Typical usage::

    from geo_aeo_optimizer.suggestions import SuggestionsGenerator

    generator = SuggestionsGenerator()
    suggestions = await generator.generate(
        content="Your article text...",
        analysis_result=result,
        max_dimensions=3,
    )

If the OpenAI API key is not configured or ``enable_ai_suggestions`` is
``False`` in settings, the generator returns an empty list gracefully.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from geo_aeo_optimizer.models import (
    AnalysisResult,
    DimensionKey,
    DimensionScore,
    Settings,
    SuggestionItem,
    get_settings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

_DIMENSION_GUIDANCE: dict[DimensionKey, str] = {
    DimensionKey.QA_ALIGNMENT: (
        "Question-Answer Alignment measures whether the content directly answers "
        "questions a user might ask. Look for missing question patterns, lack of "
        "explicit Q&A structure, and failure to address common user queries about "
        "the topic."
    ),
    DimensionKey.ENTITY_DENSITY: (
        "Entity Density measures the richness of named entities (people, places, "
        "organizations, products, dates, statistics). Low scores indicate the content "
        "is too generic and lacks specific, citable references."
    ),
    DimensionKey.STRUCTURED_FORMATTING: (
        "Structured Formatting measures the use of headings, bullet lists, numbered "
        "lists, bold/italic emphasis, and logical paragraph breaks. AI parsers strongly "
        "prefer content with clear visual hierarchy."
    ),
    DimensionKey.CITATION_CUES: (
        "Citation Cues measures whether the content references sources, statistics, "
        "authoritative organizations, or research. Content with citable evidence is "
        "far more likely to be surfaced by AI assistants."
    ),
    DimensionKey.SEMANTIC_CLARITY: (
        "Semantic Clarity measures precision and unambiguity of language. Penalised "
        "signals include filler phrases, hedge words, very long sentences, and low "
        "lexical diversity. Rewarded signals include transition words and precise vocabulary."
    ),
    DimensionKey.CONTENT_DEPTH: (
        "Content Depth measures whether the content is substantive enough to serve as "
        "a reference. Low scores indicate insufficient word count, limited unique concepts, "
        "few explanatory phrases, or lack of comparative analysis."
    ),
}


def _build_system_prompt() -> str:
    """Build the system prompt for the suggestions generation call.

    Returns:
        str: The system prompt string.
    """
    return (
        "You are an expert content strategist specializing in Generative Engine "
        "Optimization (GEO) and Answer Engine Optimization (AEO). Your role is to "
        "analyze content that has been scored across multiple dimensions and provide "
        "concrete, actionable rewrite suggestions that will improve its likelihood of "
        "being surfaced or cited by AI assistants like ChatGPT or Gemini.\n\n"
        "For each underperforming dimension provided, you must:\n"
        "1. Identify the specific problem in the content\n"
        "2. Provide a clear, actionable suggestion for improvement\n"
        "3. Extract a short excerpt from the actual content that illustrates the problem\n"
        "4. Rewrite that excerpt to demonstrate the improvement\n\n"
        "Be specific, practical, and reference the actual content provided. "
        "Avoid generic advice. Always tie suggestions back to real examples from the text."
    )


def _build_user_prompt(
    content: str,
    dimensions_to_improve: list[DimensionScore],
    target_query: str | None,
) -> str:
    """Build the user prompt for a specific content analysis.

    Args:
        content: The original content text being analyzed.
        dimensions_to_improve: The lowest-scoring dimensions needing improvement.
        target_query: Optional target search query for context.

    Returns:
        str: The formatted user prompt.
    """
    # Truncate content for the prompt to avoid token limits
    max_content_chars = 3000
    truncated_content = content[:max_content_chars]
    if len(content) > max_content_chars:
        truncated_content += "\n... [content truncated for brevity]"

    query_context = ""
    if target_query:
        query_context = f"\n**Target Query / Keyword:** {target_query}\n"

    dimension_lines = []
    for dim in dimensions_to_improve:
        guidance = _DIMENSION_GUIDANCE.get(dim.dimension, "")
        dimension_lines.append(
            f"- **{dim.display_name}** (score: {dim.raw_score:.0f}/100, "
            f"label: {dim.label.value})\n"
            f"  Current explanation: {dim.explanation}\n"
            f"  What this dimension measures: {guidance}"
        )

    dimensions_block = "\n".join(dimension_lines)

    json_schema = json.dumps(
        {
            "suggestions": [
                {
                    "dimension_key": "<one of: qa_alignment, entity_density, structured_formatting, citation_cues, semantic_clarity, content_depth>",
                    "issue": "<concise description of the specific problem found in the content>",
                    "suggestion": "<detailed, actionable advice referencing the actual content>",
                    "before_example": "<short excerpt from the content illustrating the problem, max 100 words>",
                    "after_example": "<rewritten version of the excerpt demonstrating the improvement, max 120 words>",
                }
            ]
        },
        indent=2,
    )

    return (
        f"Please analyze the following content and provide improvement suggestions "
        f"for the underperforming GEO/AEO dimensions listed below.\n"
        f"{query_context}\n"
        f"**Content to Analyze:**\n"
        f"```\n{truncated_content}\n```\n\n"
        f"**Dimensions Requiring Improvement:**\n"
        f"{dimensions_block}\n\n"
        f"**Required Response Format (JSON only, no markdown fences):**\n"
        f"{json_schema}\n\n"
        f"Respond with a valid JSON object matching the schema above. "
        f"Include exactly one suggestion object per dimension listed. "
        f"Do not include any text outside the JSON object."
    )


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _parse_suggestions_response(
    response_text: str,
    dimensions_to_improve: list[DimensionScore],
) -> list[SuggestionItem]:
    """Parse the OpenAI JSON response into a list of ``SuggestionItem`` objects.

    Handles malformed JSON gracefully by returning partial results when possible.

    Args:
        response_text: The raw text returned by OpenAI.
        dimensions_to_improve: The dimensions we requested suggestions for,
            used to look up display names.

    Returns:
        list[SuggestionItem]: Parsed suggestion items.  May be empty if parsing
            fails entirely.
    """
    # Build a lookup from dimension key string to display name
    display_name_lookup: dict[str, str] = {
        dim.dimension.value: dim.display_name for dim in dimensions_to_improve
    }

    # Strip markdown code fences if the model added them despite instructions
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove opening fence line
        lines = lines[1:]
        # Remove closing fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse OpenAI suggestions response as JSON: %s. "
            "Raw response (first 500 chars): %s",
            exc,
            response_text[:500],
        )
        return []

    raw_suggestions = data.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        logger.warning(
            "OpenAI response 'suggestions' field is not a list: %r",
            type(raw_suggestions),
        )
        return []

    items: list[SuggestionItem] = []
    valid_dimension_keys = {key.value for key in DimensionKey}

    for raw in raw_suggestions:
        if not isinstance(raw, dict):
            continue

        dim_key_str = raw.get("dimension_key", "")
        if dim_key_str not in valid_dimension_keys:
            logger.warning(
                "Unknown dimension key in suggestions response: %r. Skipping.",
                dim_key_str,
            )
            continue

        try:
            dimension = DimensionKey(dim_key_str)
        except ValueError:
            logger.warning("Could not construct DimensionKey from %r.", dim_key_str)
            continue

        display_name = display_name_lookup.get(dim_key_str, dim_key_str.replace("_", " ").title())
        issue = raw.get("issue", "").strip()
        suggestion = raw.get("suggestion", "").strip()
        before_example = raw.get("before_example") or None
        after_example = raw.get("after_example") or None

        if not issue or not suggestion:
            logger.warning(
                "Suggestion for dimension '%s' is missing 'issue' or 'suggestion' fields.",
                dim_key_str,
            )
            continue

        # Sanitize optional fields
        if isinstance(before_example, str):
            before_example = before_example.strip() or None
        else:
            before_example = None

        if isinstance(after_example, str):
            after_example = after_example.strip() or None
        else:
            after_example = None

        items.append(
            SuggestionItem(
                dimension=dimension,
                display_name=display_name,
                issue=issue,
                suggestion=suggestion,
                before_example=before_example,
                after_example=after_example,
            )
        )

    return items


# ---------------------------------------------------------------------------
# Main suggestions generator class
# ---------------------------------------------------------------------------


class SuggestionsGenerator:
    """OpenAI-powered rewrite suggestions generator.

    Generates ``SuggestionItem`` objects for the lowest-scoring GEO/AEO
    dimensions by calling the OpenAI Chat Completions API with a structured
    prompt.

    The generator is designed to fail gracefully: if the API call fails,
    times out, or returns unparseable output, it returns an empty list and
    logs a warning rather than raising an exception.

    Attributes:
        settings: Application settings used for API key, model, and feature flags.
        client: The AsyncOpenAI client instance.

    Example::

        generator = SuggestionsGenerator()
        suggestions = await generator.generate(
            content="Your article...",
            analysis_result=result,
            max_dimensions=3,
        )
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the generator with optional custom settings.

        Args:
            settings: Optional ``Settings`` instance.  If ``None``, the
                global ``get_settings()`` singleton is used.
        """
        self.settings: Settings = settings if settings is not None else get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.openai_api_key or None,
            timeout=float(self.settings.openai_timeout),
        )

    def _select_dimensions_to_improve(
        self,
        analysis_result: AnalysisResult,
        max_dimensions: int,
    ) -> list[DimensionScore]:
        """Select the lowest-scoring dimensions that need improvement suggestions.

        Dimensions are already sorted by improvement_priority in the result
        (priority 1 = most urgent).  This method picks the top ``max_dimensions``
        from that sorted list.

        Args:
            analysis_result: The completed analysis result from the scorer.
            max_dimensions: Maximum number of dimensions to generate suggestions for.

        Returns:
            list[DimensionScore]: The selected dimensions, ordered by priority.
        """
        # Dimensions are sorted by improvement_priority ascending (1 = highest priority)
        sorted_dims = sorted(
            analysis_result.dimensions,
            key=lambda d: d.improvement_priority,
        )
        # Only suggest improvements for dimensions that aren't already excellent
        dims_needing_work = [
            d for d in sorted_dims
            if d.raw_score < 80.0
        ]
        return dims_needing_work[:max_dimensions]

    async def generate(
        self,
        content: str,
        analysis_result: AnalysisResult,
        max_dimensions: int = 3,
    ) -> tuple[list[SuggestionItem], str | None]:
        """Generate AI-powered rewrite suggestions for the lowest-scoring dimensions.

        Args:
            content: The original content text that was analyzed.
            analysis_result: The completed ``AnalysisResult`` from the scorer.
            max_dimensions: Maximum number of dimensions to generate suggestions
                for.  Defaults to 3 (the three lowest-scoring dimensions).

        Returns:
            tuple[list[SuggestionItem], str | None]: A tuple of
                (suggestions list, optional error message).  The error message
                is non-None when the call failed but the failure was handled
                gracefully.
        """
        if not self.settings.enable_ai_suggestions:
            logger.info("AI suggestions disabled by configuration (ENABLE_AI_SUGGESTIONS=false).")
            return [], None

        if not self.settings.openai_api_key:
            logger.warning(
                "OpenAI API key not configured (OPENAI_API_KEY is empty). "
                "Skipping AI suggestions."
            )
            return [], "AI suggestions unavailable: OpenAI API key not configured."

        dimensions_to_improve = self._select_dimensions_to_improve(
            analysis_result, max_dimensions
        )

        if not dimensions_to_improve:
            logger.info(
                "All dimensions scored >= 80; no suggestions needed."
            )
            return [], None

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            content=content,
            dimensions_to_improve=dimensions_to_improve,
            target_query=analysis_result.target_query,
        )

        logger.debug(
            "Requesting suggestions from OpenAI for %d dimension(s): %s",
            len(dimensions_to_improve),
            [d.dimension.value for d in dimensions_to_improve],
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.settings.openai_max_tokens,
                temperature=0.4,
                response_format={"type": "text"},
            )
        except APITimeoutError as exc:
            error_msg = f"AI suggestions timed out after {self.settings.openai_timeout}s."
            logger.warning("%s Details: %s", error_msg, exc)
            return [], error_msg
        except APIConnectionError as exc:
            error_msg = "AI suggestions unavailable: could not connect to OpenAI API."
            logger.warning("%s Details: %s", error_msg, exc)
            return [], error_msg
        except APIStatusError as exc:
            error_msg = f"AI suggestions failed: OpenAI API returned status {exc.status_code}."
            logger.warning("%s Details: %s", error_msg, exc)
            return [], error_msg
        except Exception as exc:  # noqa: BLE001
            error_msg = f"AI suggestions failed with an unexpected error: {type(exc).__name__}."
            logger.exception("%s", error_msg)
            return [], error_msg

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message.content is None:
            logger.warning("OpenAI returned an empty response for suggestions.")
            return [], "AI suggestions returned an empty response."

        response_text = choice.message.content
        logger.debug(
            "OpenAI suggestions response received (%d chars).",
            len(response_text),
        )

        suggestions = _parse_suggestions_response(response_text, dimensions_to_improve)

        if not suggestions:
            return [], "AI suggestions could not be parsed from the OpenAI response."

        logger.info(
            "Successfully generated %d suggestion(s) from OpenAI.",
            len(suggestions),
        )
        return suggestions, None


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


async def generate_suggestions(
    content: str,
    analysis_result: AnalysisResult,
    max_dimensions: int = 3,
    settings: Settings | None = None,
) -> tuple[list[SuggestionItem], str | None]:
    """Module-level convenience function for generating rewrite suggestions.

    Creates a ``SuggestionsGenerator`` and calls ``generate()`` in one step.
    For repeated use in a long-running server, prefer instantiating
    ``SuggestionsGenerator`` once and reusing it.

    Args:
        content: The original content text that was analyzed.
        analysis_result: The completed ``AnalysisResult`` from the scorer.
        max_dimensions: Maximum number of dimensions to target.  Defaults to 3.
        settings: Optional custom ``Settings``; defaults to the global singleton.

    Returns:
        tuple[list[SuggestionItem], str | None]: A tuple of
            (suggestions list, optional error message).
    """
    generator = SuggestionsGenerator(settings=settings)
    return await generator.generate(
        content=content,
        analysis_result=analysis_result,
        max_dimensions=max_dimensions,
    )
