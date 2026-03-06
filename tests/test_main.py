"""Integration tests for FastAPI routes using the TestClient.

Tests cover:
- GET / returns the main UI page (200 OK, HTML content)
- GET /health returns JSON health status
- POST /api/analyze with valid JSON returns AnalysisResult
- POST /api/analyze with invalid JSON returns 422
- POST /analyze (form submission) returns HTML partial
- POST /analyze with invalid form data returns HTML partial with error
- AI suggestions disabled path (no API key configured)
- Edge cases: empty content, content at minimum length boundary

Note: These tests mock the OpenAI API to avoid real network calls.
The spaCy model must be installed for integration tests to run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from geo_aeo_optimizer.main import app
from geo_aeo_optimizer.models import (
    AnalysisResult,
    DimensionKey,
    DimensionScore,
    ScoreLabel,
    Settings,
    SuggestionItem,
    get_settings,
)

# ---------------------------------------------------------------------------
# Test content samples
# ---------------------------------------------------------------------------

SAMPLE_CONTENT = """
Python is a high-level, general-purpose programming language created by Guido van Rossum.
According to the 2024 Stack Overflow Developer Survey, Python is the most popular language.
What is Python used for? Python is widely used in data science, machine learning,
web development, automation, and scientific computing.

- Data Science & Machine Learning
- Web Development with Django and FastAPI
- Automation and Scripting
- Scientific Computing at organizations like NASA and CERN
"""

MINIMAL_VALID_CONTENT = "a" * 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a synchronous TestClient for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def no_ai_settings() -> Settings:
    """Return settings with AI suggestions disabled (no API key)."""
    return Settings(openai_api_key="", enable_ai_suggestions=False)


# ---------------------------------------------------------------------------
# GET / - Main UI page
# ---------------------------------------------------------------------------


class TestIndexRoute:
    """Tests for the GET / route."""

    def test_index_returns_200(self, client: TestClient) -> None:
        """GET / should return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_returns_html(self, client: TestClient) -> None:
        """GET / should return HTML content type."""
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_index_contains_form(self, client: TestClient) -> None:
        """Main page should include an HTML form pointing to /analyze."""
        response = client.get("/")
        html = response.text
        assert "/analyze" in html

    def test_index_contains_textarea(self, client: TestClient) -> None:
        """Main page should include a textarea for content input."""
        response = client.get("/")
        assert "textarea" in response.text.lower()

    def test_index_contains_app_title(self, client: TestClient) -> None:
        """Main page should reference GEO, AEO, or optimizer branding."""
        response = client.get("/")
        html_lower = response.text.lower()
        assert any(word in html_lower for word in ["geo", "aeo", "optimizer", "content"])

    def test_index_contains_htmx_script(self, client: TestClient) -> None:
        """Main page should load the HTMX library."""
        response = client.get("/")
        assert "htmx" in response.text.lower()

    def test_index_response_body_not_empty(self, client: TestClient) -> None:
        """Main page should have a non-empty body."""
        response = client.get("/")
        assert len(response.text) > 100

    def test_index_contains_submit_button(self, client: TestClient) -> None:
        """Main page should have a submit button."""
        response = client.get("/")
        assert "submit" in response.text.lower() or "button" in response.text.lower()

    def test_index_contains_version(self, client: TestClient) -> None:
        """Main page should display the app version somewhere."""
        from geo_aeo_optimizer import __version__

        response = client.get("/")
        assert __version__ in response.text

    def test_index_contains_six_dimensions_reference(self, client: TestClient) -> None:
        """Main page should reference the six scoring dimensions."""
        response = client.get("/")
        html_lower = response.text.lower()
        # At least one dimension name should appear
        assert any(
            word in html_lower
            for word in ["entity", "citation", "clarity", "depth", "alignment", "formatting"]
        )


# ---------------------------------------------------------------------------
# GET /health - Health check
# ---------------------------------------------------------------------------


class TestHealthRoute:
    """Tests for the GET /health route."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client: TestClient) -> None:
        """GET /health should return JSON content type."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_has_status_ok(self, client: TestClient) -> None:
        """Health response should include status=ok."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_version(self, client: TestClient) -> None:
        """Health response should include a version string."""
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_components(self, client: TestClient) -> None:
        """Health response should include a components dictionary."""
        data = client.get("/health").json()
        assert "components" in data
        components = data["components"]
        assert "scorer" in components
        assert "ai_suggestions" in components
        assert "spacy_model" in components

    def test_health_scorer_status(self, client: TestClient) -> None:
        """Scorer component status should be ok or unavailable."""
        data = client.get("/health").json()
        assert data["components"]["scorer"] in ("ok", "unavailable")

    def test_health_has_openai_model(self, client: TestClient) -> None:
        """Health response should include openai_model in components."""
        data = client.get("/health").json()
        assert "openai_model" in data["components"]

    def test_health_spacy_model_value(self, client: TestClient) -> None:
        """Health response spacy_model should match configured value."""
        data = client.get("/health").json()
        settings = get_settings()
        assert data["components"]["spacy_model"] == settings.spacy_model

    def test_health_ai_suggestions_field_is_string(self, client: TestClient) -> None:
        """ai_suggestions component value should be a string."""
        data = client.get("/health").json()
        assert isinstance(data["components"]["ai_suggestions"], str)

    def test_health_version_matches_package_version(self, client: TestClient) -> None:
        """Health response version should match package __version__."""
        from geo_aeo_optimizer import __version__

        data = client.get("/health").json()
        assert data["version"] == __version__


# ---------------------------------------------------------------------------
# POST /api/analyze - JSON API endpoint
# ---------------------------------------------------------------------------


class TestApiAnalyzeRoute:
    """Tests for the POST /api/analyze JSON API endpoint."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        """Valid JSON request should return HTTP 200."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={
                    "content": SAMPLE_CONTENT,
                    "target_query": "What is Python?",
                    "include_suggestions": False,
                },
            )
        assert response.status_code == 200

    def test_valid_request_returns_analysis_result_schema(self, client: TestClient) -> None:
        """Valid request should return all required AnalysisResult fields."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert "composite_score" in data
        assert "composite_label" in data
        assert "dimensions" in data
        assert "suggestions" in data
        assert "content_word_count" in data
        assert "content_char_count" in data

    def test_composite_score_within_range(self, client: TestClient) -> None:
        """Composite score should be in [0, 100]."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert 0.0 <= data["composite_score"] <= 100.0

    def test_six_dimensions_returned(self, client: TestClient) -> None:
        """Response should contain exactly six dimension scores."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert len(data["dimensions"]) == 6

    def test_all_dimension_keys_present(self, client: TestClient) -> None:
        """All six dimension keys should be present in the response."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        returned_keys = {d["dimension"] for d in data["dimensions"]}
        expected_keys = {
            "qa_alignment",
            "entity_density",
            "structured_formatting",
            "citation_cues",
            "semantic_clarity",
            "content_depth",
        }
        assert returned_keys == expected_keys

    def test_target_query_echoed_in_response(self, client: TestClient) -> None:
        """Target query should be echoed back in the response."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={
                    "content": SAMPLE_CONTENT,
                    "target_query": "Python programming",
                    "include_suggestions": False,
                },
            )

        data = response.json()
        assert data["target_query"] == "Python programming"

    def test_content_too_short_returns_422(self, client: TestClient) -> None:
        """Content shorter than 50 chars should return 422."""
        response = client.post(
            "/api/analyze",
            json={"content": "short"},
        )
        assert response.status_code == 422

    def test_missing_content_field_returns_422(self, client: TestClient) -> None:
        """Request missing 'content' field should return 422."""
        response = client.post(
            "/api/analyze",
            json={"target_query": "something"},
        )
        assert response.status_code == 422

    def test_empty_content_returns_422(self, client: TestClient) -> None:
        """Empty string content should return 422."""
        response = client.post(
            "/api/analyze",
            json={"content": ""},
        )
        assert response.status_code == 422

    def test_content_word_count_positive(self, client: TestClient) -> None:
        """Word count and char count should be positive for valid content."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert data["content_word_count"] > 0
        assert data["content_char_count"] > 0

    def test_dimensions_sorted_by_priority(self, client: TestClient) -> None:
        """Dimensions should be sorted by improvement_priority ascending."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        priorities = [d["improvement_priority"] for d in data["dimensions"]]
        assert priorities == sorted(priorities)

    def test_no_suggestions_when_disabled(self, client: TestClient) -> None:
        """When include_suggestions=False, suggestions should be empty."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert data["suggestions"] == []

    def test_suggestions_included_when_requested_and_mocked(self, client: TestClient) -> None:
        """When suggestions are requested and mocked, they should appear in response."""
        mock_suggestion = SuggestionItem(
            dimension=DimensionKey.CITATION_CUES,
            display_name="Citation Cues",
            issue="No citations found.",
            suggestion="Add statistical references.",
            before_example="Python is popular.",
            after_example="According to Stack Overflow 2024, Python is most popular.",
        )

        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([mock_suggestion], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={
                    "content": SAMPLE_CONTENT,
                    "include_suggestions": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["dimension"] == "citation_cues"

    def test_error_message_none_on_success(self, client: TestClient) -> None:
        """error_message should be None when analysis succeeds without issues."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert data["error_message"] is None

    def test_error_message_set_when_ai_fails(self, client: TestClient) -> None:
        """Non-fatal AI failure should set error_message but still return 200."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(
                return_value=([], "AI suggestions timed out.")
            )
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={
                    "content": SAMPLE_CONTENT,
                    "include_suggestions": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error_message"] is not None
        assert "timed out" in data["error_message"]

    def test_minimal_valid_content_accepted(self, client: TestClient) -> None:
        """Exactly 50 characters of content should be accepted."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": MINIMAL_VALID_CONTENT, "include_suggestions": False},
            )

        assert response.status_code == 200

    def test_composite_label_is_valid_string(self, client: TestClient) -> None:
        """Composite label should be one of the valid ScoreLabel values."""
        valid_labels = {label.value for label in ScoreLabel}
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert data["composite_label"] in valid_labels

    def test_each_dimension_has_required_fields(self, client: TestClient) -> None:
        """Each dimension in the response should have all required fields."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        required_fields = {
            "dimension", "display_name", "raw_score", "weight",
            "weighted_score", "label", "explanation", "improvement_priority"
        }
        for dim in data["dimensions"]:
            for field in required_fields:
                assert field in dim, f"Missing field '{field}' in dimension {dim.get('dimension')}"

    def test_each_dimension_score_in_range(self, client: TestClient) -> None:
        """Each dimension raw_score should be in [0, 100]."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        for dim in data["dimensions"]:
            assert 0.0 <= dim["raw_score"] <= 100.0

    def test_target_query_none_when_not_provided(self, client: TestClient) -> None:
        """target_query should be null when not provided."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        assert data["target_query"] is None

    def test_all_priorities_are_positive_integers(self, client: TestClient) -> None:
        """All improvement_priority values should be positive integers."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": False},
            )

        data = response.json()
        for dim in data["dimensions"]:
            assert isinstance(dim["improvement_priority"], int)
            assert dim["improvement_priority"] >= 1

    def test_whitespace_only_content_returns_422(self, client: TestClient) -> None:
        """Content that is only whitespace should return 422."""
        response = client.post(
            "/api/analyze",
            json={"content": "   " * 20},
        )
        assert response.status_code == 422

    def test_multiple_suggestions_returned(self, client: TestClient) -> None:
        """Multiple mocked suggestions should all be returned."""
        mock_suggestions = [
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
            SuggestionItem(
                dimension=DimensionKey.ENTITY_DENSITY,
                display_name="Entity Density",
                issue="Low entity count.",
                suggestion="Add named entities.",
            ),
        ]

        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=(mock_suggestions, None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": True},
            )

        data = response.json()
        assert len(data["suggestions"]) == 3

    def test_suggestion_fields_present(self, client: TestClient) -> None:
        """Suggestion items should have all required fields."""
        mock_suggestion = SuggestionItem(
            dimension=DimensionKey.CONTENT_DEPTH,
            display_name="Content Depth",
            issue="Content too short.",
            suggestion="Expand sections.",
            before_example="Short text.",
            after_example="Expanded text with details.",
        )

        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([mock_suggestion], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/api/analyze",
                json={"content": SAMPLE_CONTENT, "include_suggestions": True},
            )

        data = response.json()
        sugg = data["suggestions"][0]
        assert "dimension" in sugg
        assert "display_name" in sugg
        assert "issue" in sugg
        assert "suggestion" in sugg
        assert sugg["before_example"] == "Short text."
        assert sugg["after_example"] == "Expanded text with details."


# ---------------------------------------------------------------------------
# POST /analyze - Form-based HTMX endpoint
# ---------------------------------------------------------------------------


class TestAnalyzeFormRoute:
    """Tests for the POST /analyze form-based HTMX route."""

    def test_valid_form_returns_200(self, client: TestClient) -> None:
        """Valid form submission should return HTTP 200."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        assert response.status_code == 200

    def test_valid_form_returns_html(self, client: TestClient) -> None:
        """Form submission should return HTML content."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        assert "text/html" in response.headers["content-type"]

    def test_form_result_contains_score(self, client: TestClient) -> None:
        """Results partial should contain score-related content."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        html_lower = response.text.lower()
        assert any(
            word in html_lower
            for word in ["score", "dimension", "result", "composite", "%"]
        )

    def test_short_content_returns_error_html(self, client: TestClient) -> None:
        """Too-short content should return 422 with HTML error content."""
        response = client.post(
            "/analyze",
            data={"content": "too short"},
        )
        assert response.status_code == 422
        assert "text/html" in response.headers["content-type"]

    def test_form_with_target_query(self, client: TestClient) -> None:
        """Form with target query should succeed."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={
                    "content": SAMPLE_CONTENT,
                    "target_query": "What is Python?",
                },
            )

        assert response.status_code == 200

    def test_form_with_suggestions_checkbox_on(self, client: TestClient) -> None:
        """Form with suggestions checkbox 'on' should succeed."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={
                    "content": SAMPLE_CONTENT,
                    "include_suggestions": "on",
                },
            )

        assert response.status_code == 200

    def test_form_with_suggestions_checkbox_off(self, client: TestClient) -> None:
        """Form without suggestions checkbox should succeed."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        assert response.status_code == 200

    def test_empty_content_form_returns_error(self, client: TestClient) -> None:
        """Empty content form submission should return 4xx error."""
        response = client.post(
            "/analyze",
            data={"content": ""},
        )
        assert response.status_code in (400, 422)

    def test_form_result_is_partial_html(self, client: TestClient) -> None:
        """Results partial should not be a full HTML document (no DOCTYPE)."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        # The partial template should not start with a full DOCTYPE
        assert "<!DOCTYPE" not in response.text[:50]

    def test_form_error_html_contains_error_indicator(self, client: TestClient) -> None:
        """Error HTML should contain some error indication."""
        response = client.post(
            "/analyze",
            data={"content": "short"},
        )
        html_lower = response.text.lower()
        assert any(
            word in html_lower
            for word in ["error", "invalid", "validation", "short", "minimum", "characters"]
        )

    def test_form_with_minimal_valid_content(self, client: TestClient) -> None:
        """Form submission with exactly 50 chars should succeed."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": MINIMAL_VALID_CONTENT},
            )

        assert response.status_code == 200

    def test_form_ai_error_still_returns_200(self, client: TestClient) -> None:
        """Non-fatal AI failure during form analysis should still return 200."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(
                return_value=([], "OpenAI connection failed.")
            )
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={
                    "content": SAMPLE_CONTENT,
                    "include_suggestions": "on",
                },
            )

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_form_result_contains_dimension_names(self, client: TestClient) -> None:
        """Results partial should reference at least one dimension name."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        html_lower = response.text.lower()
        dimension_names = [
            "alignment", "entity", "formatting", "citation", "clarity", "depth"
        ]
        assert any(name in html_lower for name in dimension_names)


# ---------------------------------------------------------------------------
# 404 handler
# ---------------------------------------------------------------------------


class TestNotFoundHandler:
    """Tests for the 404 error handler."""

    def test_unknown_route_returns_404(self, client: TestClient) -> None:
        """Unknown routes should return 404."""
        response = client.get("/nonexistent-route-xyz")
        assert response.status_code == 404

    def test_404_response_is_html(self, client: TestClient) -> None:
        """404 response should be HTML."""
        response = client.get("/nonexistent-route-xyz")
        assert "text/html" in response.headers["content-type"]

    def test_404_response_contains_link(self, client: TestClient) -> None:
        """404 response should contain a link back to home."""
        response = client.get("/nonexistent-route-xyz")
        assert "/" in response.text

    def test_404_response_mentions_not_found(self, client: TestClient) -> None:
        """404 response body should reference the error state."""
        response = client.get("/nonexistent-route-xyz")
        html_lower = response.text.lower()
        assert any(word in html_lower for word in ["404", "not found", "page"])

    def test_unknown_post_route_returns_404_or_405(self, client: TestClient) -> None:
        """POST to an unknown route should return 404 or 405."""
        response = client.post("/nonexistent-route-xyz", data={"test": "data"})
        assert response.status_code in (404, 405)


# ---------------------------------------------------------------------------
# Score color/bg/bar filter tests
# ---------------------------------------------------------------------------


class TestJinja2Filters:
    """Tests for the custom Jinja2 template filters."""

    def test_score_color_excellent(self) -> None:
        """Score >= 80 should return green color class."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(80.0) == "text-green-600"
        assert _score_color_class(100.0) == "text-green-600"

    def test_score_color_good(self) -> None:
        """Score 60-79.9 should return blue color class."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(60.0) == "text-blue-600"
        assert _score_color_class(79.9) == "text-blue-600"

    def test_score_color_fair(self) -> None:
        """Score 40-59.9 should return yellow color class."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(40.0) == "text-yellow-600"
        assert _score_color_class(59.9) == "text-yellow-600"

    def test_score_color_poor(self) -> None:
        """Score 20-39.9 should return orange color class."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(20.0) == "text-orange-600"
        assert _score_color_class(39.9) == "text-orange-600"

    def test_score_color_critical(self) -> None:
        """Score < 20 should return red color class."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(0.0) == "text-red-600"
        assert _score_color_class(19.9) == "text-red-600"

    def test_score_bg_excellent(self) -> None:
        """Score >= 80 should return green background class."""
        from geo_aeo_optimizer.main import _score_bg_class

        assert _score_bg_class(85.0) == "bg-green-100"

    def test_score_bg_good(self) -> None:
        """Score 60-79.9 should return blue background class."""
        from geo_aeo_optimizer.main import _score_bg_class

        assert _score_bg_class(65.0) == "bg-blue-100"

    def test_score_bg_fair(self) -> None:
        """Score 40-59.9 should return yellow background class."""
        from geo_aeo_optimizer.main import _score_bg_class

        assert _score_bg_class(50.0) == "bg-yellow-100"

    def test_score_bg_poor(self) -> None:
        """Score 20-39.9 should return orange background class."""
        from geo_aeo_optimizer.main import _score_bg_class

        assert _score_bg_class(30.0) == "bg-orange-100"

    def test_score_bg_critical(self) -> None:
        """Score < 20 should return red background class."""
        from geo_aeo_optimizer.main import _score_bg_class

        assert _score_bg_class(10.0) == "bg-red-100"

    def test_score_bar_excellent(self) -> None:
        """Score >= 80 should return green bar class."""
        from geo_aeo_optimizer.main import _score_bar_class

        assert _score_bar_class(90.0) == "bg-green-500"

    def test_score_bar_good(self) -> None:
        """Score 60-79.9 should return blue bar class."""
        from geo_aeo_optimizer.main import _score_bar_class

        assert _score_bar_class(70.0) == "bg-blue-500"

    def test_score_bar_fair(self) -> None:
        """Score 40-59.9 should return yellow bar class."""
        from geo_aeo_optimizer.main import _score_bar_class

        assert _score_bar_class(45.0) == "bg-yellow-500"

    def test_score_bar_poor(self) -> None:
        """Score 20-39.9 should return orange bar class."""
        from geo_aeo_optimizer.main import _score_bar_class

        assert _score_bar_class(25.0) == "bg-orange-500"

    def test_score_bar_critical(self) -> None:
        """Score < 20 should return red bar class."""
        from geo_aeo_optimizer.main import _score_bar_class

        assert _score_bar_class(5.0) == "bg-red-500"

    def test_score_color_boundary_80(self) -> None:
        """Exactly 80 should be Excellent (green)."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(80.0) == "text-green-600"

    def test_score_color_boundary_60(self) -> None:
        """Exactly 60 should be Good (blue)."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(60.0) == "text-blue-600"

    def test_score_color_boundary_40(self) -> None:
        """Exactly 40 should be Fair (yellow)."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(40.0) == "text-yellow-600"

    def test_score_color_boundary_20(self) -> None:
        """Exactly 20 should be Poor (orange)."""
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(20.0) == "text-orange-600"

    def test_score_color_returns_string(self) -> None:
        """All filter functions should return strings."""
        from geo_aeo_optimizer.main import (
            _score_bar_class,
            _score_bg_class,
            _score_color_class,
        )

        for score in [0.0, 19.9, 20.0, 39.9, 40.0, 59.9, 60.0, 79.9, 80.0, 100.0]:
            assert isinstance(_score_color_class(score), str)
            assert isinstance(_score_bg_class(score), str)
            assert isinstance(_score_bar_class(score), str)


# ---------------------------------------------------------------------------
# Template rendering tests
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    """Tests for Jinja2 template rendering correctness."""

    def test_results_partial_renders_without_result(self, client: TestClient) -> None:
        """Results partial should render error state gracefully."""
        response = client.post(
            "/analyze",
            data={"content": "too short"},
        )
        assert response.status_code == 422
        # Should render the error partial, not crash
        assert len(response.text) > 0

    def test_results_partial_contains_dimension_breakdown_heading(self, client: TestClient) -> None:
        """Results should include a dimension breakdown section."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        html_lower = response.text.lower()
        assert "dimension" in html_lower or "breakdown" in html_lower

    def test_results_partial_shows_suggestions_when_available(self, client: TestClient) -> None:
        """Results partial should render suggestion content when suggestions are available."""
        mock_suggestion = SuggestionItem(
            dimension=DimensionKey.CITATION_CUES,
            display_name="Citation Cues",
            issue="No citations found.",
            suggestion="Add statistical references like percentages and study findings.",
            before_example="Python is popular.",
            after_example="According to Stack Overflow 2024, Python is the most popular language.",
        )

        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([mock_suggestion], None))
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={
                    "content": SAMPLE_CONTENT,
                    "include_suggestions": "on",
                },
            )

        html_lower = response.text.lower()
        assert "suggestion" in html_lower or "recommendation" in html_lower or "rewrite" in html_lower

    def test_results_partial_shows_warning_when_error_message(self, client: TestClient) -> None:
        """Non-fatal error messages should appear in the results partial."""
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(
                return_value=([], "AI suggestions timed out after 30s.")
            )
            mock_gen_fn.return_value = mock_gen

            response = client.post(
                "/analyze",
                data={
                    "content": SAMPLE_CONTENT,
                    "include_suggestions": "on",
                },
            )

        html_lower = response.text.lower()
        assert "timed out" in html_lower or "note" in html_lower or "warning" in html_lower
