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
        response = client.get("/")
        assert response.status_code == 200

    def test_index_returns_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_index_contains_form(self, client: TestClient) -> None:
        response = client.get("/")
        html = response.text
        # The form should have an action pointing to /analyze
        assert "/analyze" in html

    def test_index_contains_textarea(self, client: TestClient) -> None:
        response = client.get("/")
        assert "textarea" in response.text.lower()

    def test_index_contains_app_title(self, client: TestClient) -> None:
        response = client.get("/")
        # Should mention GEO, AEO, or Optimizer in some form
        html_lower = response.text.lower()
        assert any(word in html_lower for word in ["geo", "aeo", "optimizer", "content"])


# ---------------------------------------------------------------------------
# GET /health - Health check
# ---------------------------------------------------------------------------


class TestHealthRoute:
    """Tests for the GET /health route."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client: TestClient) -> None:
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_has_status_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_version(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_components(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "components" in data
        components = data["components"]
        assert "scorer" in components
        assert "ai_suggestions" in components
        assert "spacy_model" in components

    def test_health_scorer_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        # Scorer should be ok since startup loads it
        assert data["components"]["scorer"] in ("ok", "unavailable")


# ---------------------------------------------------------------------------
# POST /api/analyze - JSON API endpoint
# ---------------------------------------------------------------------------


class TestApiAnalyzeRoute:
    """Tests for the POST /api/analyze JSON API endpoint."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
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
        response = client.post(
            "/api/analyze",
            json={"content": "short"},
        )
        assert response.status_code == 422

    def test_missing_content_field_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/analyze",
            json={"target_query": "something"},
        )
        assert response.status_code == 422

    def test_empty_content_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/analyze",
            json={"content": ""},
        )
        assert response.status_code == 422

    def test_content_word_count_positive(self, client: TestClient) -> None:
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

        assert response.status_code == 200  # Non-fatal error, still 200
        data = response.json()
        assert data["error_message"] is not None
        assert "timed out" in data["error_message"]

    def test_minimal_valid_content_accepted(self, client: TestClient) -> None:
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


# ---------------------------------------------------------------------------
# POST /analyze - Form-based HTMX endpoint
# ---------------------------------------------------------------------------


class TestAnalyzeFormRoute:
    """Tests for the POST /analyze form-based HTMX route."""

    def test_valid_form_returns_200(self, client: TestClient) -> None:
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

        # The results partial should contain score-related content
        html_lower = response.text.lower()
        assert any(
            word in html_lower
            for word in ["score", "dimension", "result", "composite", "%"]
        )

    def test_short_content_returns_error_html(self, client: TestClient) -> None:
        response = client.post(
            "/analyze",
            data={"content": "too short"},
        )
        # Should return HTML with an error message (422)
        assert response.status_code == 422
        assert "text/html" in response.headers["content-type"]

    def test_form_with_target_query(self, client: TestClient) -> None:
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
        with patch(
            "geo_aeo_optimizer.main._get_suggestions_generator",
        ) as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=([], None))
            mock_gen_fn.return_value = mock_gen

            # No include_suggestions in form data (checkbox unchecked)
            response = client.post(
                "/analyze",
                data={"content": SAMPLE_CONTENT},
            )

        assert response.status_code == 200

    def test_empty_content_form_returns_error(self, client: TestClient) -> None:
        response = client.post(
            "/analyze",
            data={"content": ""},
        )
        assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 404 handler
# ---------------------------------------------------------------------------


class TestNotFoundHandler:
    """Tests for the 404 error handler."""

    def test_unknown_route_returns_404(self, client: TestClient) -> None:
        response = client.get("/nonexistent-route-xyz")
        assert response.status_code == 404

    def test_404_response_is_html(self, client: TestClient) -> None:
        response = client.get("/nonexistent-route-xyz")
        assert "text/html" in response.headers["content-type"]

    def test_404_response_contains_link(self, client: TestClient) -> None:
        response = client.get("/nonexistent-route-xyz")
        assert "/" in response.text  # Link back to home


# ---------------------------------------------------------------------------
# Score color/bg/bar filter tests
# ---------------------------------------------------------------------------


class TestJinja2Filters:
    """Tests for the custom Jinja2 template filters."""

    def test_score_color_excellent(self) -> None:
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(80.0) == "text-green-600"
        assert _score_color_class(100.0) == "text-green-600"

    def test_score_color_good(self) -> None:
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(60.0) == "text-blue-600"
        assert _score_color_class(79.9) == "text-blue-600"

    def test_score_color_fair(self) -> None:
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(40.0) == "text-yellow-600"
        assert _score_color_class(59.9) == "text-yellow-600"

    def test_score_color_poor(self) -> None:
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(20.0) == "text-orange-600"
        assert _score_color_class(39.9) == "text-orange-600"

    def test_score_color_critical(self) -> None:
        from geo_aeo_optimizer.main import _score_color_class

        assert _score_color_class(0.0) == "text-red-600"
        assert _score_color_class(19.9) == "text-red-600"

    def test_score_bg_excellent(self) -> None:
        from geo_aeo_optimizer.main import _score_bg_class

        assert _score_bg_class(85.0) == "bg-green-100"

    def test_score_bg_critical(self) -> None:
        from geo_aeo_optimizer.main import _score_bg_class

        assert _score_bg_class(10.0) == "bg-red-100"

    def test_score_bar_excellent(self) -> None:
        from geo_aeo_optimizer.main import _score_bar_class

        assert _score_bar_class(90.0) == "bg-green-500"

    def test_score_bar_critical(self) -> None:
        from geo_aeo_optimizer.main import _score_bar_class

        assert _score_bar_class(5.0) == "bg-red-500"
