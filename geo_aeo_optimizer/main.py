"""FastAPI application entrypoint for the GEO/AEO Content Optimizer.

This module defines the FastAPI application instance, all HTTP routes,
Jinja2 template rendering, startup/shutdown lifecycle events, and error
handlers.

Routes:
    GET  /              - Main single-page UI (renders index.html template)
    POST /analyze       - Accept form submission, run scoring + suggestions,
                          return HTMX partial (results_partial.html)
    GET  /health        - Simple health check endpoint returning JSON
    POST /api/analyze   - JSON API endpoint for programmatic access

Startup events:
    - Load spaCy model eagerly to avoid cold-start latency on first request
    - Validate OpenAI API key presence and log a warning if missing
    - Log all active configuration values (excluding the API key value)

Typical usage::

    uvicorn geo_aeo_optimizer.main:app --reload

Or via the installed CLI entry point::

    geo-aeo-optimizer
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncGenerator

from fastapi import FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from geo_aeo_optimizer import __version__
from geo_aeo_optimizer.models import (
    AnalysisResult,
    ContentInput,
    Settings,
    get_settings,
)
from geo_aeo_optimizer.scorer import ContentScorer, _load_spacy_model
from geo_aeo_optimizer.suggestions import SuggestionsGenerator

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template directory resolution
# ---------------------------------------------------------------------------

import pathlib

_PACKAGE_DIR = pathlib.Path(__file__).parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"

# ---------------------------------------------------------------------------
# Application-level singletons (initialised during startup)
# ---------------------------------------------------------------------------

_scorer: ContentScorer | None = None
_suggestions_generator: SuggestionsGenerator | None = None


def _get_scorer() -> ContentScorer:
    """Return the application-level ``ContentScorer`` singleton.

    Returns:
        ContentScorer: The initialised scorer.

    Raises:
        RuntimeError: If the scorer has not been initialised (startup failed).
    """
    if _scorer is None:
        raise RuntimeError(
            "ContentScorer has not been initialised. "
            "The application startup event may not have completed."
        )
    return _scorer


def _get_suggestions_generator() -> SuggestionsGenerator:
    """Return the application-level ``SuggestionsGenerator`` singleton.

    Returns:
        SuggestionsGenerator: The initialised suggestions generator.

    Raises:
        RuntimeError: If the generator has not been initialised.
    """
    if _suggestions_generator is None:
        raise RuntimeError(
            "SuggestionsGenerator has not been initialised. "
            "The application startup event may not have completed."
        )
    return _suggestions_generator


# ---------------------------------------------------------------------------
# Lifespan context manager (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager handling startup and shutdown events.

    Startup:
        - Load configuration from environment / .env file
        - Eagerly load the spaCy model to warm up the cache
        - Initialise the ContentScorer and SuggestionsGenerator singletons
        - Log configuration summary

    Shutdown:
        - Log shutdown message

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded to FastAPI during the application lifetime.
    """
    global _scorer, _suggestions_generator  # noqa: PLW0603

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #
    settings = get_settings()

    logger.info(
        "Starting GEO/AEO Content Optimizer v%s on %s:%d",
        __version__,
        settings.app_host,
        settings.app_port,
    )
    logger.info("Log level: %s", settings.log_level)
    logger.info("spaCy model: %s", settings.spacy_model)
    logger.info("OpenAI model: %s", settings.openai_model)
    logger.info(
        "AI suggestions enabled: %s",
        settings.enable_ai_suggestions,
    )
    logger.info(
        "Scoring weights: QA=%.1f, Entity=%.1f, Format=%.1f, "
        "Citation=%.1f, Clarity=%.1f, Depth=%.1f",
        settings.weight_qa_alignment,
        settings.weight_entity_density,
        settings.weight_structured_formatting,
        settings.weight_citation_cues,
        settings.weight_semantic_clarity,
        settings.weight_content_depth,
    )

    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. AI-powered suggestions will be disabled. "
            "Set OPENAI_API_KEY in your .env file to enable them."
        )

    # Eagerly load spaCy model to warm the cache
    try:
        _load_spacy_model(settings.spacy_model)
        logger.info("spaCy model '%s' loaded and cached.", settings.spacy_model)
    except OSError as exc:
        logger.error(
            "Failed to load spaCy model '%s': %s. "
            "Run: python -m spacy download %s",
            settings.spacy_model,
            exc,
            settings.spacy_model,
        )
        # Do not abort startup; let individual requests fail with a clear error

    # Initialise application singletons
    _scorer = ContentScorer(settings=settings)
    _suggestions_generator = SuggestionsGenerator(settings=settings)

    logger.info("Application startup complete. Ready to serve requests.")

    yield

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    logger.info("GEO/AEO Content Optimizer shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GEO/AEO Content Optimizer",
    description=(
        "Analyze and score marketing or editorial content for Generative Engine "
        "Optimization (GEO) and Answer Engine Optimization (AEO) — measuring how "
        "likely it is to be surfaced or cited by AI assistants."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Custom Jinja2 filters
# ---------------------------------------------------------------------------


def _score_color_class(score: float) -> str:
    """Return a Tailwind CSS color class based on a score value.

    Args:
        score: Numeric score in the range [0, 100].

    Returns:
        str: A Tailwind CSS text color class string.
    """
    if score >= 80:
        return "text-green-600"
    if score >= 60:
        return "text-blue-600"
    if score >= 40:
        return "text-yellow-600"
    if score >= 20:
        return "text-orange-600"
    return "text-red-600"


def _score_bg_class(score: float) -> str:
    """Return a Tailwind CSS background color class based on a score value.

    Args:
        score: Numeric score in the range [0, 100].

    Returns:
        str: A Tailwind CSS background color class string.
    """
    if score >= 80:
        return "bg-green-100"
    if score >= 60:
        return "bg-blue-100"
    if score >= 40:
        return "bg-yellow-100"
    if score >= 20:
        return "bg-orange-100"
    return "bg-red-100"


def _score_bar_class(score: float) -> str:
    """Return a Tailwind CSS bar fill color class based on a score value.

    Args:
        score: Numeric score in the range [0, 100].

    Returns:
        str: A Tailwind CSS background color class for progress bars.
    """
    if score >= 80:
        return "bg-green-500"
    if score >= 60:
        return "bg-blue-500"
    if score >= 40:
        return "bg-yellow-500"
    if score >= 20:
        return "bg-orange-500"
    return "bg-red-500"


# Register custom filters on the Jinja2 environment
templates.env.filters["score_color"] = _score_color_class
templates.env.filters["score_bg"] = _score_bg_class
templates.env.filters["score_bar"] = _score_bar_class


# ---------------------------------------------------------------------------
# Helper: run full analysis pipeline
# ---------------------------------------------------------------------------


async def _run_analysis(content_input: ContentInput) -> AnalysisResult:
    """Run the complete GEO/AEO analysis pipeline for a given input.

    Executes heuristic scoring synchronously and then (if enabled) calls
    the OpenAI suggestions generator asynchronously.

    Args:
        content_input: Validated content input model.

    Returns:
        AnalysisResult: Fully populated analysis result including AI suggestions
            (if enabled and successful) and any non-fatal error messages.
    """
    scorer = _get_scorer()
    suggestions_gen = _get_suggestions_generator()
    settings = get_settings()

    start_time = time.perf_counter()

    # Step 1: Heuristic scoring (synchronous, deterministic)
    try:
        result = scorer.score_from_input(content_input)
    except Exception as exc:
        logger.exception("Heuristic scoring failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content scoring failed: {exc}",
        ) from exc

    scoring_elapsed = time.perf_counter() - start_time
    logger.info(
        "Heuristic scoring completed in %.3fs. Composite score: %.1f",
        scoring_elapsed,
        result.composite_score,
    )

    # Step 2: AI suggestions (async, optional, graceful failure)
    error_message: str | None = None
    suggestions = []

    if content_input.include_suggestions and settings.enable_ai_suggestions:
        suggestions_start = time.perf_counter()
        try:
            suggestions, error_message = await suggestions_gen.generate(
                content=content_input.content,
                analysis_result=result,
                max_dimensions=3,
            )
        except Exception as exc:  # noqa: BLE001
            error_message = f"AI suggestions failed unexpectedly: {type(exc).__name__}"
            logger.exception("Unexpected error generating suggestions: %s", exc)

        suggestions_elapsed = time.perf_counter() - suggestions_start
        logger.info(
            "Suggestions generation completed in %.3fs. "
            "%d suggestion(s) generated. Error: %s",
            suggestions_elapsed,
            len(suggestions),
            error_message,
        )

    # Rebuild result with suggestions and any error message
    final_result = AnalysisResult.build(
        dimensions=list(result.dimensions),
        settings=settings,
        suggestions=suggestions,
        target_query=result.target_query,
        content_word_count=result.content_word_count,
        content_char_count=result.content_char_count,
        error_message=error_message,
    )

    total_elapsed = time.perf_counter() - start_time
    logger.info("Full analysis pipeline completed in %.3fs.", total_elapsed)

    return final_result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    """Render the main single-page UI.

    Args:
        request: The incoming HTTP request (required by Jinja2 templates).

    Returns:
        HTMLResponse: The rendered index.html template.
    """
    settings = get_settings()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "version": __version__,
            "ai_suggestions_enabled": settings.enable_ai_suggestions
            and bool(settings.openai_api_key),
            "max_content_length": settings.max_content_length,
        },
    )


@app.post("/analyze", response_class=HTMLResponse, include_in_schema=False)
async def analyze_form(
    request: Request,
    content: Annotated[str, Form()],
    target_query: Annotated[str | None, Form()] = None,
    include_suggestions: Annotated[str | None, Form()] = None,
) -> HTMLResponse:
    """Handle form-based content analysis requests from the HTMX UI.

    Accepts ``application/x-www-form-urlencoded`` POST data from the HTML
    form, runs the full analysis pipeline, and returns the
    ``results_partial.html`` HTMX partial template.

    Args:
        request: The incoming HTTP request.
        content: The content text submitted via the form.
        target_query: Optional target keyword/query from the form.
        include_suggestions: Optional checkbox value ("on" when checked).

    Returns:
        HTMLResponse: The rendered ``results_partial.html`` partial, suitable
            for HTMX to swap into the page.
    """
    # Convert form checkbox value to boolean
    include_suggestions_bool = include_suggestions in ("on", "true", "1", "yes")

    # Validate and build the ContentInput model
    try:
        content_input = ContentInput(
            content=content,
            target_query=target_query if target_query else None,
            include_suggestions=include_suggestions_bool,
        )
    except ValidationError as exc:
        errors = exc.errors()
        error_messages = [
            f"{e['loc'][0] if e['loc'] else 'input'}: {e['msg']}" for e in errors
        ]
        return templates.TemplateResponse(
            "results_partial.html",
            {
                "request": request,
                "error": "Validation error: " + "; ".join(error_messages),
                "result": None,
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    try:
        result = await _run_analysis(content_input)
    except HTTPException as exc:
        return templates.TemplateResponse(
            "results_partial.html",
            {
                "request": request,
                "error": exc.detail,
                "result": None,
            },
            status_code=exc.status_code,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in analyze_form: %s", exc)
        return templates.TemplateResponse(
            "results_partial.html",
            {
                "request": request,
                "error": "An unexpected error occurred. Please try again.",
                "result": None,
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return templates.TemplateResponse(
        "results_partial.html",
        {
            "request": request,
            "result": result,
            "error": None,
        },
    )


@app.post(
    "/api/analyze",
    response_model=AnalysisResult,
    summary="Analyze content for GEO/AEO scores",
    description=(
        "Submit content for heuristic GEO/AEO scoring across six dimensions. "
        "Optionally includes AI-powered rewrite suggestions for the lowest-scoring "
        "dimensions when ``include_suggestions`` is ``true`` and the OpenAI API key "
        "is configured."
    ),
    responses={
        200: {"description": "Analysis completed successfully."},
        422: {"description": "Validation error in request body."},
        500: {"description": "Internal server error during scoring."},
    },
)
async def analyze_api(content_input: ContentInput) -> AnalysisResult:
    """JSON API endpoint for programmatic GEO/AEO content analysis.

    Accepts a JSON request body matching the ``ContentInput`` schema and
    returns a full ``AnalysisResult`` including composite score, per-dimension
    breakdowns with improvement priorities, and optional AI suggestions.

    Args:
        content_input: Validated request body parsed from JSON.

    Returns:
        AnalysisResult: The complete analysis result.

    Raises:
        HTTPException: 500 if heuristic scoring fails unexpectedly.
    """
    return await _run_analysis(content_input)


@app.get(
    "/health",
    summary="Health check",
    description="Returns application health status and version information.",
    response_class=JSONResponse,
)
async def health_check() -> dict[str, Any]:
    """Simple health check endpoint.

    Returns:
        dict: A JSON object with status, version, and component availability.
    """
    settings = get_settings()
    spacy_ok = _scorer is not None
    ai_configured = bool(settings.openai_api_key) and settings.enable_ai_suggestions

    return {
        "status": "ok",
        "version": __version__,
        "components": {
            "scorer": "ok" if spacy_ok else "unavailable",
            "ai_suggestions": "configured" if ai_configured else "disabled",
            "spacy_model": settings.spacy_model,
            "openai_model": settings.openai_model,
        },
    }


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> HTMLResponse:
    """Return a simple HTML 404 response.

    Args:
        request: The incoming HTTP request.
        exc: The caught exception.

    Returns:
        HTMLResponse: A minimal 404 HTML page.
    """
    return HTMLResponse(
        content=(
            "<html><body>"
            "<h1>404 — Page Not Found</h1>"
            "<p><a href='/'>Return to the optimizer</a></p>"
            "</body></html>"
        ),
        status_code=404,
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> HTMLResponse:
    """Return a simple HTML 500 response.

    Args:
        request: The incoming HTTP request.
        exc: The caught exception.

    Returns:
        HTMLResponse: A minimal 500 HTML page.
    """
    logger.exception("Unhandled internal server error: %s", exc)
    return HTMLResponse(
        content=(
            "<html><body>"
            "<h1>500 — Internal Server Error</h1>"
            "<p>Something went wrong. Please try again later.</p>"
            "<p><a href='/'>Return to the optimizer</a></p>"
            "</body></html>"
        ),
        status_code=500,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """CLI entry point for starting the Uvicorn server.

    This function is registered as the ``geo-aeo-optimizer`` console script
    in ``pyproject.toml``.  It reads host, port, log level, and reload
    settings from the application ``Settings``.
    """
    import uvicorn

    settings = get_settings()

    log_level = settings.log_level.lower()

    uvicorn.run(
        "geo_aeo_optimizer.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run()
