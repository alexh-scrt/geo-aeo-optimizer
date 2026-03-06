# GEO/AEO Content Optimizer

> Score your content for AI discoverability — before the AI decides to ignore it.

The GEO/AEO Content Optimizer analyzes marketing and editorial content and scores it across six dimensions that determine how likely it is to be surfaced or cited by AI assistants like ChatGPT or Gemini. It combines a fast, deterministic heuristic engine with optional OpenAI-powered rewrite suggestions to deliver a composite score and concrete, prioritized improvements. Paste your content, get your score, and know exactly what to fix.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-org/geo_aeo_optimizer.git
cd geo_aeo_optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy language model
python -m spacy download en_core_web_sm

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start the server
uvicorn geo_aeo_optimizer.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser, paste your content, and click **Analyze**.

> **Note:** An OpenAI API key is optional. Without one, you still get full heuristic scores — only the AI-generated rewrite suggestions are skipped.

---

## Features

- **Six-dimension heuristic scoring** — Evaluates question-answer alignment, entity density, structured formatting, citation cues, semantic clarity, and content depth using spaCy. No LLM call required, so scoring is fast, free, and fully deterministic.
- **AI-powered rewrite suggestions** — When an OpenAI API key is configured, the app generates prioritized before/after rewrite examples targeting your weakest dimensions, explaining exactly why each score is low.
- **Real-time HTMX UI** — Results swap into the page instantly without a full reload, displaying a radial score gauge and color-coded per-dimension breakdown cards.
- **Configurable scoring weights** — Tune which GEO/AEO signals matter most for your domain via environment variables, without touching source code.
- **JSON API for programmatic access** — A `POST /api/analyze` endpoint accepts JSON and returns structured `AnalysisResult` payloads, making it easy to integrate into CI pipelines or content workflows.

---

## Usage Examples

### Web UI

Navigate to `http://localhost:8000`, paste your article or marketing copy into the text area, optionally provide a target query, and click **Analyze**. Results appear inline with a composite score and per-dimension breakdown.

### JSON API

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "OpenAI released GPT-4 in March 2023. According to their technical report, the model achieves state-of-the-art results on numerous benchmarks. How does GPT-4 differ from GPT-3.5? GPT-4 supports multimodal inputs and demonstrates stronger reasoning.",
    "target_query": "how is GPT-4 different from GPT-3.5"
  }'
```

**Example response:**

```json
{
  "composite_score": 74.2,
  "label": "Good",
  "dimensions": [
    { "name": "Question-Answer Alignment", "score": 88.0, "label": "Excellent", "explanation": "Content directly addresses the target query." },
    { "name": "Entity Density",            "score": 81.5, "label": "Good",      "explanation": "Key named entities are present and well-distributed." },
    { "name": "Citation Cues",             "score": 62.0, "label": "Fair",      "explanation": "References are present but could be more specific." },
    { "name": "Structured Formatting",     "score": 55.0, "label": "Fair",      "explanation": "Consider adding headers or bullet lists." },
    { "name": "Semantic Clarity",          "score": 78.0, "label": "Good",      "explanation": "Language is precise with minimal filler." },
    { "name": "Content Depth",             "score": 61.0, "label": "Fair",      "explanation": "Content could be expanded with more detail." }
  ],
  "suggestions": [
    {
      "dimension": "Structured Formatting",
      "problem": "Content is presented as a single block of prose.",
      "advice": "Break content into sections with H2 headings and use a bullet list for key differences.",
      "before": "GPT-4 supports multimodal inputs and demonstrates stronger reasoning.",
      "after": "## Key Differences\\n- **Multimodal input:** GPT-4 accepts images and text; GPT-3.5 is text-only.\\n- **Reasoning:** GPT-4 scores higher on standardized benchmarks."
    }
  ],
  "word_count": 47,
  "char_count": 312
}
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "spacy_model": "en_core_web_sm", "openai_configured": true}
```

### Running Tests

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run only the scoring unit tests (no network required)
pytest tests/test_scorer.py -v

# Run only model validation tests
pytest tests/test_models.py -v
```

---

## Project Structure

```
geo_aeo_optimizer/
├── pyproject.toml                        # Project metadata and build config
├── requirements.txt                      # Pinned runtime dependencies
├── .env.example                          # Environment variable template
│
├── geo_aeo_optimizer/
│   ├── __init__.py                       # Package initializer, version export
│   ├── main.py                           # FastAPI app, routes, lifecycle events
│   ├── scorer.py                         # Heuristic scoring engine (spaCy, LLM-free)
│   ├── suggestions.py                    # OpenAI rewrite suggestions generator
│   ├── models.py                         # Pydantic request/response schemas
│   └── templates/
│       ├── index.html                    # Main single-page UI (HTMX + Tailwind)
│       └── results_partial.html          # HTMX results partial (swapped on analyze)
│
└── tests/
    ├── __init__.py
    ├── test_scorer.py                    # Unit tests for all scoring functions
    ├── test_models.py                    # Unit tests for Pydantic model validation
    └── test_main.py                      # Integration tests for FastAPI routes
```

---

## Configuration

Copy `.env.example` to `.env` and set the following variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | Your OpenAI API key. Required for AI suggestions; scoring works without it. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model used for suggestion generation. |
| `OPENAI_MAX_TOKENS` | `1024` | Max tokens for the suggestions response. |
| `OPENAI_TIMEOUT` | `30` | OpenAI request timeout in seconds. |
| `WEIGHT_QA_ALIGNMENT` | `0.20` | Scoring weight for Question-Answer Alignment. |
| `WEIGHT_ENTITY_DENSITY` | `0.15` | Scoring weight for Entity Density. |
| `WEIGHT_STRUCTURED_FORMATTING` | `0.20` | Scoring weight for Structured Formatting. |
| `WEIGHT_CITATION_CUES` | `0.15` | Scoring weight for Citation Cues. |
| `WEIGHT_SEMANTIC_CLARITY` | `0.15` | Scoring weight for Semantic Clarity. |
| `WEIGHT_CONTENT_DEPTH` | `0.15` | Scoring weight for Content Depth. |
| `MAX_SUGGESTION_DIMENSIONS` | `3` | Number of weakest dimensions to generate suggestions for. |
| `HOST` | `0.0.0.0` | Server bind host. |
| `PORT` | `8000` | Server bind port. |

> **Tip:** Weights do not need to sum to exactly 1.0 — the engine normalizes them automatically. Adjust them to prioritize the signals that matter most for your content domain.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
