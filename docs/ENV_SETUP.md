# Environment variables

The app reads its configuration from a `.env` file at the repo root
(`config/settings.py`, loaded via `pydantic-settings`). Copy the template and
fill in real values:

```bash
cp .env-example .env
```

None of these are required just to run the test suite (`uv run pytest`) —
everything LLM-backed is mocked in tests, and the one live-API test
(`tests/dspy_lab/test_scene_parser_integration.py`) auto-skips when
`GEMINI_API_KEY` is absent or `"xxx"`. They *are* required to actually run
`scripts/run_experiment.py` against a real model.

## `GEMINI_API_KEY` (required to run the scene parser)

Used by `SceneParser` (`config/models.py`'s `VISION_MODEL`,
`gemini-2.5-flash`) — the model that turns an image into a structured
`Scene`, and by the `ocr`/`detect_objects` tools.

1. Go to **https://aistudio.google.com/apikey**.
2. Sign in with a Google account, click **Create API key**, pick or create a
   Google Cloud project.
3. Copy the key into `.env`:
   ```
   GEMINI_API_KEY=AIza...
   ```

Free tier is generally enough for this lab's dataset size (2 images).

## `OPENROUTER_API_KEY` (required to run the agent's tool planner)

Used by `ToolPlanner` (`config/models.py`'s `PLANNER_MODEL`,
`qwen/qwen2.5-vl` via OpenRouter) — the model that decides which tools to
call in the agent loop (`agents/loop.py`'s `CreativeAgent`).

1. Go to **https://openrouter.ai/keys** (sign in or create an account first
   at https://openrouter.ai).
2. Click **Create Key**, name it, copy the value.
3. Add credit to the account if the model you're using isn't free — check
   the model's pricing at https://openrouter.ai/models before running at
   scale.
4. Copy the key into `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

## `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` (optional — tracing)

Used by `observability/langfuse_tracing.py` to send a trace of every LM call
to Langfuse Cloud. If these are absent, `langfuse_tracing.setup()` silently
no-ops (logs one line, does not raise) — tracing is a nice-to-have, not a
requirement for the pipeline to run or score.

1. Go to **https://cloud.langfuse.com** and sign up for the free tier.
2. Create a project.
3. In the project's **Settings → API Keys**, click **Create new API keys**.
4. Copy both keys into `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
   Leave `LANGFUSE_HOST` as the default unless you're self-hosting Langfuse.

## `ENVIRONMENT`

Free-form label (`dev` by default) — not consumed by any provider, just
available on `settings.environment` if you want to branch behavior by
environment later.

## Verifying it worked

```bash
uv run pytest tests/dspy_lab/test_scene_parser_integration.py -v
```

With a real `GEMINI_API_KEY` in `.env`, this test runs for real (instead of
skipping) and asserts `SceneParser` returns a valid `Scene` for a real
dataset image.

```bash
uv run python3 -m scripts.run_experiment
```

Runs the full pipeline over `datasets/manifests/train.json` and prints one
score line per sample. If Langfuse keys are set, a trace should appear in
the Langfuse Cloud project dashboard within a minute or two.

## A note on secrets

`.env` is gitignored — never commit real keys. `.env-example` only ever
holds the placeholder `xxx`. `GeminiAdapter` sends its key via the
`x-goog-api-key` HTTP header (not a URL query string) specifically so it
can't leak into request logs or proxy traces.
