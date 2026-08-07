# DSPy Lab Experimental Setup — Design

Date: 2026-08-07

## Goal

Get the existing skeleton in this repo to a genuinely runnable end-to-end
experiment: an image of a creative (poster/ad) goes in, a DSPy program
parses it into a structured scene graph, that scene graph is scored against
a hand-authored ground truth, and `dspy.MIPROv2` can optimize the parser
against that score. Langfuse Cloud captures traces of every LM call. This is
scoped to the "inverse design" core of the research direction in the
ChatGPT transcript (reconstruct a design as a structured scene), not the
full multi-stage evaluator / branching-search agenda — those are future
work once this loop is proven.

Everything described below already exists as a stub (function signatures
and class shapes are in place); the work is filling in real logic, not
designing new modules from scratch, except where noted.

## 1. Fix the foundation

**Package name collision.** The project's own `dspy/` directory shadows the
real `dspy` framework — when anything in this repo (or its dependencies)
does `import dspy`, Python resolves it to `dspy/__init__.py` in this repo
first, not the installed library, because the repo root is on `sys.path`.
`eval/judge.py`, `dspy/lm.py`, `dspy/modules/*.py` all rely on `import dspy`
meaning the real framework (`dspy.Module`, `dspy.Signature`,
`dspy.ChainOfThought`, `dspy.Predict`, `dspy.LM`, `dspy.MIPROv2`). Fix:
rename the local package `dspy/` → `dspy_lab/`. Only two external call
sites reference it by dotted path and need updating:
`agents/planner.py` (`from dspy.modules.tool_planner import ToolPlanner`)
and `app/factory.py` (`from dspy.modules.scene_parser import SceneParser`).
Internal relative imports (`from .scene_parser import ...`,
`from ..signatures.scene import ...`) are unaffected by the rename.

**No dependency manifest exists.** Add `pyproject.toml` (uv-managed, project
name `dspy-lab`, `requires-python = ">=3.12"`), matching the conventions
used in `~/paceline/backend/pyproject.toml`. Dependencies:
`dspy`, `pydantic`, `pydantic-settings`, `aiohttp`, `orjson`, `Pillow`,
`cairosvg`, `langfuse>=3,<4`, `openinference-instrumentation-dspy`. Dev
group: `pytest`, `pytest-asyncio`.

## 2. Rendering — no browser dependency

Following the precedent in `~/paceline/docs/research/agentic-photo-editing.md`
(Pillow/ImageDraw + cairosvg, explicitly avoiding a Rust toolchain or
subprocess-based headless browser):

- `tools/generation/svg.py` (`generate_svg`): builds an SVG string
  deterministically from a `Scene` — `<rect>`/background for the canvas,
  one node per `VisualObject` (as a rect placeholder using its bbox/type),
  one `<text>` per `TextElement` positioned by its bbox with font-family/
  size/weight/color from the scene fields.
- `tools/generation/screenshot.py` (rename `render_browser` →
  `rasterize_svg`): uses `cairosvg.svg2png` to rasterize the SVG string
  into a PNG, written via `ArtifactStorage.save_bytes`. No Playwright, no
  subprocess.
- `tools/generation/html.py` (`generate_html`) stays a thin stub returning
  empty html/css — not on the critical path for the parser-optimization
  loop; only exercised by the smoke-test agent loop (section 7).

## 3. Vision tools — deterministic where possible, LLM only where necessary

- `tools/vision/image_analysis.py` (`analyze_image`): real — Pillow opens
  the image, returns width/height/aspect_ratio.
- `tools/design/palette.py` (`extract_palette`): real — Pillow's adaptive
  palette quantization (`Image.quantize`) to extract N dominant colors as
  hex.
- `tools/vision/ocr.py` (`ocr`) and `tools/vision/objects.py`
  (`detect_objects`): no local OCR/object-detection library is installed,
  and adding one (tesseract, a CV model) is out of scope. These call the
  Gemini adapter's `vision()` method with a prompt asking for text+boxes
  (OCR) or salient objects+boxes (objects), parsing the JSON response. This
  keeps them real (not empty stubs) without a new heavy dependency.
- `tools/design/layout.py` (`analyze_layout`): real — computed directly
  from bounding boxes already present in a parsed `Scene` (alignment
  buckets, grid inference, spacing deltas) rather than another LLM call.
- `tools/design/typography.py` (`estimate_typography`): stays a stub
  returning empty values — no font-detection capability available and it's
  not on the optimization critical path; documented as a known gap, not
  silently faked.

## 4. Scene parsing (the core LLM call)

`dspy/signatures/scene.py`'s `ExtractScene` signature and
`dspy_lab/modules/scene_parser.py`'s `SceneParser` (a `dspy.ChainOfThought`)
stay structurally the same, but:

- `models/adapters/gemini.py`'s `GeminiAdapter` is wired into `dspy_lab/lm.py`'s
  `AdapterLM` so `dspy.settings.configure(lm=...)` uses Gemini
  (`config/models.py`'s `VISION_MODEL`, currently `gemini-2.5-flash`) for
  this call.
- The signature's output field description is tightened so the model
  returns JSON matching the `Scene` pydantic schema directly (width,
  height, background, objects[], text[], colors[], layout) — this is what
  gets parsed into a `Scene` via `scene/parser.py`'s `SceneParser.from_json`.

`dspy_lab/lm.py`'s `AdapterLM.aforward` needs a real body: convert DSPy's
`messages` format into the adapter's expected `generate()`/`vision()` call
shape and convert the adapter's raw HTTP JSON response back into the
structure DSPy expects (`choices[0].message.content` or equivalent).

`models/adapters/openrouter.py`'s `OpenRouterAdapter` is wired the same way
for the planner LM (`config/models.py`'s `PLANNER_MODEL`).

## 5. Dataset

Fetch 2–3 real, simple-layout ad/poster images from the web (public,
non-copyright-sensitive use for internal research/eval — not
redistributed), saved to `datasets/raw/images/`. For each, hand-author a
matching ground-truth `Scene` JSON under `datasets/processed/scenes/`
(canvas size, the handful of text elements with approximate bbox/content,
dominant colors). `datasets/manifests/train.json` gets one entry per
sample (replacing the current single placeholder `poster_001` entry, which
points at a nonexistent image).

## 6. Scoring / optimization metric

`eval/scorer.py`'s `SceneScorer` becomes the DSPy metric: given a gold
`Scene` (from the manifest) and a predicted `Scene` (from `SceneParser`),
compute:
- `text`: average `text_similarity` (already implemented in
  `eval/metrics.py`) over matched text elements (matched by nearest bbox).
- `objects`: count/type overlap between gold and predicted objects.
- `layout`: bbox similarity (already implemented) averaged across matched
  elements.
- `colors`: `color_similarity` (already implemented).
- `aesthetic`: fixed at a neutral placeholder (e.g. `1.0`) for this pass —
  no vision-judge stage in scope yet (see "Out of scope").

This becomes a plain Python function `parser_metric(gold, pred) -> float`
matching DSPy's metric signature, passed into
`dspy/optimizers/compile.py`'s `MIPROv2(metric=parser_metric, auto="light")`
in place of the current `metric=None`.

## 7. Langfuse Cloud tracing

New `observability/langfuse_tracing.py`, mirroring the structure of
`~/paceline/backend/app/contexts/conversations/infrastructure/agent/observability.py`:
- `setup()`: reads `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/
  `LANGFUSE_HOST` from `config/settings.py`'s `Settings` (three new fields,
  `LANGFUSE_HOST` defaulting to `https://cloud.langfuse.com`); no-ops with a
  log line if the keys are absent; instruments DSPy via
  `openinference.instrumentation.dspy.DSPyInstrumentor().instrument()` so
  every `dspy.LM` call becomes a traced generation automatically, without
  per-call wiring. Never raises — wrapped in try/except like the paceline
  original.
- `flush()`: flushes the Langfuse client on shutdown/end of script.
- Called once at the top of `scripts/run_experiment.py`.

This replaces relying on the bare local `observability/tracer.py` `Trace`
class for LM-call visibility; `Trace`/`Logger`/`Metrics` stay as-is for
lightweight local run bookkeeping (they're cheap and already work).

You'll need to create a Langfuse Cloud free-tier project and add
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` to `.env` before the final run
produces visible traces — the pipeline runs and scores without them
(tracing just no-ops), so this isn't a hard blocker for development.

## 8. Entry point — `scripts/run_experiment.py`

New script (the thing that proves this all works):
1. `observability.langfuse_tracing.setup()`
2. Load `.env` via `config.settings`, configure `dspy_lab.config.configure(lm=...)`
   with the Gemini-backed `AdapterLM`.
3. `datasets.loader.DatasetLoader().load(load_manifest(...))` →
   list of `CreativeSample`.
4. For each sample: run `SceneParser()(image=...)`, load the paired
   ground-truth scene, score with `SceneScorer`/`parser_metric`.
5. Optionally (flag `--optimize`): run `dspy_lab.optimizers.compile.optimize`
   (MIPROv2) over the samples as a trainset.
6. Save results via `experiments/tracker.py`'s `ExperimentTracker` and
   print a summary table.
7. `observability.langfuse_tracing.flush()`

## 9. Agent loop / tool planner — real but secondary

`agents/executor.py`, `tools/runner.py`, `dspy_lab/modules/tool_planner.py`
already have working shapes; once the tools in sections 2–3 are real, the
existing `CreativeAgent.run()` loop (parse → plan → execute → iterate)
becomes exercisable as a smoke test (`app/pipeline.py`'s
`CreativePipeline.recreate()`), producing an actual rasterized SVG
recreation via the tool chain. This is not the optimization target (section
6 is) — it's validated by running once per dataset sample and confirming no
exceptions and a non-empty output image, not by a formal metric in this
pass.

## Out of scope for this pass

Explicitly deferred to future work, per the ChatGPT transcript's larger
agenda: hard-constraint checks (offscreen text, contrast ratio, etc.),
structural alignment/symmetry metrics, the vision-judge stage
(`dspy/signatures/judge.py`'s `JudgeCreative` / `dspy_lab/modules/critic.py`'s
`CreativeCritic` stay stubbed), reference-similarity scoring, preference
models, search/branching over candidate designs, and the semantic
scene-graph-as-AST protocol for third-party creative reconstruction. The
scorer's `aesthetic` weight is a placeholder until the vision judge exists.

## Testing

- Unit tests for the now-real deterministic pieces: `eval/metrics.py`
  similarity functions, `tools/design/palette.py` quantization,
  `tools/design/layout.py` alignment inference, `scene/parser.py` /
  `scene/validator.py`.
- One integration-style test (marked to skip if API keys are absent) that
  runs `SceneParser` against a real dataset image and asserts a
  well-formed, valid `Scene` comes back.
- No test asserts on absolute score values from the LLM-backed parser
  (non-deterministic); tests assert shape/validity, not score thresholds.
