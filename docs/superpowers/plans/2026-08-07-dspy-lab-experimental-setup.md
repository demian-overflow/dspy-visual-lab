# DSPy Lab Experimental Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing DSPy skeleton in this repo genuinely runnable end to end: an image goes in, `SceneParser` (a `dspy.ChainOfThought` backed by Gemini) parses it into a structured `Scene`, `SceneScorer` scores it against a hand-authored ground truth, `dspy.MIPROv2` can optimize the parser against that score, and every LM call is traced to Langfuse Cloud.

**Architecture:** Existing modules keep their current shapes (`Scene` pydantic schema, `SceneParser` DSPy module, `SceneScorer`, `CreativeAgent` loop, `ToolRunner`/tool registry) — this plan fills in real logic behind stubs and fixes one structural defect (the local `dspy/` package shadows the real `dspy` framework). Deterministic work (palette extraction, layout metrics, SVG rendering) runs without an LLM; only scene parsing and the two vision tools that need pixel-level judgment (`ocr`, `detect_objects`) call out to Gemini.

**Tech Stack:** Python 3.12, `dspy` 3.3.x, `pydantic`/`pydantic-settings`, `aiohttp`, `orjson`, `Pillow`, `cairosvg`, `langfuse` (pinned `>=3,<4` — v3's OTel-based span API is what's used here), `openinference-instrumentation-dspy`, `uv` for dependency management, `pytest`/`pytest-asyncio`.

## Global Constraints

- `requires-python = ">=3.12"` (matches installed interpreter).
- `langfuse` must be pinned `>=3,<4` — the `start_as_current_span`/`update_current_trace` API used here is v3's; v4 changed the client API.
- No headless browser / subprocess-based rendering — SVG → PNG goes through `cairosvg` in-process (matches the precedent in `~/paceline/docs/research/agentic-photo-editing.md`).
- No new local OCR/object-detection library — `ocr` and `detect_objects` tools call Gemini vision instead of adding a CV dependency.
- Tests must not assert on absolute LLM-produced score values (non-deterministic); assert shape/validity/no-exception instead, and skip real-API tests when `GEMINI_API_KEY`/`OPENROUTER_API_KEY` are unset or `"xxx"`.
- Bounding box coordinates are pixel values in the source image's coordinate space (matches `eval/metrics.py`'s existing `bbox_similarity`, which sums raw deltas).
- Existing internal relative imports (`from .scene_parser import ...`) must keep working — only dotted cross-package imports of the local `dspy` package change.

---

### Task 1: Fix the package name collision + add dependency management

**Files:**
- Rename (git mv): `dspy/` → `dspy_lab/` (every file under it, including `__init__.py`s)
- Modify: `agents/planner.py:1`, `app/factory.py:7`
- Create: `pyproject.toml`
- Test: `tests/test_imports.py`

**Interfaces:**
- Produces: the `dspy_lab` package, importable as `dspy_lab.modules.scene_parser.SceneParser`, `dspy_lab.modules.tool_planner.ToolPlanner`, `dspy_lab.modules.pipeline.CreativePipeline`, `dspy_lab.signatures.scene.ExtractScene`, `dspy_lab.signatures.planner.CreatePlan`, `dspy_lab.signatures.judge.JudgeCreative`, `dspy_lab.lm.AdapterLM`, `dspy_lab.config.configure`, `dspy_lab.optimizers.compile.optimize`, `dspy_lab.programs.recreate.run`.

- [ ] **Step 1: Rename the local package**

```bash
git mv dspy dspy_lab
```

- [ ] **Step 2: Update the two external call sites**

In `agents/planner.py`, change:
```python
from dspy.modules.tool_planner import ToolPlanner
```
to:
```python
from dspy_lab.modules.tool_planner import ToolPlanner
```

In `app/factory.py`, change:
```python
from dspy.modules.scene_parser import SceneParser
```
to:
```python
from dspy_lab.modules.scene_parser import SceneParser
```

- [ ] **Step 3: Add `pyproject.toml`**

```toml
[project]
name = "dspy-lab"
version = "0.1.0"
description = "DSPy experimental lab for visual/creative editing"
requires-python = ">=3.12"
dependencies = [
    "dspy>=3.3,<4",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "aiohttp>=3.9",
    "orjson>=3.10",
    "Pillow>=10.4",
    "cairosvg>=2.7,<3",
    "langfuse>=3,<4",
    "openinference-instrumentation-dspy>=0.1.20",
]

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "-ra -v"
testpaths = ["tests"]
```

- [ ] **Step 4: Install with uv**

Run: `cd /home/demian/labs/dspy_visual && uv sync`
Expected: creates `.venv` and `uv.lock`, installs `dspy`, `pydantic`, `pydantic-settings`, `aiohttp`, `orjson`, `Pillow`, `cairosvg`, `langfuse`, `openinference-instrumentation-dspy`, `pytest`, `pytest-asyncio` with no errors.

- [ ] **Step 5: Write the failing test**

```python
# tests/test_imports.py
import dspy


def test_real_dspy_framework_is_importable_not_shadowed():
    assert hasattr(dspy, "Signature")
    assert hasattr(dspy, "Module")
    assert hasattr(dspy, "ChainOfThought")
    assert hasattr(dspy, "LM")
    assert hasattr(dspy, "MIPROv2")


def test_local_package_renamed():
    import dspy_lab.modules.scene_parser  # noqa: F401
    import dspy_lab.signatures.scene  # noqa: F401

    from agents.planner import AgentPlanner  # noqa: F401
    from app.factory import create_app  # noqa: F401
```

- [ ] **Step 6: Run test, confirm behavior**

Run: `uv run pytest tests/test_imports.py -v`
Expected: PASS. Before the rename this would have failed (`dspy.Signature` etc. don't exist on the local stub package; `AttributeError`).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Rename local dspy/ package to dspy_lab/ to stop shadowing the real dspy framework; add pyproject.toml"
```

---

### Task 2: Settings — add Langfuse fields, update env files

**Files:**
- Modify: `config/settings.py`
- Modify: `.env-example`
- Modify: `.env`
- Test: `tests/config/test_settings.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `config.settings.settings.langfuse_public_key: str | None`, `settings.langfuse_secret_key: str | None`, `settings.langfuse_host: str` (default `"https://cloud.langfuse.com"`) — consumed by Task 10.

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_settings.py
from config.settings import Settings


def test_settings_has_langfuse_fields_with_cloud_default():
    s = Settings(_env_file=None)
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "https://cloud.langfuse.com"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_settings.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'langfuse_public_key'`

- [ ] **Step 3: Add the fields**

In `config/settings.py`, add after the existing `gemini_api_key` field:

```python
    langfuse_public_key: str | None = None

    langfuse_secret_key: str | None = None

    langfuse_host: str = "https://cloud.langfuse.com"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_settings.py -v`
Expected: PASS

- [ ] **Step 5: Update env files**

`.env-example`:
```
OPENROUTER_API_KEY=xxx
GEMINI_API_KEY=xxx

LANGFUSE_PUBLIC_KEY=xxx
LANGFUSE_SECRET_KEY=xxx
LANGFUSE_HOST=https://cloud.langfuse.com

ENVIRONMENT=dev
```

`.env` (local, gitignored — keep existing placeholder style, append the three new vars):
```
OPENROUTER_API_KEY=xxx
GEMINI_API_KEY=xxx

LANGFUSE_PUBLIC_KEY=xxx
LANGFUSE_SECRET_KEY=xxx
LANGFUSE_HOST=https://cloud.langfuse.com

ENVIRONMENT=dev
```

- [ ] **Step 6: Commit**

```bash
git add config/settings.py .env-example tests/config/test_settings.py
git commit -m "Add Langfuse settings fields with Cloud default host"
```

Note: `.env` is gitignored, don't `git add` it.

---

### Task 3: Wire adapters into DSPy — `AdapterLM`

**Files:**
- Modify: `dspy_lab/lm.py`
- Modify: `models/adapters/gemini.py`, `models/adapters/openrouter.py`
- Create: `models/factory.py`
- Test: `tests/dspy_lab/test_lm.py`

**Interfaces:**
- Consumes: `models.adapters.base.BaseAdapter` (`.generate(messages, **kwargs)`, `.vision(image, prompt, **kwargs)`), `models.adapters.aiohttp_client.HTTPClient`.
- Produces: `dspy_lab.lm.AdapterLM(adapter, model_name)` — a `dspy.LM` subclass usable via `dspy.settings.configure(lm=AdapterLM(...))`; `models.factory.build_adapter(config: config.models.ModelConfig, api_key: str, client: HTTPClient) -> BaseAdapter`, consumed by Task 8 and Task 12.

First, confirm the exact contract `dspy.LM` expects from a subclass overriding `forward`/`aforward` (installed as part of Task 1's `uv sync`):

- [ ] **Step 1: Inspect the installed dspy.LM contract**

Run: `uv run python3 -c "import dspy, inspect; print(inspect.getsource(dspy.LM.__init__)); print(dspy.LM.forward_contract)"`
Confirm: `dspy.LM` is instantiated with `model: str` (a string, even though we won't use litellm's `provider/model` resolution — `AdapterLM` bypasses litellm entirely by overriding `forward`/`aforward`), and `forward_contract = "legacy"` — meaning `forward()`/`aforward()` must return an OpenAI-Chat-Completions-shaped object (dict access `["choices"][0]["message"]["content"]` is supported).

- [ ] **Step 2: Write the failing test (using a fake adapter, no network)**

```python
# tests/dspy_lab/test_lm.py
import pytest

from dspy_lab.lm import AdapterLM


class FakeAdapter:
    def __init__(self, reply):
        self.reply = reply
        self.last_call = None

    async def generate(self, messages, **kwargs):
        self.last_call = messages
        return {"candidates": [{"content": {"parts": [{"text": self.reply}]}}]}


@pytest.mark.asyncio
async def test_aforward_returns_openai_shaped_response_with_adapter_text():
    adapter = FakeAdapter(reply="hello from adapter")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    result = await lm.aforward(messages=[{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "hello from adapter"
    assert adapter.last_call == [{"role": "user", "content": "hi"}]


def test_forward_is_sync_wrapper_over_aforward():
    adapter = FakeAdapter(reply="sync path")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    result = lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "sync path"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/dspy_lab/test_lm.py -v`
Expected: FAIL — `AdapterLM.__init__` doesn't call `super().__init__`, `aforward`/`forward` don't return the expected shape yet.

- [ ] **Step 4: Implement `AdapterLM`**

```python
# dspy_lab/lm.py
import asyncio

import dspy


class AdapterLM(dspy.LM):

    forward_contract = "legacy"

    def __init__(self, adapter, model_name):
        super().__init__(model=model_name, cache=False)
        self.adapter = adapter
        self.model = model_name

    @staticmethod
    def _extract_text(raw_response):
        candidates = raw_response.get("candidates")
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)

        choices = raw_response.get("choices")
        if choices:
            return choices[0].get("message", {}).get("content", "")

        raise ValueError(f"Unrecognized adapter response shape: {raw_response!r}")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]

        raw_response = await self.adapter.generate(messages, **kwargs)
        text = self._extract_text(raw_response)

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "model": self.model,
        }

    def forward(self, prompt=None, messages=None, **kwargs):
        return asyncio.run(self.aforward(prompt=prompt, messages=messages, **kwargs))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/dspy_lab/test_lm.py -v`
Expected: PASS

- [ ] **Step 6: Add `models/factory.py`**

```python
# models/factory.py
from config.models import ModelConfig
from models.adapters.base import BaseAdapter
from models.adapters.gemini import GeminiAdapter
from models.adapters.openrouter import OpenRouterAdapter


def build_adapter(config: ModelConfig, api_key: str, client) -> BaseAdapter:
    if config.provider == "gemini":
        return GeminiAdapter(client=client, api_key=api_key, model=config.name)
    if config.provider == "openrouter":
        return OpenRouterAdapter(client=client, api_key=api_key, model=config.name)
    raise ValueError(f"Unknown provider: {config.provider!r}")
```

- [ ] **Step 7: Write test for the factory**

Append to `tests/dspy_lab/test_lm.py`... actually put this in its own file:

```python
# tests/models/test_factory.py
from config.models import ModelConfig
from models.adapters.gemini import GeminiAdapter
from models.adapters.openrouter import OpenRouterAdapter
from models.factory import build_adapter


def test_build_adapter_gemini():
    config = ModelConfig(name="gemini-2.5-flash", provider="gemini")
    adapter = build_adapter(config, api_key="key", client=object())
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.model == "gemini-2.5-flash"


def test_build_adapter_openrouter():
    config = ModelConfig(name="qwen/qwen2.5-vl", provider="openrouter")
    adapter = build_adapter(config, api_key="key", client=object())
    assert isinstance(adapter, OpenRouterAdapter)
    assert adapter.model == "qwen/qwen2.5-vl"


def test_build_adapter_unknown_provider_raises():
    config = ModelConfig(name="x", provider="nope")
    try:
        build_adapter(config, api_key="key", client=object())
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 8: Run and confirm pass**

Run: `uv run pytest tests/models/test_factory.py tests/dspy_lab/test_lm.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add dspy_lab/lm.py models/factory.py tests/dspy_lab tests/models
git commit -m "Implement AdapterLM (legacy dspy.LM contract) and an adapter factory"
```

---

### Task 4: Deterministic image tools — `analyze_image`, `extract_palette`

**Files:**
- Modify: `tools/vision/image_analysis.py`
- Modify: `tools/design/palette.py`
- Test: `tests/tools/test_image_tools.py`

**Interfaces:**
- Consumes: nothing beyond `Pillow` and `tools/registry.py`'s existing `@register` decorator (unchanged).
- Produces: `analyze_image(image: str) -> dict` with keys `width`, `height`, `aspect_ratio` (image is a filesystem path); `extract_palette(image: str, n: int = 5) -> dict` with key `colors: list[str]` (hex strings, most-dominant first).

- [ ] **Step 1: Write a small fixture image generator used by this and later tool tests**

```python
# tests/conftest.py
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def two_color_image(tmp_path: Path) -> Path:
    img = Image.new("RGB", (200, 100), color="#ff0000")
    for x in range(100, 200):
        for y in range(100):
            img.putpixel((x, y), (0, 0, 255))
    path = tmp_path / "two_color.png"
    img.save(path)
    return path
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/tools/test_image_tools.py
import pytest

from tools.vision.image_analysis import analyze_image
from tools.design.palette import extract_palette


@pytest.mark.asyncio
async def test_analyze_image_returns_dimensions(two_color_image):
    result = await analyze_image(str(two_color_image))
    assert result["width"] == 200
    assert result["height"] == 100
    assert result["aspect_ratio"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_extract_palette_finds_dominant_colors(two_color_image):
    result = await extract_palette(str(two_color_image), n=2)
    colors = {c.lower() for c in result["colors"]}
    assert "#ff0000" in colors
    assert "#0000ff" in colors
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/tools/test_image_tools.py -v`
Expected: FAIL — both tools currently return zeros/empty list.

- [ ] **Step 4: Implement `analyze_image`**

```python
# tools/vision/image_analysis.py
from PIL import Image

from ..registry import register


@register(
    "analyze_image",
    "General image properties"
)
async def analyze_image(image):
    with Image.open(image) as img:
        width, height = img.size

    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else 0,
    }
```

- [ ] **Step 5: Implement `extract_palette`**

```python
# tools/design/palette.py
from PIL import Image

from ..registry import register


@register(
    "extract_palette",
    "Extract dominant colors"
)
async def extract_palette(image, n=5):
    with Image.open(image) as img:
        rgb = img.convert("RGB")
        quantized = rgb.quantize(colors=n, method=Image.MEDIANCUT)
        palette = quantized.getpalette()[: n * 3]
        counts = sorted(quantized.getcolors(), reverse=True)

    colors = [
        "#{:02x}{:02x}{:02x}".format(
            palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2]
        )
        for _count, idx in counts
    ]

    return {"colors": colors}
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/tools/test_image_tools.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tools/vision/image_analysis.py tools/design/palette.py tests/tools/test_image_tools.py tests/conftest.py
git commit -m "Implement analyze_image and extract_palette with Pillow"
```

---

### Task 5: Scene-graph layout metrics — `analyze_layout`

**Files:**
- Modify: `tools/design/layout.py`
- Test: `tests/tools/test_layout.py`

**Interfaces:**
- Consumes: a `scene` argument shaped like `scene/schema.py`'s `Scene.model_dump()` (dict with `objects`/`text`, each item having `bbox: {x, y, width, height}`).
- Produces: `analyze_layout(scene: dict) -> dict` with keys `grid` (`"unknown"` if <2 elements, else `"aligned"`/`"unaligned"`), `alignment` (`"left"`/`"center"`/`"right"`/`"mixed"`), `spacing` (`dict` with `min_gap`, `max_gap` between vertically-sorted element bboxes).

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_layout.py
import pytest

from tools.design.layout import analyze_layout


def _elem(x, y, w, h):
    return {"bbox": {"x": x, "y": y, "width": w, "height": h}}


@pytest.mark.asyncio
async def test_analyze_layout_detects_left_alignment():
    scene = {
        "objects": [_elem(10, 10, 50, 20), _elem(10, 50, 80, 20)],
        "text": [_elem(10, 90, 60, 20)],
    }
    result = await analyze_layout(scene)
    assert result["alignment"] == "left"
    assert result["grid"] == "aligned"


@pytest.mark.asyncio
async def test_analyze_layout_reports_unknown_grid_for_single_element():
    scene = {"objects": [_elem(0, 0, 10, 10)], "text": []}
    result = await analyze_layout(scene)
    assert result["grid"] == "unknown"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/tools/test_layout.py -v`
Expected: FAIL — stub returns empty strings.

- [ ] **Step 3: Implement**

```python
# tools/design/layout.py
from ..registry import register


def _all_elements(scene):
    return list(scene.get("objects", [])) + list(scene.get("text", []))


@register(
    "analyze_layout",
    "Analyze composition and spacing"
)
async def analyze_layout(scene):
    elements = _all_elements(scene)

    if len(elements) < 2:
        return {"grid": "unknown", "alignment": "unknown", "spacing": {}}

    lefts = [e["bbox"]["x"] for e in elements]
    rights = [e["bbox"]["x"] + e["bbox"]["width"] for e in elements]
    centers = [e["bbox"]["x"] + e["bbox"]["width"] / 2 for e in elements]

    tolerance = 5.0

    def _spread(values):
        return max(values) - min(values)

    spreads = {
        "left": _spread(lefts),
        "right": _spread(rights),
        "center": _spread(centers),
    }
    alignment = min(spreads, key=spreads.get)
    if spreads[alignment] > tolerance:
        alignment = "mixed"

    grid = "aligned" if alignment != "mixed" else "unaligned"

    sorted_by_y = sorted(elements, key=lambda e: e["bbox"]["y"])
    gaps = [
        sorted_by_y[i + 1]["bbox"]["y"]
        - (sorted_by_y[i]["bbox"]["y"] + sorted_by_y[i]["bbox"]["height"])
        for i in range(len(sorted_by_y) - 1)
    ]

    spacing = {"min_gap": min(gaps), "max_gap": max(gaps)} if gaps else {}

    return {"grid": grid, "alignment": alignment, "spacing": spacing}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/tools/test_layout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/design/layout.py tests/tools/test_layout.py
git commit -m "Implement analyze_layout as a deterministic bbox-based metric"
```

---

### Task 6: Vision LLM tools — `ocr`, `detect_objects`

**Files:**
- Modify: `tools/vision/ocr.py`
- Modify: `tools/vision/objects.py`
- Modify: `models/adapters/gemini.py` (fix `vision()` to accept a filesystem path and base64-encode it)
- Test: `tests/tools/test_vision_llm_tools.py`

**Interfaces:**
- Consumes: `models.factory.build_adapter` (Task 3), `config.settings.settings`, `config.models.VISION_MODEL`.
- Produces: `ocr(image: str) -> dict` with keys `text: list[str]`, `boxes: list[dict]` (each `{x, y, width, height}`); `detect_objects(image: str) -> dict` with key `objects: list[dict]` (each `{type, bbox}`).

These two tools need a configured Gemini adapter instance. Rather than constructing one per call, add a lazy module-level singleton.

- [ ] **Step 1: Fix `GeminiAdapter.vision()` to accept a file path**

Currently `vision(image, prompt)` treats `image` as already-base64 bytes. Update `models/adapters/gemini.py`:

```python
# models/adapters/gemini.py
import base64

from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):

    def __init__(self, client, api_key, model="gemini-2.5-flash"):
        self.client = client
        self.api_key = api_key
        self.model = model

    def url(self):
        return (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent?key={self.api_key}"
        )

    async def generate(self, contents, **kwargs):
        return await self.client.post(
            self.url(),
            payload={"contents": contents, **kwargs},
        )

    async def vision(self, image, prompt, **kwargs):
        with open(image, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        return await self.generate(
            [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": encoded,
                            }
                        },
                    ]
                }
            ],
            **kwargs,
        )
```

- [ ] **Step 2: Write the failing tests (mocking the adapter, no real network call)**

```python
# tests/tools/test_vision_llm_tools.py
import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.vision.ocr import ocr
from tools.vision.objects import detect_objects


def _gemini_response(payload: dict):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


@pytest.mark.asyncio
async def test_ocr_parses_gemini_json_response(two_color_image):
    fake_vision = AsyncMock(
        return_value=_gemini_response(
            {"text": ["Hello"], "boxes": [{"x": 1, "y": 2, "width": 3, "height": 4}]}
        )
    )
    with patch("tools.vision.ocr._get_adapter") as get_adapter:
        get_adapter.return_value.vision = fake_vision
        result = await ocr(str(two_color_image))

    assert result["text"] == ["Hello"]
    assert result["boxes"] == [{"x": 1, "y": 2, "width": 3, "height": 4}]


@pytest.mark.asyncio
async def test_detect_objects_parses_gemini_json_response(two_color_image):
    fake_vision = AsyncMock(
        return_value=_gemini_response(
            {"objects": [{"type": "logo", "bbox": {"x": 0, "y": 0, "width": 5, "height": 5}}]}
        )
    )
    with patch("tools.vision.objects._get_adapter") as get_adapter:
        get_adapter.return_value.vision = fake_vision
        result = await detect_objects(str(two_color_image))

    assert result["objects"][0]["type"] == "logo"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/tools/test_vision_llm_tools.py -v`
Expected: FAIL — `_get_adapter` doesn't exist yet, tools return empty stubs.

- [ ] **Step 4: Implement a shared lazy adapter helper + `ocr`**

```python
# tools/vision/ocr.py
import json

from config.models import VISION_MODEL
from config.settings import settings
from models.adapters.aiohttp_client import HTTPClient
from models.factory import build_adapter

from ..registry import register

_adapter = None
_client = None


def _get_adapter():
    global _adapter, _client
    if _adapter is None:
        _client = HTTPClient()
        _adapter = build_adapter(VISION_MODEL, api_key=settings.gemini_api_key, client=_client)
    return _adapter


def _extract_text(raw_response):
    return raw_response["candidates"][0]["content"]["parts"][0]["text"]


@register(
    "ocr",
    "Extract text with coordinates"
)
async def ocr(image):
    adapter = _get_adapter()
    raw_response = await adapter.vision(
        image,
        prompt=(
            "Return ONLY JSON matching "
            '{"text": [strings], "boxes": [{"x":num,"y":num,"width":num,"height":num}]} '
            "for every text element visible in this image, in pixel coordinates."
        ),
    )
    return json.loads(_extract_text(raw_response))
```

- [ ] **Step 5: Implement `detect_objects`**

```python
# tools/vision/objects.py
import json

from config.models import VISION_MODEL
from config.settings import settings
from models.adapters.aiohttp_client import HTTPClient
from models.factory import build_adapter

from ..registry import register

_adapter = None
_client = None


def _get_adapter():
    global _adapter, _client
    if _adapter is None:
        _client = HTTPClient()
        _adapter = build_adapter(VISION_MODEL, api_key=settings.gemini_api_key, client=_client)
    return _adapter


def _extract_text(raw_response):
    return raw_response["candidates"][0]["content"]["parts"][0]["text"]


@register(
    "detect_objects",
    "Detect objects and bounding boxes"
)
async def detect_objects(image):
    adapter = _get_adapter()
    raw_response = await adapter.vision(
        image,
        prompt=(
            "Return ONLY JSON matching "
            '{"objects": [{"type": string, "bbox": {"x":num,"y":num,"width":num,"height":num}}]} '
            "for every salient visual object (not text) in this image, in pixel coordinates."
        ),
    )
    return json.loads(_extract_text(raw_response))
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/tools/test_vision_llm_tools.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add models/adapters/gemini.py tools/vision/ocr.py tools/vision/objects.py tests/tools/test_vision_llm_tools.py
git commit -m "Implement ocr and detect_objects via Gemini vision, fix GeminiAdapter.vision to accept file paths"
```

---

### Task 7: SVG generation + rasterization (no browser)

**Files:**
- Modify: `tools/generation/svg.py`
- Modify: `tools/generation/screenshot.py` (rename registered tool `render_browser` → `rasterize_svg`)
- Modify: `agents/executor.py` if it references `render_browser` by name (check via grep first)
- Test: `tests/tools/test_generation.py`

**Interfaces:**
- Consumes: a `scene` dict shaped like `Scene.model_dump()`.
- Produces: `generate_svg(scene: dict) -> dict` with key `svg: str`; `rasterize_svg(svg: str) -> dict` with key `image_path: str` (a written PNG file under `storage`'s artifacts dir).

- [ ] **Step 1: Check for other references to `render_browser`**

Run: `grep -rn "render_browser" /home/demian/labs/dspy_visual --include="*.py"`
If any call site outside `tools/generation/screenshot.py` references it by string name (e.g. in a plan JSON or `tools/schema.py`), note it and update in Step 4. As of this plan's writing there are none — the tool is only referenced via the registry.

- [ ] **Step 2: Write the failing tests**

```python
# tests/tools/test_generation.py
import pytest

from tools.generation.svg import generate_svg
from tools.generation.screenshot import rasterize_svg


@pytest.mark.asyncio
async def test_generate_svg_includes_text_and_background():
    scene = {
        "width": 100,
        "height": 50,
        "background": "#ffffff",
        "objects": [],
        "text": [
            {
                "id": "t1",
                "content": "Hello",
                "bbox": {"x": 5, "y": 5, "width": 40, "height": 10},
                "font_family": "sans-serif",
                "font_size": 12,
                "color": "#000000",
            }
        ],
    }
    result = await generate_svg(scene)
    assert "<svg" in result["svg"]
    assert "Hello" in result["svg"]
    assert 'width="100"' in result["svg"]


@pytest.mark.asyncio
async def test_rasterize_svg_writes_a_real_png(tmp_path, monkeypatch):
    from storage import paths
    monkeypatch.setattr(paths, "ROOT", tmp_path)

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        '<rect width="10" height="10" fill="red"/></svg>'
    )
    result = await rasterize_svg(svg)
    image_path = result["image_path"]

    from PIL import Image
    with Image.open(image_path) as img:
        assert img.size == (10, 10)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/tools/test_generation.py -v`
Expected: FAIL — both tools currently return empty strings.

- [ ] **Step 4: Implement `generate_svg`**

```python
# tools/generation/svg.py
from xml.sax.saxutils import escape

from ..registry import register


def _rect_for(obj, fill="#cccccc"):
    b = obj["bbox"]
    return f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["width"]}" height="{b["height"]}" fill="{fill}"/>'


def _text_for(item):
    b = item["bbox"]
    family = item.get("font_family") or "sans-serif"
    size = item.get("font_size") or 16
    color = item.get("color") or "#000000"
    content = escape(item.get("content", ""))
    return (
        f'<text x="{b["x"]}" y="{b["y"] + size}" '
        f'font-family="{escape(family)}" font-size="{size}" fill="{color}">{content}</text>'
    )


@register(
    "generate_svg",
    "Generate SVG design"
)
async def generate_svg(scene):
    width = scene.get("width", 0)
    height = scene.get("height", 0)
    background = scene.get("background", "#ffffff")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{background}"/>',
    ]
    parts += [_rect_for(obj) for obj in scene.get("objects", [])]
    parts += [_text_for(item) for item in scene.get("text", [])]
    parts.append("</svg>")

    return {"svg": "".join(parts)}
```

- [ ] **Step 5: Implement `rasterize_svg`**

```python
# tools/generation/screenshot.py
import uuid

import cairosvg

from storage.paths import ROOT
from ..registry import register


@register(
    "rasterize_svg",
    "Rasterize an SVG string to a PNG file"
)
async def rasterize_svg(svg):
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / f"{uuid.uuid4()}.png"
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(image_path))

    return {"image_path": str(image_path)}
```

Note: `storage/paths.py`'s `ROOT = Path("runs")` is specific to run storage; this reuses that module's `ROOT` symbol name for the monkeypatch in the test but writes to an `artifacts/` sibling directory rather than reusing `RUNS`. If `storage/paths.py` only exports `ROOT` (used for runs) and not a generic project root, add one:

```python
# storage/paths.py — add alongside existing ROOT/run_path
from pathlib import Path

ROOT = Path("runs")


def run_path(run_id: str) -> Path:
    path = ROOT / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
```

Since `ROOT` here is `Path("runs")`, not a generic project root, fix the import in `tools/generation/screenshot.py` to use `config.paths.ARTIFACTS` instead (already defined in `config/paths.py` as `ROOT / "artifacts"`):

```python
# tools/generation/screenshot.py (corrected)
import uuid

import cairosvg

from config.paths import ARTIFACTS
from ..registry import register


@register(
    "rasterize_svg",
    "Rasterize an SVG string to a PNG file"
)
async def rasterize_svg(svg):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    image_path = ARTIFACTS / f"{uuid.uuid4()}.png"
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(image_path))

    return {"image_path": str(image_path)}
```

And correct the test's monkeypatch target accordingly:

```python
# tests/tools/test_generation.py — corrected rasterize test
@pytest.mark.asyncio
async def test_rasterize_svg_writes_a_real_png(tmp_path, monkeypatch):
    import config.paths as cfg_paths
    monkeypatch.setattr(cfg_paths, "ARTIFACTS", tmp_path / "artifacts")

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        '<rect width="10" height="10" fill="red"/></svg>'
    )
    result = await rasterize_svg(svg)
    image_path = result["image_path"]

    from PIL import Image
    with Image.open(image_path) as img:
        assert img.size == (10, 10)
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/tools/test_generation.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tools/generation/svg.py tools/generation/screenshot.py tests/tools/test_generation.py
git commit -m "Implement generate_svg and rasterize_svg (cairosvg, no browser dependency)"
```

---

### Task 8: Scoring — `SceneScorer` as a DSPy metric

**Files:**
- Modify: `eval/scorer.py`
- Modify: `eval/metrics.py` (add element-matching helper)
- Test: `tests/eval/test_scorer.py`

**Interfaces:**
- Consumes: two `Scene`-shaped dicts (gold, predicted), each with `text: list[{content, bbox}]`, `objects: list[{type, bbox}]`, `colors: list[{hex}]`.
- Produces: `eval.metrics.match_by_bbox(gold_items: list[dict], pred_items: list[dict]) -> list[tuple[dict, dict | None]]` (pairs each gold item with its nearest predicted item by bbox center distance, or `None` if `pred_items` is empty); `eval.scorer.SceneScorer.score(gold: dict, pred: dict) -> float`; `eval.scorer.parser_metric(gold, pred, trace=None) -> float` — the DSPy-metric-shaped wrapper consumed by Task 9. `gold`/`pred` in `parser_metric` are `dspy.Example`/`dspy.Prediction` objects whose `.scene` attribute is a JSON string (matching `ExtractScene`'s `scene` output field) or a dict — `parser_metric` must handle both (parse JSON string if needed).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_scorer.py
import json

from eval.metrics import match_by_bbox
from eval.scorer import SceneScorer, parser_metric


def test_match_by_bbox_pairs_nearest_elements():
    gold = [{"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}, "content": "A"}]
    pred = [
        {"bbox": {"x": 100, "y": 100, "width": 10, "height": 10}, "content": "far"},
        {"bbox": {"x": 1, "y": 1, "width": 10, "height": 10}, "content": "near"},
    ]
    pairs = match_by_bbox(gold, pred)
    assert pairs[0][1]["content"] == "near"


def test_match_by_bbox_handles_empty_pred():
    gold = [{"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}, "content": "A"}]
    pairs = match_by_bbox(gold, [])
    assert pairs == [(gold[0], None)]


def test_scene_scorer_perfect_match_scores_near_one():
    scene = {
        "text": [{"content": "Hello", "bbox": {"x": 0, "y": 0, "width": 10, "height": 10}}],
        "objects": [{"type": "logo", "bbox": {"x": 20, "y": 20, "width": 5, "height": 5}}],
        "colors": [{"hex": "#ff0000"}],
    }
    score = SceneScorer().score(scene, scene)
    assert score > 0.95


def test_scene_scorer_no_overlap_scores_low():
    gold = {
        "text": [{"content": "Hello", "bbox": {"x": 0, "y": 0, "width": 10, "height": 10}}],
        "objects": [],
        "colors": [{"hex": "#ff0000"}],
    }
    pred = {"text": [], "objects": [], "colors": [{"hex": "#00ff00"}]}
    score = SceneScorer().score(gold, pred)
    assert score < 0.5


def test_parser_metric_parses_json_string_scene():
    class FakeExample:
        scene = json.dumps({"text": [], "objects": [], "colors": []})

    class FakePrediction:
        scene = json.dumps({"text": [], "objects": [], "colors": []})

    result = parser_metric(FakeExample(), FakePrediction())
    assert isinstance(result, float)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/eval/test_scorer.py -v`
Expected: FAIL — `match_by_bbox` and `parser_metric` don't exist; `SceneScorer.score` expects a flat `result` dict of pre-computed sub-scores, not two scenes.

- [ ] **Step 3: Add `match_by_bbox` to `eval/metrics.py`**

Append to the existing file (keep `text_similarity`, `color_similarity`, `bbox_similarity` as-is):

```python
def _bbox_center(bbox):
    return (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)


def _center_distance(a, b):
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def match_by_bbox(gold_items, pred_items):
    if not pred_items:
        return [(g, None) for g in gold_items]

    pairs = []
    for g in gold_items:
        nearest = min(pred_items, key=lambda p: _center_distance(g["bbox"], p["bbox"]))
        pairs.append((g, nearest))
    return pairs
```

- [ ] **Step 4: Rewrite `eval/scorer.py`**

```python
# eval/scorer.py
import json

from .metrics import bbox_similarity, color_similarity, match_by_bbox, text_similarity


class SceneScorer:

    WEIGHTS = {
        "text": 0.25,
        "objects": 0.25,
        "layout": 0.20,
        "colors": 0.15,
        "aesthetic": 0.15,
    }

    def _text_score(self, gold, pred):
        pairs = match_by_bbox(gold.get("text", []), pred.get("text", []))
        if not pairs:
            return 1.0
        scores = [
            text_similarity(g.get("content", ""), p.get("content", "")) if p else 0.0
            for g, p in pairs
        ]
        return sum(scores) / len(scores)

    def _objects_score(self, gold, pred):
        gold_objects = gold.get("objects", [])
        if not gold_objects:
            return 1.0
        pairs = match_by_bbox(gold_objects, pred.get("objects", []))
        scores = [
            1.0 if p and p.get("type") == g.get("type") else 0.0
            for g, p in pairs
        ]
        return sum(scores) / len(scores)

    def _layout_score(self, gold, pred):
        gold_elements = gold.get("objects", []) + gold.get("text", [])
        pred_elements = pred.get("objects", []) + pred.get("text", [])
        if not gold_elements:
            return 1.0
        pairs = match_by_bbox(gold_elements, pred_elements)

        class Box:
            def __init__(self, bbox):
                self.x = bbox["x"]
                self.y = bbox["y"]
                self.width = bbox["width"]
                self.height = bbox["height"]

        scores = [
            max(0.0, bbox_similarity(Box(g["bbox"]), Box(p["bbox"]))) if p else 0.0
            for g, p in pairs
        ]
        return sum(scores) / len(scores)

    def _colors_score(self, gold, pred):
        gold_colors = [c["hex"] for c in gold.get("colors", [])]
        pred_colors = [c["hex"] for c in pred.get("colors", [])]
        return color_similarity(gold_colors, pred_colors)

    def score(self, gold: dict, pred: dict) -> float:
        sub_scores = {
            "text": self._text_score(gold, pred),
            "objects": self._objects_score(gold, pred),
            "layout": self._layout_score(gold, pred),
            "colors": self._colors_score(gold, pred),
            "aesthetic": 1.0,  # placeholder until a vision-judge stage exists
        }
        return sum(sub_scores[k] * w for k, w in self.WEIGHTS.items())


def _as_scene_dict(value):
    scene = getattr(value, "scene", value)
    if isinstance(scene, str):
        return json.loads(scene)
    return scene


def parser_metric(gold, pred, trace=None) -> float:
    gold_scene = _as_scene_dict(gold)
    pred_scene = _as_scene_dict(pred)
    return SceneScorer().score(gold_scene, pred_scene)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/eval/test_scorer.py -v`
Expected: PASS

- [ ] **Step 6: Update `eval/__init__.py` export**

```python
# eval/__init__.py
from .scorer import SceneScorer, parser_metric
from .judge import VisionJudge
from .dataset import CreativeExample


__all__ = [
    "SceneScorer",
    "parser_metric",
    "VisionJudge",
    "CreativeExample"
]
```

- [ ] **Step 7: Commit**

```bash
git add eval/scorer.py eval/metrics.py eval/__init__.py tests/eval/test_scorer.py
git commit -m "Implement SceneScorer.score over matched scene elements and a parser_metric DSPy metric"
```

---

### Task 9: Wire the metric into the MIPROv2 optimizer

**Files:**
- Modify: `dspy_lab/optimizers/compile.py`
- Test: `tests/dspy_lab/test_compile.py`

**Interfaces:**
- Consumes: `eval.scorer.parser_metric` (Task 8).
- Produces: `dspy_lab.optimizers.compile.optimize(program, examples, metric=parser_metric) -> program` — `metric` now has a real default instead of `None`; still overridable.

- [ ] **Step 1: Write the failing test**

```python
# tests/dspy_lab/test_compile.py
from unittest.mock import MagicMock, patch

from dspy_lab.optimizers.compile import optimize


def test_optimize_passes_parser_metric_to_miprov2_by_default():
    fake_program = MagicMock()
    fake_examples = [MagicMock()]

    with patch("dspy_lab.optimizers.compile.dspy.MIPROv2") as mock_mipro_cls:
        mock_optimizer = MagicMock()
        mock_mipro_cls.return_value = mock_optimizer

        optimize(fake_program, fake_examples)

        from eval.scorer import parser_metric
        _, kwargs = mock_mipro_cls.call_args
        assert kwargs["metric"] is parser_metric
        mock_optimizer.compile.assert_called_once_with(fake_program, trainset=fake_examples)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/dspy_lab/test_compile.py -v`
Expected: FAIL — current code passes `metric=None`.

- [ ] **Step 3: Implement**

```python
# dspy_lab/optimizers/compile.py
import dspy

from eval.scorer import parser_metric


def optimize(program, examples, metric=None):
    optimizer = dspy.MIPROv2(
        metric=metric or parser_metric,
        auto="light"
    )

    return optimizer.compile(
        program,
        trainset=examples
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/dspy_lab/test_compile.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dspy_lab/optimizers/compile.py tests/dspy_lab/test_compile.py
git commit -m "Default MIPROv2 optimizer to parser_metric instead of None"
```

---

### Task 10: Langfuse Cloud tracing

**Files:**
- Create: `observability/langfuse_tracing.py`
- Modify: `observability/__init__.py`
- Test: `tests/observability/test_langfuse_tracing.py`

**Interfaces:**
- Consumes: `config.settings.settings.langfuse_public_key/secret_key/host` (Task 2).
- Produces: `observability.langfuse_tracing.setup() -> None`, `.is_enabled() -> bool`, `.flush() -> None` — consumed by Task 12's `scripts/run_experiment.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/observability/test_langfuse_tracing.py
from unittest.mock import MagicMock, patch

from observability import langfuse_tracing


def test_setup_is_noop_when_keys_absent(monkeypatch):
    langfuse_tracing._enabled = False
    langfuse_tracing._client = None
    monkeypatch.setattr("config.settings.settings.langfuse_public_key", None)
    monkeypatch.setattr("config.settings.settings.langfuse_secret_key", None)

    langfuse_tracing.setup()

    assert langfuse_tracing.is_enabled() is False


def test_setup_enables_when_keys_present(monkeypatch):
    langfuse_tracing._enabled = False
    langfuse_tracing._client = None
    monkeypatch.setattr("config.settings.settings.langfuse_public_key", "pk")
    monkeypatch.setattr("config.settings.settings.langfuse_secret_key", "sk")
    monkeypatch.setattr("config.settings.settings.langfuse_host", "https://cloud.langfuse.com")

    with patch("langfuse.Langfuse") as mock_langfuse_cls, \
         patch("openinference.instrumentation.dspy.DSPyInstrumentor") as mock_instrumentor_cls:
        mock_langfuse_cls.return_value = MagicMock()
        mock_instrumentor_cls.return_value = MagicMock()

        langfuse_tracing.setup()

        assert langfuse_tracing.is_enabled() is True
        mock_instrumentor_cls.return_value.instrument.assert_called_once()

    langfuse_tracing._enabled = False
    langfuse_tracing._client = None


def test_setup_never_raises_on_sdk_failure(monkeypatch):
    langfuse_tracing._enabled = False
    langfuse_tracing._client = None
    monkeypatch.setattr("config.settings.settings.langfuse_public_key", "pk")
    monkeypatch.setattr("config.settings.settings.langfuse_secret_key", "sk")

    with patch("langfuse.Langfuse", side_effect=RuntimeError("boom")):
        langfuse_tracing.setup()  # must not raise

    assert langfuse_tracing.is_enabled() is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/observability/test_langfuse_tracing.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# observability/langfuse_tracing.py
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

_enabled = False
_client = None


def setup() -> None:
    global _enabled, _client
    if _enabled:
        return

    if not (settings.langfuse_public_key and settings.langfuse_secret_key and settings.langfuse_host):
        logger.info("Langfuse disabled (LANGFUSE_* not set) — LLM tracing is a no-op")
        return

    try:
        from langfuse import Langfuse
        from openinference.instrumentation.dspy import DSPyInstrumentor

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        DSPyInstrumentor().instrument()
        _enabled = True
        logger.info("Langfuse observability enabled (host=%s)", settings.langfuse_host)
    except Exception:
        logger.exception("Langfuse init failed — continuing without LLM tracing")
        _client = None
        _enabled = False


def is_enabled() -> bool:
    return _enabled


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/observability/test_langfuse_tracing.py -v`
Expected: PASS

- [ ] **Step 5: Export from `observability/__init__.py`**

```python
# observability/__init__.py
from .logger import Logger
from .tracer import Trace
from .metrics import Metrics
from . import langfuse_tracing


__all__ = [
    "Logger",
    "Trace",
    "Metrics",
    "langfuse_tracing"
]
```

- [ ] **Step 6: Commit**

```bash
git add observability/langfuse_tracing.py observability/__init__.py tests/observability/test_langfuse_tracing.py
git commit -m "Add Langfuse Cloud tracing via openinference DSPy auto-instrumentation, no-op when unconfigured"
```

---

### Task 11: Dataset — real images + hand-authored ground truth

**Files:**
- Create: `datasets/raw/images/poster_001.png`, `datasets/raw/images/poster_002.png` (fetched)
- Create: `datasets/processed/scenes/poster_001.json`, `datasets/processed/scenes/poster_002.json` (hand-authored, matching `scene/schema.py`'s `Scene`)
- Modify: `datasets/manifests/train.json`
- Test: `tests/datasets/test_manifest.py`

**Interfaces:**
- Consumes: `datasets/schema.py`'s `CreativeSample`, `scene/schema.py`'s `Scene`.
- Produces: a manifest loadable by `datasets.loader.DatasetLoader().load(...)` where every `image_path`/`scene_path` resolves to a real file — consumed by Task 12.

- [ ] **Step 1: Find 2 public-domain poster/ad images with simple, hand-annotatable layouts**

Use `WebSearch` for something like `"public domain vintage poster wikimedia commons simple text layout"`. Prefer Wikimedia Commons results — check the file's description page confirms public domain / CC0 licensing before using. Favor posters with: a solid or simple background, 1-3 short text elements (a title and maybe a subtitle), few distinct flat colors. Avoid photo-heavy or highly detailed illustrations — they're hard to hand-annotate accurately.

- [ ] **Step 2: Download the images**

```bash
mkdir -p datasets/raw/images
curl -L -o datasets/raw/images/poster_001.png "<direct image URL found in step 1>"
curl -L -o datasets/raw/images/poster_002.png "<second direct image URL>"
```

- [ ] **Step 3: Verify the downloads are valid images and note their pixel dimensions**

```bash
uv run python3 -c "
from PIL import Image
for p in ['datasets/raw/images/poster_001.png', 'datasets/raw/images/poster_002.png']:
    with Image.open(p) as img:
        print(p, img.size, img.format)
"
```
If a file isn't a valid image (wrong URL, HTML error page downloaded instead), re-fetch with a corrected URL. If the source isn't PNG, either keep the extension accurate (e.g. `.jpg`) or convert with Pillow — just keep the manifest's `image_path` consistent with the actual file.

- [ ] **Step 4: Open each image and hand-author its ground-truth `Scene` JSON**

View the downloaded image (e.g. via the Read tool, which can display images) and note: canvas width/height (from step 3), each text element's approximate content and bounding box in pixel coordinates, and 2-4 dominant hex colors (cross-check with `extract_palette` from Task 4 by running it against the file, then adjust by eye if the auto-extracted palette misses an obviously dominant color).

```bash
mkdir -p datasets/processed/scenes
```

```json
// datasets/processed/scenes/poster_001.json — EXAMPLE SHAPE, fill with real observed values
{
  "width": 1024,
  "height": 1536,
  "background": "#f2e9dc",
  "objects": [],
  "text": [
    {
      "id": "title",
      "content": "<actual title text read off the image>",
      "bbox": {"x": 80, "y": 120, "width": 860, "height": 140},
      "font_family": "serif",
      "font_size": 96,
      "color": "#1a1a1a"
    }
  ],
  "colors": [
    {"hex": "#f2e9dc"},
    {"hex": "#1a1a1a"},
    {"hex": "#b0342d"}
  ],
  "layout": {"alignment": "center", "grid": "unknown", "spacing": {}}
}
```

Repeat for `poster_002.json` with that image's actual observed content.

- [ ] **Step 5: Update the manifest**

```json
// datasets/manifests/train.json
[
  {
    "id": "poster_001",
    "image_path": "raw/images/poster_001.png",
    "category": "poster",
    "difficulty": "easy",
    "scene_path": "processed/scenes/poster_001.json"
  },
  {
    "id": "poster_002",
    "image_path": "raw/images/poster_002.png",
    "category": "poster",
    "difficulty": "easy",
    "scene_path": "processed/scenes/poster_002.json"
  }
]
```

- [ ] **Step 6: Write the failing test**

```python
# tests/datasets/test_manifest.py
import json
from pathlib import Path

from PIL import Image

from datasets.loader import DatasetLoader
from datasets.manifest import load_manifest

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"


def test_manifest_entries_resolve_to_real_files():
    manifest = load_manifest(str(DATASETS_ROOT / "manifests" / "train.json"))
    samples = DatasetLoader().load(manifest)

    assert len(samples) >= 2
    for sample in samples:
        image_path = DATASETS_ROOT / sample.image_path
        scene_path = DATASETS_ROOT / sample.scene_path
        assert image_path.exists(), f"missing {image_path}"
        assert scene_path.exists(), f"missing {scene_path}"

        with Image.open(image_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0

        scene_data = json.loads(scene_path.read_text())
        assert scene_data["width"] > 0
        assert scene_data["height"] > 0
        assert len(scene_data["text"]) >= 1
```

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/datasets/test_manifest.py -v`
Expected: PASS. If it fails, fix whichever manifest entry/path/JSON is wrong before proceeding — this test is the hard guarantee that Task 12's experiment script has real data to run against.

- [ ] **Step 8: Commit**

```bash
git add datasets/raw/images datasets/processed/scenes datasets/manifests/train.json tests/datasets/test_manifest.py
git commit -m "Add real dataset: 2 poster images with hand-authored ground-truth scene JSON"
```

---

### Task 12: Entry point — `scripts/run_experiment.py`

**Files:**
- Create: `scripts/run_experiment.py`
- Create: `scripts/__init__.py` (empty, so it can be imported by tests)
- Test: `tests/scripts/test_run_experiment.py`

**Interfaces:**
- Consumes: everything from Tasks 1-11 (`dspy_lab.config.configure`, `dspy_lab.lm.AdapterLM`, `models.factory.build_adapter`, `config.models.VISION_MODEL`, `dspy_lab.modules.scene_parser.SceneParser`, `datasets.loader.DatasetLoader`, `datasets.manifest.load_manifest`, `eval.scorer.SceneScorer`, `experiments.tracker.ExperimentTracker`, `observability.langfuse_tracing`).
- Produces: a runnable CLI script; `run(dataset_samples, dataset_root, optimize=False) -> dict` (the testable core, called by a thin `if __name__ == "__main__":` block).

- [ ] **Step 1: Write the failing test (mocking the LM call, no real API key needed)**

```python
# tests/scripts/test_run_experiment.py
import json
from pathlib import Path
from unittest.mock import patch

from datasets.schema import CreativeSample
from scripts.run_experiment import run


def test_run_scores_each_sample_against_ground_truth(tmp_path):
    scene_json = {
        "width": 10, "height": 10, "background": "#fff",
        "objects": [], "text": [], "colors": [],
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene_json))

    image_path = tmp_path / "image.png"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(image_path)

    sample = CreativeSample(
        id="s1",
        image_path="image.png",
        category="poster",
        scene_path="scene.json",
    )

    with patch("scripts.run_experiment.SceneParser") as mock_parser_cls:
        mock_parser_cls.return_value.return_value.scene = json.dumps(scene_json)

        result = run([sample], dataset_root=tmp_path, optimize=False)

    assert result["results"][0]["sample"] == "s1"
    assert result["results"][0]["score"] > 0.9  # identical scene => near-perfect score
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/scripts/test_run_experiment.py -v`
Expected: FAIL — `scripts/run_experiment.py` doesn't exist.

- [ ] **Step 3: Implement**

```python
# scripts/__init__.py
```

```python
# scripts/run_experiment.py
import argparse
import json
from datetime import datetime
from pathlib import Path

import dspy

from config.models import VISION_MODEL
from config.settings import settings
from datasets.loader import DatasetLoader
from datasets.manifest import load_manifest
from dspy_lab.lm import AdapterLM
from dspy_lab.modules.scene_parser import SceneParser
from dspy_lab.optimizers.compile import optimize as run_optimize
from eval.scorer import SceneScorer, parser_metric
from experiments.tracker import ExperimentTracker
from models.adapters.aiohttp_client import HTTPClient
from models.factory import build_adapter
from observability import langfuse_tracing

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "datasets"


def _load_scene_json(dataset_root: Path, sample) -> dict:
    return json.loads((dataset_root / sample.scene_path).read_text())


def run(samples, dataset_root: Path, optimize: bool = False) -> dict:
    scorer = SceneScorer()
    parser = SceneParser()

    results = []
    examples = []

    for sample in samples:
        image_path = str(dataset_root / sample.image_path)
        gold_scene = _load_scene_json(dataset_root, sample)

        prediction = parser(image=image_path)
        pred_scene = json.loads(prediction.scene) if isinstance(prediction.scene, str) else prediction.scene

        score = scorer.score(gold_scene, pred_scene)
        results.append({"sample": sample.id, "score": score})

        examples.append(
            dspy.Example(image=image_path, scene=json.dumps(gold_scene)).with_inputs("image")
        )

    if optimize:
        run_optimize(parser, examples, metric=parser_metric)

    return {
        "experiment": "creative-reconstruction-v1",
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_DATASET_ROOT / "manifests" / "train.json"))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    langfuse_tracing.setup()

    client = HTTPClient()
    adapter = build_adapter(VISION_MODEL, api_key=settings.gemini_api_key, client=client)
    dspy.settings.configure(lm=AdapterLM(adapter=adapter, model_name=VISION_MODEL.name))

    manifest = load_manifest(args.manifest)
    samples = DatasetLoader().load(manifest)

    result = run(samples, dataset_root=Path(args.dataset_root), optimize=args.optimize)

    ExperimentTracker().save(result["experiment"], result)

    for r in result["results"]:
        print(f"{r['sample']}: {r['score']:.3f}")

    langfuse_tracing.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/scripts/test_run_experiment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/__init__.py tests/scripts/test_run_experiment.py
git commit -m "Add scripts/run_experiment.py: end-to-end scene-parsing experiment entry point"
```

- [ ] **Step 6: Real run (requires real API keys — manual verification, not automated)**

Once `GEMINI_API_KEY` (and `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` if you want traces) are real values in `.env`, run:

```bash
uv run python3 -m scripts.run_experiment
```

Expected: prints one score line per dataset sample, writes `experiments/results/creative-reconstruction-v1.json`, and (if Langfuse keys are set) a new trace appears in the Langfuse Cloud project within a minute or two.

---

### Task 13: Agent loop smoke test

**Files:**
- Modify: `agents/loop.py` (only if the smoke test surfaces a real bug — see Step 1)
- Test: `tests/agents/test_smoke.py`

**Interfaces:**
- Consumes: `app.factory.create_app`, `agents.loop.CreativeAgent`, all tools implemented in Tasks 4-7.
- Produces: confidence that `CreativeAgent.run(image)` completes without exceptions once its tools do real work, given a mocked planner LM (real planner LM calls are exercised manually per Task 12 Step 6, same reasoning — non-deterministic, not asserted on in CI).

- [ ] **Step 1: Write the smoke test**

```python
# tests/agents/test_smoke.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.factory import create_app


@pytest.mark.asyncio
async def test_creative_agent_runs_full_loop_without_exceptions(two_color_image):
    scene_json = {
        "width": 200, "height": 100, "background": "#ffffff",
        "objects": [], "text": [], "colors": [],
    }
    plan = [{"tool": "extract_palette", "arguments": {"image": str(two_color_image)}}]

    with patch("dspy_lab.modules.scene_parser.SceneParser.forward") as mock_parse, \
         patch("dspy_lab.modules.tool_planner.ToolPlanner.forward") as mock_plan:
        mock_parse.return_value = MagicMock(scene=json.dumps(scene_json))
        mock_plan.return_value = MagicMock(plan=plan)

        agent = create_app()
        state = await agent.run(str(two_color_image))

    assert state.finished is True
    assert state.iteration >= 1
    assert len(state.tool_results) >= 1
    assert state.tool_results[0]["tool"] == "extract_palette"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/agents/test_smoke.py -v`
Expected: likely FAIL on the first attempt — `AgentPlanner.create_plan` (`agents/planner.py`) accesses `state.scene` (a dict, per `AgentState.scene: dict`), but `agents/loop.py:33-36` sets `state.scene = self.parser(image).scene`, which is the *string* from `SceneParser`'s output field, not a parsed dict. This is a real bug in the existing skeleton, not a test-authoring mistake — fix it.

- [ ] **Step 3: Fix `agents/loop.py` to parse the scene string into a dict**

```python
# agents/loop.py
import json

from .state import AgentState


class CreativeAgent:

    def __init__(self, parser, planner, executor, evaluator):
        self.parser = parser
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator

    async def run(self, image):
        state = AgentState(task_id="run", image=image)

        raw_scene = self.parser(image).scene
        state.scene = json.loads(raw_scene) if isinstance(raw_scene, str) else raw_scene

        while not state.finished:
            state.plan = await self.planner.create_plan(state)

            results = await self.executor.execute(state.plan)

            state.tool_results.extend(results)

            state.iteration += 1

            if state.iteration >= 5:
                state.finished = True

        return state
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/agents/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/loop.py tests/agents/test_smoke.py
git commit -m "Fix CreativeAgent to parse SceneParser's JSON string output; add agent-loop smoke test"
```

---

### Task 14: Full test suite + README pointer

**Files:**
- Modify: `README.md` (create if absent)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing consumed by other tasks — final verification + a pointer for future runs.

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests PASS (integration-style tests that need real API keys are either mocked per the tasks above, or — if any were added expecting real keys — they must be marked `@pytest.mark.skipif` on `settings.gemini_api_key in (None, "xxx")` per the Global Constraints; none should hard-fail in an environment with placeholder keys).

- [ ] **Step 2: Write a short README**

```markdown
# DSPy Lab — Visual Editing

Experimental setup: parse a creative (poster/ad) image into a structured
`Scene` graph via a Gemini-backed DSPy `SceneParser`, score it against
hand-authored ground truth, and optimize the parser with `dspy.MIPROv2`.

## Setup

```bash
uv sync
cp .env-example .env  # fill in real OPENROUTER_API_KEY, GEMINI_API_KEY,
                       # and LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY
```

## Run

```bash
uv run python3 -m scripts.run_experiment            # score only
uv run python3 -m scripts.run_experiment --optimize # + MIPROv2 compile
```

## Test

```bash
uv run pytest
```

See `docs/superpowers/specs/2026-08-07-dspy-lab-experimental-setup-design.md`
for the design and its explicitly out-of-scope future work (hard
constraints, vision-judge scoring, reference similarity, search/branching).
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README with setup/run/test instructions"
```

---

## Self-Review Notes

- Spec section 1 (foundation) → Task 1. Section 2 (rendering) → Task 7. Section 3 (vision tools) → Tasks 4 & 6. Section 4 (scene parsing/adapters) → Task 3. Section 5 (dataset) → Task 11. Section 6 (scoring/metric) → Task 8, wired in Task 9. Section 7 (Langfuse) → Task 10. Section 8 (entry point) → Task 12. Section 9 (agent loop smoke test) → Task 13. "Out of scope" items are not implemented anywhere in this plan, matching the spec.
- Fixed one inconsistency during drafting: `tools/generation/screenshot.py` initially referenced `storage.paths.ROOT` (which is `Path("runs")`, run-specific) — corrected to `config.paths.ARTIFACTS` (already defined for exactly this purpose) in Task 7 Step 5, with the test's monkeypatch target corrected to match.
- Task 13 documents a real latent bug in `agents/loop.py` (assigns the parser's raw JSON string to `state.scene`, which `AgentPlanner`/tools then expect to be a dict) discovered by writing the smoke test — not a placeholder, an actual fix with its own test.
