from . import registry

# tools/vision, tools/design, tools/generation, and tools/evaluation are
# namespace packages (no __init__.py), so importing them alone does not
# import their submodules and therefore never runs the @register
# decorators inside. Import each submodule explicitly so every tool is
# registered as soon as `tools` (or anything that imports it, e.g.
# tools.runner) is imported.
from .vision import image_analysis, objects, ocr
from .design import layout, palette, typography
from .generation import html, screenshot, svg
from .evaluation import compare, score


__all__ = [
    "registry"
]
