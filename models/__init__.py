# NOTE: `registry.py`, `router.py`, and `capabilities.py` are scaffolded for
# later tasks in the dspy-lab-experimental-setup plan and do not exist yet.
# Importing them eagerly here breaks `import models` (and anything importing
# a submodule, e.g. `models.factory`, `models.adapters.*`) until those tasks
# land. Re-add the imports/`__all__` entries below as each module is
# implemented:
#
# from .registry import load
# from .router import Router
# from .capabilities import Capability
#
# __all__ = ["load", "Router", "Capability"]
