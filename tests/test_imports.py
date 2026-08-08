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
