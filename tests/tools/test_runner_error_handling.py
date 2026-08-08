import pytest

from tools.registry import TOOLS, call_tool, register
from tools.runner import ToolRunner


@pytest.mark.asyncio
async def test_call_tool_raises_clear_error_for_unknown_tool():
    with pytest.raises(KeyError, match="unknown tool"):
        await call_tool("does_not_exist")


@pytest.mark.asyncio
async def test_runner_records_a_failed_step_instead_of_aborting_the_plan():
    @register("_boom", "raises for testing")
    async def _boom(**kwargs):
        raise ValueError("simulated tool failure")

    try:
        plan = [
            {"tool": "_boom", "arguments": {}},
            {"tool": "_boom", "arguments": {}},
        ]
        results = await ToolRunner().execute(plan)
    finally:
        TOOLS.pop("_boom", None)

    assert len(results) == 2
    for result in results:
        assert result["tool"] == "_boom"
        assert result["success"] is False
        assert "simulated tool failure" in result["error"]


@pytest.mark.asyncio
async def test_runner_marks_successful_steps_accordingly():
    @register("_ok", "succeeds for testing")
    async def _ok(**kwargs):
        return {"value": 42}

    try:
        results = await ToolRunner().execute([{"tool": "_ok", "arguments": {}}])
    finally:
        TOOLS.pop("_ok", None)

    assert results[0]["success"] is True
    assert results[0]["result"] == {"value": 42}
