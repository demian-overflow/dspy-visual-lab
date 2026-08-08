import json
from unittest.mock import MagicMock, patch

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

    # SceneParser.forward returns `scene` as a JSON string (see
    # dspy_lab/signatures/scene.py); CreativeAgent.run must parse it into
    # a dict before storing it on AgentState (state.scene: dict). Without
    # that parsing this assertion catches the regression even though the
    # mocked ToolPlanner.forward above would silently accept either shape.
    assert isinstance(state.scene, dict)
    assert state.scene == scene_json
