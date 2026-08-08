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
