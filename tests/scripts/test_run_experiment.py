import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets.schema import CreativeSample
from scripts.run_experiment import run


def _make_sample(tmp_path, scene_json):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene_json))

    image_path = tmp_path / "image.png"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(image_path)

    return CreativeSample(
        id="s1",
        image_path="image.png",
        category="poster",
        scene_path="scene.json",
    )


def test_run_scores_each_sample_against_ground_truth(tmp_path):
    scene_json = {
        "width": 10, "height": 10, "background": "#fff",
        "objects": [], "text": [], "colors": [{"hex": "#ffffff"}],
    }
    sample = _make_sample(tmp_path, scene_json)

    with patch("scripts.run_experiment.SceneParser") as mock_parser_cls:
        mock_parser_cls.return_value.return_value.scene = json.dumps(scene_json)

        result = run([sample], dataset_root=tmp_path, optimize=False)

    assert result["results"][0]["sample"] == "s1"
    assert result["results"][0]["score"] > 0.9  # identical scene => near-perfect score
    assert "optimized_results" not in result


def test_run_with_optimize_rescoring_the_optimized_program(tmp_path):
    scene_json = {
        "width": 10, "height": 10, "background": "#fff",
        "objects": [], "text": [{"id": "t1", "content": "hi", "bbox": {"x": 0, "y": 0, "width": 1, "height": 1}}],
        "colors": [{"hex": "#ffffff"}],
    }
    sample = _make_sample(tmp_path, scene_json)

    with patch("scripts.run_experiment.SceneParser") as mock_parser_cls, \
         patch("scripts.run_experiment.run_optimize") as mock_optimize:
        # baseline parser predicts a perfect match
        mock_parser_cls.return_value.return_value.scene = json.dumps(scene_json)

        # the optimized program predicts something worse (no text), so the
        # test can tell the two scoring passes apart -- if run() silently
        # discarded the optimized program (the prior bug), both scores
        # would come out identical.
        worse_scene = {**scene_json, "text": []}
        mock_optimized_program = MagicMock()
        mock_optimized_program.return_value.scene = json.dumps(worse_scene)
        mock_optimize.return_value = mock_optimized_program

        result = run([sample], dataset_root=tmp_path, optimize=True)

    mock_optimize.assert_called_once()
    assert "optimized_results" in result
    assert result["optimized_results"][0]["sample"] == "s1"
    assert result["optimized_results"][0]["score"] < result["results"][0]["score"]
