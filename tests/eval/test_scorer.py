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
