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

    @staticmethod
    def _canvas(gold, gold_elements, axis):
        """Canvas extent to normalize layout deltas against.

        Prefers the gold scene's declared size; falls back to the furthest
        gold element edge so scenes without width/height still score sanely.
        """
        declared = gold.get(axis)
        if declared:
            return declared

        size = "width" if axis == "width" else "height"
        origin = "x" if axis == "width" else "y"
        extents = [e["bbox"][origin] + e["bbox"][size] for e in gold_elements]
        return max(extents) or 1

    def _layout_score(self, gold, pred):
        gold_elements = gold.get("objects", []) + gold.get("text", [])
        pred_elements = pred.get("objects", []) + pred.get("text", [])
        if not gold_elements:
            return 1.0
        pairs = match_by_bbox(gold_elements, pred_elements)

        canvas_width = self._canvas(gold, gold_elements, "width")
        canvas_height = self._canvas(gold, gold_elements, "height")

        scores = [
            max(0.0, bbox_similarity(g["bbox"], p["bbox"], canvas_width, canvas_height))
            if p
            else 0.0
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
    if hasattr(scene, "model_dump"):
        # A real Scene pydantic instance (ExtractScene.scene is typed Scene,
        # so a dspy.Example/dspy.Prediction's .scene may be one directly)
        # rather than a dict or JSON string.
        return scene.model_dump()
    return scene


def parser_metric(gold, pred, trace=None) -> float:
    gold_scene = _as_scene_dict(gold)
    pred_scene = _as_scene_dict(pred)
    return SceneScorer().score(gold_scene, pred_scene)
