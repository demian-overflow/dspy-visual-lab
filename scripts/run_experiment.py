import argparse
import json
from datetime import datetime, timezone
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--manifest", default=str(DEFAULT_DATASET_ROOT / "manifests" / "train.json"))
    arg_parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    arg_parser.add_argument("--optimize", action="store_true")
    args = arg_parser.parse_args()

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
