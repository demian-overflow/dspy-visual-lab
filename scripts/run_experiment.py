import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import dspy
import pydantic

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
from scene.schema import Scene

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "datasets"


def _load_scene_json(dataset_root: Path, sample) -> dict:
    return json.loads((dataset_root / sample.scene_path).read_text())


def _parse_with_retry(parser, image, sample_id, max_attempts=3):
    """Call `parser(image=...)`, retrying on structured-output validation
    failures.

    Smaller/free vision models occasionally drift on ExtractScene.scene's
    typed schema (e.g. a bbox as a bare [x,y,w,h] list instead of
    {x,y,width,height}) -- a real, observed failure mode, not
    hypothetical. One bad response on one sample must not abort scoring
    the rest of the dataset.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return parser(image=image)
        except pydantic.ValidationError as exc:
            last_exc = exc
            print(
                f"  {sample_id}: attempt {attempt}/{max_attempts} returned an invalid Scene, retrying",
                file=sys.stderr,
            )
    raise last_exc


def _score_samples(parser, scorer, samples, dataset_root: Path):
    """Run `parser` over every sample and score it against gold.

    Returns (results, examples) -- `examples` carries real Scene instances
    (matching ExtractScene.scene's typed field) so it can double as a
    MIPROv2 trainset without a second image-loading pass. A sample whose
    parser output never validates after retrying is skipped (logged, not
    silently dropped) rather than crashing the whole run.
    """
    results = []
    examples = []

    for sample in samples:
        # dspy.Image carries the actual image bytes to the model; a bare path
        # string would be sent as text.
        image = dspy.Image.from_path(str(dataset_root / sample.image_path))
        gold_scene = _load_scene_json(dataset_root, sample)

        try:
            prediction = _parse_with_retry(parser, image, sample.id)
        except pydantic.ValidationError:
            print(f"  {sample.id}: giving up after retries, skipping this sample", file=sys.stderr)
            continue

        pred_scene = json.loads(prediction.scene) if isinstance(prediction.scene, str) else prediction.scene

        score = scorer.score(gold_scene, pred_scene)
        results.append({"sample": sample.id, "score": score})

        examples.append(
            dspy.Example(image=image, scene=Scene(**gold_scene)).with_inputs("image")
        )

    return results, examples


def run(samples, dataset_root: Path, optimize: bool = False) -> dict:
    scorer = SceneScorer()
    parser = SceneParser()

    results, examples = _score_samples(parser, scorer, samples, dataset_root)

    output = {
        "experiment": "creative-reconstruction-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    if optimize:
        # Re-use `examples` (real Scene instances, already built above) as
        # the MIPROv2 trainset, then re-score every sample with the
        # *returned* optimized program -- the whole point of --optimize is
        # to see whether the optimized prompt actually scores higher, so
        # discarding the compiled program here would make this flag a
        # no-op.
        optimized_parser = run_optimize(parser, examples, metric=parser_metric)
        optimized_results, _ = _score_samples(optimized_parser, scorer, samples, dataset_root)
        output["optimized_results"] = optimized_results

    return output


async def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--manifest", default=str(DEFAULT_DATASET_ROOT / "manifests" / "train.json"))
    arg_parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    arg_parser.add_argument("--optimize", action="store_true")
    args = arg_parser.parse_args()

    langfuse_tracing.setup()

    # HTTPClient wraps an aiohttp.ClientSession, which must be constructed
    # inside a running event loop -- hence main() being async.
    client = HTTPClient()
    try:
        adapter = build_adapter(VISION_MODEL, api_key=settings.api_key_for(VISION_MODEL.provider), client=client)
        dspy.settings.configure(lm=AdapterLM(adapter=adapter, model_name=VISION_MODEL.name))

        manifest = load_manifest(args.manifest)
        samples = DatasetLoader().load(manifest)

        # `run` is sync; AdapterLM bridges its LM calls onto a background loop,
        # so calling it from here does not clash with this running loop.
        result = run(samples, dataset_root=Path(args.dataset_root), optimize=args.optimize)

        ExperimentTracker().save(result["experiment"], result)

        if "optimized_results" in result:
            optimized_by_sample = {r["sample"]: r["score"] for r in result["optimized_results"]}
            print(f"{'sample':<14}{'before':>10}{'after':>10}{'diff':>10}")
            paired = []
            for r in result["results"]:
                after = optimized_by_sample.get(r["sample"])
                if after is None:
                    print(f"{r['sample']:<14}{r['score']:>10.3f}{'skipped':>10}{'':>10}")
                    continue
                paired.append((r["score"], after))
                print(f"{r['sample']:<14}{r['score']:>10.3f}{after:>10.3f}{after - r['score']:>+10.3f}")
            if paired:
                before_avg = sum(b for b, _ in paired) / len(paired)
                after_avg = sum(a for _, a in paired) / len(paired)
                print(f"{'avg':<14}{before_avg:>10.3f}{after_avg:>10.3f}{after_avg - before_avg:>+10.3f}")
        else:
            for r in result["results"]:
                print(f"{r['sample']}: {r['score']:.3f}")

        langfuse_tracing.flush()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
