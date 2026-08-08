# DSPy Lab — Visual Editing

Experimental setup: parse a creative (poster/ad) image into a structured
`Scene` graph via a Gemini-backed DSPy `SceneParser`, score it against
hand-authored ground truth, and optimize the parser with `dspy.MIPROv2`.

## Setup

```bash
uv sync
cp .env-example .env  # fill in real OPENROUTER_API_KEY, GEMINI_API_KEY,
                       # and LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY
```

See [`docs/ENV_SETUP.md`](docs/ENV_SETUP.md) for where to get each key and
what each one is used for.

## Run

```bash
uv run python3 -m scripts.run_experiment            # score only
uv run python3 -m scripts.run_experiment --optimize # + MIPROv2 compile
```

## Test

```bash
uv run pytest
```

See `docs/superpowers/specs/2026-08-07-dspy-lab-experimental-setup-design.md`
for the design and its explicitly out-of-scope future work (hard
constraints, vision-judge scoring, reference similarity, search/branching).
