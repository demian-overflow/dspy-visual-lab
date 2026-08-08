import dspy

from scene.schema import Scene


class ExtractScene(dspy.Signature):

    """
    Analyze an image and produce a structured Scene description: canvas
    size, background color, visual objects, text elements (verbatim
    content + pixel bounding box), dominant colors, and layout.
    """

    image: dspy.Image = dspy.InputField(
        desc="input image"
    )

    # A typed pydantic output field (rather than a string field asking the
    # model to hand-write JSON) lets DSPy's adapter drive structured-output
    # formatting/parsing directly against Scene's real schema. Verified
    # live: asking a small free vision model to emit "a JSON object roughly
    # matching this shape" into a str field produced prose or malformed
    # JSON; the same model reliably fills this typed field correctly.
    scene: Scene = dspy.OutputField(
        desc="the scene depicted in the image"
    )
