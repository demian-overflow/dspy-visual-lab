import dspy


class ExtractScene(dspy.Signature):

    """
    Analyze an image and produce
    structured scene JSON.
    """

    image: dspy.Image = dspy.InputField(
        desc="input image"
    )


    scene = dspy.OutputField(
        desc="""
        Return ONLY a JSON object matching this EXACT shape -- these are the
        only keys and field names allowed, do not invent alternative names
        (e.g. never "label" for a type, never "bbox_2d" for a bounding box):

        {
          "width": <image pixel width, integer>,
          "height": <image pixel height, integer>,
          "background": "<dominant background color as a hex string, e.g. #ffffff>",
          "objects": [
            {"id": "<short unique string id>", "type": "<what kind of object, e.g. logo/illustration/icon>",
             "bbox": {"x": <int>, "y": <int>, "width": <int>, "height": <int>}}
          ],
          "text": [
            {"id": "<short unique string id>", "content": "<the literal text>",
             "bbox": {"x": <int>, "y": <int>, "width": <int>, "height": <int>},
             "color": "<hex string or null>"}
          ],
          "colors": [
            {"hex": "<hex string, e.g. #1a191b>"}
          ],
          "layout": {"alignment": "<left|center|right|mixed|null>", "grid": "<string or null>"}
        }

        Every bounding box MUST be an object with numeric x/y/width/height
        keys in pixel coordinates -- never a bare array like [x, y, w, h].
        "colors" MUST be a list of {"hex": ...} objects -- never bare color
        name strings like "blue". "text" and "layout" MUST be the object
        shapes above -- never a plain sentence string summarizing them.
        Omit "objects"/"text"/"colors" entries you are not confident about
        rather than guessing; an empty list is valid.
        """
    )
