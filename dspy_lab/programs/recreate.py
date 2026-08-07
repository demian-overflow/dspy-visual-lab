from ..modules.pipeline import CreativePipeline


pipeline = CreativePipeline()


def run(image):

    result = pipeline(
        image=image
    )

    return result
