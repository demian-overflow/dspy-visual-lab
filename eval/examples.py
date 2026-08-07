from .dataset import CreativeExample


examples = [

    CreativeExample(

        image="poster1.png",

        scene={
            "type":"poster"
        },

        expected_tools=[
            "ocr",
            "extract_palette",
            "generate_svg"
        ],

        reference_output=
            "poster1.html"
    )

]
