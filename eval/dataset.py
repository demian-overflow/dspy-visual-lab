from dataclasses import dataclass


@dataclass
class CreativeExample:

    image:str

    scene:dict

    expected_tools:list

    reference_output:str
