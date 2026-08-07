from dataclasses import dataclass


@dataclass
class AppContext:

    agent: object

    evaluator: object

    storage: object
