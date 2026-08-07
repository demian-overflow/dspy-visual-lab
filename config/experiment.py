from dataclasses import dataclass


@dataclass
class ExperimentConfig:


    name: str


    optimizer: str = "MIPROv2"


    train_size: int = 50


    iterations: int = 10


    metric_weights = {

        "layout":0.3,

        "text":0.2,

        "color":0.2,

        "vision":0.3
    }



default_experiment = ExperimentConfig(
    name="creative-reconstruction-v1"
)
