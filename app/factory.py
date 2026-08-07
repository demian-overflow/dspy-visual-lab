from agents import (
    CreativeAgent,
    AgentPlanner,
    AgentExecutor
)

from dspy.modules.scene_parser import SceneParser

from eval import SceneScorer



def create_app():

    parser = SceneParser()

    planner = AgentPlanner()

    executor = AgentExecutor()

    evaluator = SceneScorer()


    agent = CreativeAgent(
        parser=parser,
        planner=planner,
        executor=executor,
        evaluator=evaluator
    )


    return agent
