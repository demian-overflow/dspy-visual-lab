from dspy_lab.modules.tool_planner import ToolPlanner


class AgentPlanner:


    def __init__(
        self,
        planner=None
    ):

        self.planner = (
            planner
            or ToolPlanner()
        )


    async def create_plan(
        self,
        state
    ):

        result = self.planner(
            scene=state.scene
        )

        return result.plan
