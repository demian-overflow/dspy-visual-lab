from tools.runner import ToolRunner


class AgentExecutor:


    def __init__(
        self
    ):

        self.runner = ToolRunner()


    async def execute(
        self,
        plan
    ):

        return await self.runner.execute(
            plan
        )
