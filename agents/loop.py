from .state import AgentState


class CreativeAgent:


    def __init__(
        self,
        parser,
        planner,
        executor,
        evaluator
    ):

        self.parser = parser
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator



    async def run(
        self,
        image
    ):

        state = AgentState(
            task_id="run",
            image=image
        )


        state.scene = (
            self.parser(image)
            .scene
        )


        while not state.finished:


            state.plan = await self.planner.create_plan(
                state
            )


            results = await self.executor.execute(
                state.plan
            )


            state.tool_results.extend(
                results
            )


            state.iteration += 1


            if state.iteration >= 5:
                state.finished = True


        return state
