from .factory import create_app


class CreativePipeline:


    def __init__(self):

        self.agent = create_app()



    async def recreate(
        self,
        image
    ):

        result = await self.agent.run(
            image
        )

        return result
