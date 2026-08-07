import dspy


class AdapterLM(dspy.LM):

    def __init__(
        self,
        adapter,
        model_name
    ):
        self.adapter = adapter
        self.model = model_name


    async def aforward(
        self,
        messages,
        **kwargs
    ):

        return await self.adapter.generate(
            messages,
            **kwargs
        )
