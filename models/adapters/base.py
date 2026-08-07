from abc import ABC, abstractmethod


class BaseAdapter(ABC):

    @abstractmethod
    async def generate(
        self,
        messages,
        **kwargs
    ):
        pass


    async def vision(
        self,
        image,
        prompt,
        **kwargs
    ):
        raise NotImplementedError


    async def tool_call(
        self,
        messages,
        tools,
        **kwargs
    ):
        raise NotImplementedError
