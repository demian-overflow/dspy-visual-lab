TOOLS = {}


def register(name, description):

    def wrapper(fn):

        TOOLS[name] = {
            "function": fn,
            "description": description
        }

        return fn

    return wrapper


def list_tools():

    return [
        {
            "name": name,
            "description": data["description"]
        }
        for name, data in TOOLS.items()
    ]


async def call_tool(name, **kwargs):

    if name not in TOOLS:
        raise KeyError(
            f"call_tool: unknown tool {name!r} (registered: {sorted(TOOLS)})"
        )

    tool = TOOLS[name]

    return await tool["function"](
        **kwargs
    )
