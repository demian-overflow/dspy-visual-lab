from .registry import call_tool


class ToolRunner:


    async def execute(self, plan):

        results = []

        for step in plan:

            result = await call_tool(
                step["tool"],
                **step["arguments"]
            )

            results.append({
                "tool": step["tool"],
                "result": result
            })

        return results
