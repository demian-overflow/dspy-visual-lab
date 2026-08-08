from .registry import call_tool


class ToolRunner:


    async def execute(self, plan):

        results = []

        for step in plan:

            try:
                result = await call_tool(
                    step["tool"],
                    **step["arguments"]
                )
                results.append({
                    "tool": step["tool"],
                    "result": result,
                    "success": True,
                })
            except Exception as exc:
                # One bad step (unknown tool, malformed LLM output, network
                # failure) must not abort the whole plan -- record it and
                # let the caller/agent loop decide what to do next.
                results.append({
                    "tool": step["tool"],
                    "result": None,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return results
