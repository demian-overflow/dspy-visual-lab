import asyncio

import dspy


class AdapterLM(dspy.LM):

    forward_contract = "legacy"

    def __init__(self, adapter, model_name):
        super().__init__(model=model_name, cache=False)
        self.adapter = adapter
        self.model = model_name

    @staticmethod
    def _extract_text(raw_response):
        candidates = raw_response.get("candidates")
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)

        choices = raw_response.get("choices")
        if choices:
            return choices[0].get("message", {}).get("content", "")

        raise ValueError(f"Unrecognized adapter response shape: {raw_response!r}")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]

        raw_response = await self.adapter.generate(messages, **kwargs)
        text = self._extract_text(raw_response)

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "model": self.model,
        }

    def forward(self, prompt=None, messages=None, **kwargs):
        return asyncio.run(self.aforward(prompt=prompt, messages=messages, **kwargs))
