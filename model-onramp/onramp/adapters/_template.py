"""Template adapter — the entire cost of onboarding a new model.

1. cp _template.py my_new_model.py
2. Fill in model_id, provider, pricing, and the two calls below.
3. python -m onramp probe <model-id>

Do NOT declare capabilities here (context length, JSON support, ...) — the
probe suite measures those and writes the manifest.
"""

from ..adapter import Pricing

# from ..registry import register
# @register  # uncomment in your copy; the template itself must not register


class TemplateAdapter:
    model_id = "my-new-model"
    provider = "my-provider"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def generate(self, prompt: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        return self.chat([{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature)

    def chat(self, messages: list, *, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        raise NotImplementedError("call your provider's SDK/API here")
