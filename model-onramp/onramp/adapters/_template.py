"""Template adapter — the entire cost of onboarding a new model.

1. cp _template.py my_new_model.py
2. Fill in model_id, provider, pricing, and _complete().
3. python -m onramp probe <model-id>

Do NOT declare capabilities here (context length, JSON support, ...) — the
probe suite measures those and writes the manifest.
"""

from ..adapter import AdapterBase, Pricing

# from ..registry import register
# @register  # uncomment in your copy; the template itself must not register


class TemplateAdapter(AdapterBase):
    model_id = "my-new-model"
    provider = "my-provider"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def _complete(self, messages: list[dict], max_tokens: int,
                  temperature: float) -> tuple[str, int | None, int | None]:
        # Call your provider here. Return (text, input_tokens, output_tokens);
        # pass None for token counts if the provider doesn't report usage.
        raise NotImplementedError("call your provider's SDK/API here")
