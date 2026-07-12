"""Query the registry by capability instead of by model name."""

from onramp import get_registry

registry = get_registry()

print("All registered models:", registry.model_ids())
print("Models with >=90% JSON reliability (cheapest first):",
      registry.find(json_reliability=0.9))
print("Long-context + reliable-tools models:",
      registry.find(usable_context_tokens=64_000, tool_use_reliability=0.9))
