"""Query the registry by capability instead of by model name."""

from onramp import get_registry

registry = get_registry()

print("All registered models:", registry.model_ids())

reliable_json = registry.find(json_reliability=0.9)
print("Models with >=90% JSON reliability (cheapest first):", reliable_json)
