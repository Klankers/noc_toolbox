import os
import importlib
from .base_step import BaseStep, REGISTERED_STEPS

STEP_CLASSES = {}
STEP_DEPENDENCIES = {
    "QC: Salinity": ["Load OG1"],
}


def discover_steps():
    """Import all custom step modules to populate the decorator-based registry."""
    custom_dir = os.path.join(os.path.dirname(__file__), "custom")
    module_files = [
        f for f in os.listdir(custom_dir) if f.endswith(".py") and f != "__init__.py"
    ]

    print(f"Discovered step modules: {module_files}")
    for module_file in module_files:
        module_name = module_file[:-3]
        importlib.import_module(f".custom.{module_name}", package="steps")

    # Populate STEP_CLASSES using decorator-based registry
    STEP_CLASSES.update(REGISTERED_STEPS)
    for step_name in STEP_CLASSES:
        print(f"Registered step: {step_name}")


# Automatically discover on import
discover_steps()


def create_step(step_config, _context):
    step_name = step_config["name"]
    step_class = STEP_CLASSES.get(step_name)
    if not step_class:
        raise ValueError(
            f"Step '{step_name}' not recognized or missing @register_step."
        )

    return step_class(
        name=step_name,
        parameters=step_config.get("parameters", {}),
        diagnostics=step_config.get("diagnostics", False),
        context=_context,
    )
