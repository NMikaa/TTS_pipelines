"""Stage registry — lazy loading to avoid importing heavy deps at discovery time."""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict

# Registry: stage_name -> module (lazy loaded)
STAGE_REGISTRY: Dict[str, object] = {}
_STAGE_MODULES: Dict[str, str] = {}  # name -> module_name (for lazy load)


def _discover_stages():
    """Scan package for stage modules without importing them."""
    package_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        # Read NAME from the file without importing
        module_path = package_dir / f"{module_name}.py"
        with open(module_path) as f:
            for line in f:
                if line.startswith("NAME"):
                    # Extract NAME = "..." value
                    name = line.split("=", 1)[1].strip().strip('"').strip("'")
                    _STAGE_MODULES[name] = module_name
                    break


def _load_stage(name: str):
    """Lazily import a stage module."""
    if name in STAGE_REGISTRY:
        return STAGE_REGISTRY[name]
    if name not in _STAGE_MODULES:
        raise ValueError(f"Unknown stage '{name}'. Available: {list(_STAGE_MODULES.keys())}")
    module = importlib.import_module(f".{_STAGE_MODULES[name]}", package=__name__)
    STAGE_REGISTRY[name] = module
    return module


def get_stage(name: str):
    """Get a stage module by name (lazy-loads on first access)."""
    return _load_stage(name)


def available_stages():
    """Return list of all discovered stage names."""
    return list(_STAGE_MODULES.keys())


_discover_stages()
