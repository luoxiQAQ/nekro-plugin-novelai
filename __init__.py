import sys

for _stale in [name for name in sys.modules if name.startswith(f"{__name__}.")]:
    sys.modules.pop(_stale, None)

try:
    from .plugin import plugin
except ModuleNotFoundError as exc:
    optional_runtime_modules = ("nekro_agent", "httpx")
    if not (exc.name or "").startswith(optional_runtime_modules):
        raise
    plugin = None

__all__ = ["plugin"]
