try:
    from .plugin import plugin
except ModuleNotFoundError as exc:
    optional_runtime_modules = ("nekro_agent", "httpx")
    if not (exc.name or "").startswith(optional_runtime_modules):
        raise
    plugin = None

__all__ = ["plugin"]
