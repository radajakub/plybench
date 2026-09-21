__all__ = ["PlyBench"]  # pyright: ignore[reportUnsupportedDunderAll] -- exposed by __getattr__ below


# Lazily expose PlyBench so `from plybench import PlyBench` works (the intended entry point when using
# the package as an application) WITHOUT importing the whole graph -- app -> registry/games -> engine ->
# pyspiel, and the LLM providers -- on every `import plybench.<submodule>`. Internal code imports
# submodules directly, so it stays decoupled and light. This is the standard PEP 562 module __getattr__.
def __getattr__(name: str) -> object:
    if name == "PlyBench":
        from plybench.app import PlyBench

        return PlyBench
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
