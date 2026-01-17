# hook to load information from tally configuration

import sys
import os
import importlib.util
from typing import Any
from collections.abc import Callable


LOADED: set[str] = set()


def load_config_module(
    file: str,
    module_name: str,
    test_import_func: Callable[[], None],
    URL: str | None = None,
) -> Any:
    if module_name in LOADED:
        return sys.modules[module_name]

    # reload?
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        spec = importlib.util.spec_from_file_location(module_name, file)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert (
            spec.loader is not None
        ), "Error importing configuration, must be on python3.5+?"
        spec.loader.exec_module(mod)
        if module_name not in sys.modules:
            sys.modules[module_name] = mod

        # make sure its importable
        test_import_func()
        LOADED.add(module_name)
        return mod
    except ImportError as e:
        raise ImportError(
            f"""Importing '{module_name}' from '{file}' failed! (error: {e}).""" + ""
            if URL is None
            else f"""\nSee {URL} for more information"""
        )


# when 'ttally' module is imported, run default ttally.config
def setup_ttally_config() -> None:
    # if user doesn't want this to happen, set this envvar
    if "TTALLY_SKIP_DEFAULT_IMPORT" not in os.environ:
        from .core import Extension

        Extension().import_config()


setup_ttally_config()
del setup_ttally_config
