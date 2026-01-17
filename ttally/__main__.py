from typing import NamedTuple
from collections.abc import Iterator, Callable

from .core import Extension

ext = Extension()


def __getattr__(name: str) -> Callable[[], Iterator[NamedTuple]]:
    """
    use with hpi query, like:
    hpi query ttally.__main__.food
    """
    return ext.funccreator()(name)


def main() -> None:
    ext.wrap_cli(prog_name="ttally")


if __name__ == "__main__":
    main()
