from datetime import datetime
from typing import NamedTuple
from enum import Enum

SelfT = Enum("SelfT", ["one", "two"])


class Self(NamedTuple):
    when: datetime
    what: SelfT  # type: ignore
