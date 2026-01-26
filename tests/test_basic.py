from datetime import timezone, datetime
import os

this_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(this_dir, "ttally_config.py")

os.environ["TTALLY_CFG"] = config_file
os.environ["TTALLY_SKIP_DEFAULT_IMPORT"] = "1"

import ttally.core

ext = ttally.core.Extension(data_dir=this_dir)


def test_config() -> None:
    assert len(ext.MODELS) == 1
    assert list(ext.MODELS.keys()) == ["self"]


def test_ttally_when_import() -> None:
    import ttally.when  # noqa


def test_load_yaml() -> None:
    self_type = list(ext.MODELS.values())[0]
    values = ext.glob_namedtuple_by_datetime(self_type)
    assert values == [
        self_type(
            when=datetime(2026, 1, 26, 5, 1, 33, tzinfo=timezone.utc),
            what=self_type.__annotations__["what"].one,
        )
    ]
