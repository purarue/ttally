import os

this_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(this_dir, "ttally_config.py")

os.environ["TTALLY_CFG"] = config_file
os.environ["TTALLY_SKIP_DEFAULT_IMPORT"] = "1"

from click.testing import CliRunner

import ttally.core
from ttally.main import wrap_accessor


def _group(tmp_path):
    ext = ttally.core.Extension(config_file=config_file, data_dir=str(tmp_path))
    return wrap_accessor(extension=ext)


def test_merge_without_datafiles_is_an_error(tmp_path) -> None:
    result = CliRunner().invoke(_group(tmp_path), ["merge", "self"])
    assert result.exit_code == 1
    assert "No datafiles for model self" in result.output
