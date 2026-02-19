import os
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from lsst_survey_sim import simulate_lsst


@contextmanager
def temp_cwd() -> Iterator:
    with TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            yield
        finally:
            os.chdir(old_cwd)


class TestSimulateLsst(unittest.TestCase):
    def setUp(self) -> None:
        with temp_cwd():
            # setup also tests pulling the git repo
            # Get git repo for scheduler directory
            config_dir = "ts_config_scheduler"
            simulate_lsst.get_config_repo(
                ts_config_scheduler_commit="develop",
                clone_path=config_dir,
            )
            self.assertTrue(os.path.isdir(config_dir))


if __name__ == "__main__":
    unittest.main()
