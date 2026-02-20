import os
import pickle
import subprocess
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import rubin_nights.dayobs_utils as rn_dayobs
from astropy.time import Time
from rubin_scheduler.scheduler.model_observatory.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler
from rubin_scheduler.scheduler.schedulers.filter_scheduler import BandSwapScheduler
from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.utils import SURVEY_START_MJD

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


class TestCLI(unittest.TestCase):

    @unittest.skip("redundant")
    def test_make_scheduler_cli(self) -> None:
        with temp_cwd():
            scheduler_pickle = "scheduler.p"

            # Get git repo for scheduler directory
            simulate_lsst.get_config_repo(
                ts_config_scheduler_commit="develop", clone_path="ts_config_scheduler"
            )
            return_status = simulate_lsst.make_lsst_scheduler_cli([scheduler_pickle])
            assert return_status == 0

            with open(scheduler_pickle, "rb") as pickle_io:
                scheduler = pickle.load(pickle_io)

            assert isinstance(scheduler, CoreScheduler)

    @unittest.skip("redundant")
    def test_make_model_observatory_cli(self) -> None:
        with temp_cwd():
            observatory_pickle = "observatory.p"
            return_status = simulate_lsst.make_model_observatory_cli([observatory_pickle])
            assert return_status == 0

            with open(observatory_pickle, "rb") as pickle_io:
                observatory = pickle.load(pickle_io)

            assert isinstance(observatory, ModelObservatory)

    def test_make_band_scheduler_cli(self) -> None:
        with temp_cwd():
            band_scheduler_pickle = "band_scheduler.p"
            return_status = simulate_lsst.make_band_scheduler_cli([band_scheduler_pickle])
            assert return_status == 0

            with open(band_scheduler_pickle, "rb") as pickle_io:
                band_scheduler = pickle.load(pickle_io)

            assert isinstance(band_scheduler, BandSwapScheduler)

    def test_run_lsst_sim_cli(self) -> None:
        with temp_cwd():
            scheduler_pickle = "scheduler.p"
            # Get git repo for scheduler directory
            simulate_lsst.get_config_repo(
                ts_config_scheduler_commit="develop", clone_path="ts_config_scheduler"
            )
            # Generate the DDF array here, so we can just pass simple
            # command line arguments next ..
            _ = subprocess.run(simulate_lsst.CONFIG_DDF_SCRIPT_PATH, capture_output=True)
            return_status = simulate_lsst.make_lsst_scheduler_cli([scheduler_pickle])
            assert return_status == 0
            with open(scheduler_pickle, "rb") as pickle_io:
                scheduler = pickle.load(pickle_io)
            assert isinstance(scheduler, CoreScheduler)
            nside = scheduler.nside

            observatory_pickle = "observatory.p"
            return_status = simulate_lsst.make_model_observatory_cli(
                [observatory_pickle, "--nside", f"{nside}"]
            )
            assert return_status == 0
            with open(observatory_pickle, "rb") as pickle_io:
                observatory = pickle.load(pickle_io)
            assert isinstance(observatory, ModelObservatory)

            init_opsim = ""
            # Use SURVEY_START_MJD so we get skybrightness files by default.
            t_start = Time(SURVEY_START_MJD, format="mjd")
            day_obs = str(rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(t_start)))
            sim_nights = "1"
            run_name = "test_opsim_output"
            simulate_lsst.run_lsst_sim_cli(
                [scheduler_pickle, observatory_pickle, init_opsim, day_obs, sim_nights, run_name]
            )

            obs = SchemaConverter().opsim2obs(f"{run_name}.db")
            assert len(obs) > 500
            assert Time(obs["mjd"] - 0.5, format="mjd").min().datetime.strftime("%Y%m%d") == day_obs


if __name__ == "__main__":
    unittest.main()
