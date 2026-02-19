import unittest

import rubin_nights.dayobs_utils as rn_dayobs
from astropy.time import TimeDelta

from lsst_survey_sim import lsst_support


class TestLsstSupport(unittest.TestCase):

    def test_set_sim_flags(self) -> None:
        expected_sim_flags = [
            "day_obs",
            "next_day_obs",
            "end_day_obs",
            "downtime_start_day_obs",
            "today_day_obs",
            "add_downtime",
            "real_downtime",
            "add_clouds",
        ]
        # We need to dynamically set "today"
        day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.today_day_obs())
        today = day_obs
        sim_nights = 1
        sim_flags = lsst_support.set_sim_flags(day_obs, sim_nights=sim_nights)
        for flag in expected_sim_flags:
            self.assertTrue(flag in sim_flags)
        # In this case - doing a simulation for today, one night only
        self.assertEqual(sim_flags["day_obs"], today)
        self.assertEqual(sim_flags["today_day_obs"], today)
        end_day_obs_time = rn_dayobs.day_obs_to_time(day_obs) + TimeDelta(sim_nights, format="jd")
        end_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(end_day_obs_time))
        self.assertEqual(sim_flags["end_day_obs"], end_day_obs)
        self.assertEqual(sim_flags["downtime_start_day_obs"], today)
        self.assertEqual(sim_flags["add_downtime"], False)
        self.assertEqual(sim_flags["real_downtime"], False)
        self.assertEqual(sim_flags["add_clouds"], False)
        # For a day in the past - should by default try to use real downtime
        day_obs = rn_dayobs.day_obs_to_time(rn_dayobs.today_day_obs()) - TimeDelta(2, format="jd")
        day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(day_obs))
        tomorrow = rn_dayobs.day_obs_to_time(rn_dayobs.today_day_obs()) + TimeDelta(1, format="jd")
        tomorrow = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(tomorrow))
        sim_nights = 1
        sim_flags = lsst_support.set_sim_flags(day_obs, sim_nights=sim_nights)
        self.assertEqual(sim_flags["day_obs"], day_obs)
        self.assertEqual(sim_flags["today_day_obs"], today)
        end_day_obs_time = rn_dayobs.day_obs_to_time(day_obs) + TimeDelta(sim_nights, format="jd")
        end_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(end_day_obs_time))
        self.assertEqual(sim_flags["end_day_obs"], end_day_obs)
        # Simulated downtime won't start today, even if today is not finished
        self.assertEqual(sim_flags["downtime_start_day_obs"], tomorrow)
        self.assertEqual(sim_flags["add_downtime"], True)
        self.assertEqual(sim_flags["real_downtime"], True)
        self.assertEqual(sim_flags["add_clouds"], False)
        # For a day in the past with sim running beyond today
        sim_nights = 5
        sim_flags = lsst_support.set_sim_flags(day_obs, sim_nights=sim_nights)
        self.assertEqual(sim_flags["day_obs"], day_obs)
        self.assertEqual(sim_flags["today_day_obs"], today)
        end_day_obs_time = rn_dayobs.day_obs_to_time(day_obs) + TimeDelta(sim_nights, format="jd")
        end_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(end_day_obs_time))
        self.assertEqual(sim_flags["end_day_obs"], end_day_obs)
        self.assertEqual(sim_flags["downtime_start_day_obs"], tomorrow)
        self.assertEqual(sim_flags["add_downtime"], True)
        self.assertEqual(sim_flags["real_downtime"], True)
        self.assertEqual(sim_flags["add_clouds"], False)
        # For many days now or in the future
        day_obs = rn_dayobs.day_obs_to_time(rn_dayobs.today_day_obs()) + TimeDelta(2, format="jd")
        day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(day_obs))
        sim_nights = 30
        sim_flags = lsst_support.set_sim_flags(day_obs, sim_nights=sim_nights)
        self.assertEqual(sim_flags["day_obs"], day_obs)
        self.assertEqual(sim_flags["today_day_obs"], today)
        end_day_obs_time = rn_dayobs.day_obs_to_time(day_obs) + TimeDelta(sim_nights, format="jd")
        end_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(end_day_obs_time))
        self.assertEqual(sim_flags["end_day_obs"], end_day_obs)
        self.assertEqual(sim_flags["downtime_start_day_obs"], day_obs)
        self.assertEqual(sim_flags["add_downtime"], True)
        self.assertEqual(sim_flags["add_clouds"], True)

    def test_survey_footprint(self) -> None:
        footprint_arrays = lsst_support.survey_footprint(nside=32)
        # simply check that the footprint, labels and wfd_footprint are present
        for key in ["footprint", "labels", "wfd_fp"]:
            self.assertTrue(key in footprint_arrays)

        footprint_bands = footprint_arrays["footprint"].dtype.names
        if footprint_bands is None:
            self.fail()
        for band in ["u", "g", "r", "i", "z", "y"]:
            self.assertTrue(band in footprint_bands)

    def test_survey_downtimes(self) -> None:
        # Check without downtime
        today = rn_dayobs.day_obs_str_to_int(rn_dayobs.today_day_obs())
        survey_info = lsst_support.survey_times(downtime_start_day_obs=today, add_downtime=False)
        self.assertEqual(survey_info["downtime_per_night"].sum(), 0)
        # Check with downtime
        survey_info = lsst_support.survey_times(downtime_start_day_obs=today, add_downtime=True)
        self.assertTrue(survey_info["downtime_per_night"].sum() > 0)


if __name__ == "__main__":
    unittest.main()
