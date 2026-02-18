import logging
import sqlite3
import warnings

import astropy
import numpy as np
import numpy.typing as npt
import pandas as pd
import rubin_nights.dayobs_utils as rn_dayobs
from astropy.time import Time, TimeDelta
from erfa import ErfaWarning
from rubin_nights import connections, observatory_status
from rubin_nights.influx_query import InfluxQueryClient
from rubin_scheduler.scheduler.model_observatory import ModelObservatory, tma_movement
from rubin_scheduler.scheduler.utils import (
    ObservationArray,
    SchemaConverter,
    SimTargetooServer,
    get_current_footprint,
    run_info_table,
)
from rubin_scheduler.site_models import (
    Almanac,
    ConstantSeeingData,
    ScheduledDowntimeData,
    SeeingModel,
    UnscheduledDowntimeMoreY1Data,
)
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, Site

"""Module supporting the configuration of simulation inputs."""

__all__ = [
    "set_sim_flags",
    "survey_footprint",
    "survey_times",
    "setup_observatory_summit",
    "setup_observatory_simulation",
    "SlewScatter",
    "save_opsim",
]

astropy.utils.iers.conf.iers_degraded_accuracy = "ignore"

LOGGER = logging.getLogger(__name__)


def set_sim_flags(day_obs: int, sim_nights: int) -> dict:
    """Set some likely flags for the observatory setup, based on
    the day_obs (of the simulation start) compared to today's day_obs.

    Parameters
    ----------
    day_obs
        The YYYYMMDD of the start of the simulation.
    sim_nights
        The number of nights for which to run the simulation.

    Returns
    -------
    sim_flags :  `dict`
        Dictionary of day_obs (today, next day, sim end day)
        and associated flags for setting up the observatory
        (add_downtime, real_downtime, and add_clouds) based on the
        most likely combinations that would be useful given day_obs
        and the current (today) day_obs.
    """
    # today_dayobs is the day of today -
    # if it is larger than day_obs, then we are running in the past.
    today_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.today_day_obs())
    tomorrow_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.tomorrow_day_obs())

    # Knowing the day after day_obs can be useful:
    next_day_obs_time = rn_dayobs.day_obs_to_time(day_obs) + TimeDelta(1, format="jd")
    next_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(next_day_obs_time))

    # End day of sim
    end_day_obs_time = rn_dayobs.day_obs_to_time(day_obs) + TimeDelta(sim_nights, format="jd")
    end_day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(end_day_obs_time))

    # Some parameters relating to downtime setup for model observatory
    if sim_nights <= 2:
        # One or two nights, probably no downtime or clouds.
        add_downtime = False
        real_downtime = False
        add_clouds = False
        downtime_start = day_obs
    else:
        # Multiple nights, probably want downtime and clouds.
        add_downtime = True
        real_downtime = True
        add_clouds = True
        downtime_start = day_obs
    # But -- beyond those, if we are running in the PAST,
    # we will want to use the real on-sky time for the FBS visits.
    # This means adding downtime, using the real-downtime,
    # but not adding clouds.
    # We will also need to be careful about what data we query for
    # and what we add to the FBS.
    if day_obs < today_day_obs:
        LOGGER.info("Checking the past, will restrict uptime to time acquiring science visits")
        add_downtime = True
        real_downtime = True
        add_clouds = False
        # simulated downtime should start (by default) when real
        # data likely to end (but not counting tonight)
        downtime_start = tomorrow_day_obs

    sim_flags = {
        "day_obs": day_obs,
        "next_day_obs": next_day_obs,
        "end_day_obs": end_day_obs,
        "downtime_start_day_obs": downtime_start,
        "today_day_obs": today_day_obs,
        "add_downtime": add_downtime,
        "real_downtime": real_downtime,
        "add_clouds": add_clouds,
    }
    return sim_flags


def survey_footprint(
    nside: int = DEFAULT_NSIDE,
) -> dict[str, npt.NDArray]:
    """Simple utility to fetch the footprint and provide quick label for WFD.

    Parameters
    -----------
    nside
        The HEALpix nside for the footprint.

    Returns
    -------
    footprint_arrays : `dict` [`str`, `np.ndarray`]
        A dictionary with various pieces of footprint information.
        `footprint` contains the total footprint weights per band.
        `wfd_fp` is a True/False array of pixels in the WFD regions.
    """
    fp_array, labels = get_current_footprint(nside=nside)
    wfd_labels = ["lowdust", "euclid_overlap", "virgo", "LMC_SMC", "bulgy"]
    wfd_fp = np.isin(labels, wfd_labels)
    rolling_labels = ["lowdust", "virgo"]
    rolling_fp = np.isin(labels, rolling_labels)
    footprint_arrays = {"footprint": fp_array, "labels": labels, "wfd_fp": wfd_fp, "rolling_fp": rolling_fp}
    return footprint_arrays


def survey_times(
    downtime_start_day_obs: int,
    new_downtime_ndays: float = 365.0,
    random_seed: int = 55,
    minutes_after_sunset12: float = 0,
    early_dome_closure: float = 0,
    add_downtime: bool = True,
    real_downtime: bool = False,
    visits: pd.DataFrame | None = None,
    survey_start_mjd: float = SURVEY_START_MJD,
) -> dict:
    """Set up basic LSST survey conditions.

    Most importantly, this includes accounting for downtime.

    Because the up/downtime could be expensive to calculate (since it
    involves masking the last hour before sunrise), specify
    `sim_length` to be the time for the desired simulation (and end
    of the calculated downtimes). This will start at downtime_dayobs_start.

    Parameters
    ----------
    downtime_start_day_obs
        Start simulating downtime in this module on downtime_start_day_obs,
        and add (new) downtime to downtime_start_day_obs + downtime_ndays.
        Note that the 'real' downtime (from on-sky availability) will
        be added only for the period before this date.
    new_downtime_ndays
        Generate downtime values from day_obs to day_obs + downtime_ndays.
        Use rubin_scheduler.site_models downtime models beyond this.
    random_seed
        Random value to seed downtimes with
    minutes_after_sunset12
        How long after -12 deg sunset to get on sky, in minutes.
        In theory, this should be 0.
        In practice, this tends to be about 10 or 20 minutes at the moment.
    early_dome_closure
        Close the dome (start downtime) `early_dome_closure` hours before
        (or after) -12-degree sunrise. (as of Dec 2025, dome closure is -12).
        In hours. Note that sim_runner's ModelObservatory will also stop
        at a pre-determined sun altitude.
    add_downtime
        If False, do not add unscheduled downtime - this still adds early
        dome closure and late-start from day_obs to downtime_ndays.
        If True, add downtime - from downtime_start_day_obs to downtime_ndays,
        generated here -(includes short downtimes and multiple per night,
        as well as early dome closures); beyond day_obs + downtime_ndays,
        the standard downtime generated by rubin_scheduler.site_models
        downtime modules is used.
    real_downtime
        Use the time the LSST FBS survey was on-sky to determine
        the downtime up to downtime_start_day_obs.
        This is useful to check simulations running from the start
        of survey (without using real visits) match
        the appropriate general characteristics of the real survey to date.
        It does also give an approximately realistic view of up/down time
        prior to the current date.
    visits
        Pass in the visits from the consdb.
        Only needed if real_downtime is True.
    survey_start_mjd
        The nominal start of the survey.
        Used for standard Scheduled and UnscheduledMoreY1 downtimes
        that extend after new_downtime_ndays.

    Returns
    -------
    survey_info : `dict`
        Returns a dictionary with keys containing information about the
        survey. Among others, this includes:
        `downtimes` - the downtimes to feed to the ModelObservatory.
    """
    warnings.filterwarnings("ignore", category=ErfaWarning)

    # We need 'survey_start' to anchor normal rubin_scheduler Downtime
    survey_start = Time(survey_start_mjd, format="mjd", scale="utc")
    survey_end = survey_start + TimeDelta(365 * 10.2, format="jd")
    # Time limits for simulating downtime in this function.
    downtime_start = rn_dayobs.day_obs_to_time(downtime_start_day_obs)
    # Simulated downtime period,
    downtime_length = TimeDelta(new_downtime_ndays, format="jd")
    downtime_end = downtime_start + downtime_length

    count_start = downtime_start
    if visits is not None:
        # But we might need sunset/rise times earlier too.
        # Note that visits should be consdb or converted consdb -
        # so obs_end_mjd should be present regardless of name of startMJD
        if "obs_start_mjd" in visits.columns:
            obs_start_mjd_key = "obs_start_mjd"
        else:
            obs_start_mjd_key = "observationStartMJD"
        if add_downtime and real_downtime:
            visit_day_obs_start = rn_dayobs.time_to_day_obs(
                Time(visits[obs_start_mjd_key].min(), format="mjd", scale="tai")
            )
            count_start = rn_dayobs.day_obs_to_time(visit_day_obs_start)

    dayobsmjd = np.arange(count_start.mjd, survey_end.mjd + 0.5, 1)

    # Find sunset and sunrise info during the period of downtime to model here
    lsst_site = Site("LSST")
    almanac = Almanac(mjd_start=survey_start.mjd)

    # Start with these configured only over the downtime simulated here
    alm_start = np.where(abs(almanac.sunsets["sunset"] - count_start.mjd) < 0.5)[0][0]
    alm_end = np.where(abs(almanac.sunsets["sunset"] - downtime_end.mjd) < 0.5)[0][0]
    evening_twi = almanac.sunsets[alm_start:alm_end]["sun_n12_setting"]
    morning_twi = almanac.sunsets[alm_start:alm_end]["sun_n12_rising"]
    actual_sunsets = almanac.sunsets[alm_start:alm_end]["sunset"]
    actual_sunrises = almanac.sunsets[alm_start:alm_end]["sunrise"]

    survey_info = {
        "almanac": almanac,
        "site": lsst_site,
        "survey_start": survey_start,
        "survey_end": survey_end,
        "downtime_start": downtime_start,
        "downtime_end": downtime_end,
        "late_start": minutes_after_sunset12,
        "early_end": early_dome_closure,
    }

    # Add time limits and downtime
    # early_dome_closure is the time ahead of 12-deg sunrise to close

    # Always add dome_closure during downtime_start to downtime_end
    down_starts = morning_twi - early_dome_closure / 24
    down_ends = actual_sunrises

    # And we might as well throw in being slow on sky
    d_starts = actual_sunsets
    d_ends = evening_twi + minutes_after_sunset12 / 60 / 24

    down_starts = np.concatenate([down_starts, d_starts])
    down_ends = np.concatenate([down_ends, d_ends])

    # Generate simulated downtime
    if add_downtime:

        # Placeholder if we want to put in known upcoming weather
        # problems for simulations (or scheduled maintenance)
        weather_starts: list[Time] = [
            Time("2026-06-19T12:00:00", scale="utc"),  # ~ June shutdown
        ]
        weather_ends: list[Time] = [
            Time("2026-07-04T12:00:00", scale="utc"),  # ~ June shutdown
        ]

        # Generate downtimes during downtime_start to downtime_end
        rng = np.random.default_rng(seed=random_seed)

        night_start = evening_twi
        night_end = morning_twi - early_dome_closure / 24.0

        # Add random periods of downtime within each night,
        # Very similar to the unscheduled downtime in
        # UnscheduledDowntimeMoreY1Data but more frequent + shorter
        random_downtime = 0
        nn = len(night_start)
        for count in range(3):
            threshold = 1.0 - (count / 4)
            prob_down = rng.random(nn)
            # Reduced downtime to better match current (20260101) distribution
            time_down = rng.gumbel(loc=0.3, scale=0.5, size=nn)  # hours
            time_down = np.where(time_down < 0, 0, time_down)
            # apply probability of having downtime or not -
            # But always at least 2 minutes for each cycle of 'count'
            time_down = np.where(prob_down <= threshold, time_down, 2 / 60 / 24)
            avail_in_night = (night_end - night_start) * 24
            time_down = np.where(time_down >= avail_in_night, avail_in_night, time_down)
            time_down = np.where(time_down <= 0, 3 / 60 / 24, time_down)
            d_starts = rng.uniform(low=night_start, high=night_end - time_down / 24)
            d_ends = d_starts + time_down / 24.0
            random_downtime += ((d_ends - d_starts) * 24).sum()
            # night_hours = avail_in_night.sum()
            # print(
            #     "cycle",
            #     count,
            #     random_downtime,
            #     night_hours,
            #     random_downtime / night_hours,
            # )
            # combine previous expected downtime and random downtime
            down_starts = np.concatenate([down_starts, d_starts])
            down_ends = np.concatenate([down_ends, d_ends])

        # Remove some additional whole nights for weather
        for ws, we in zip(weather_starts, weather_ends):
            # because later we want to count time PER NIGHT,
            # these are refactored to have start/end values on each night
            d_starts = np.arange(ws.mjd, we.mjd - 1, 1)
            d_ends = np.arange(ws.mjd + 1, we.mjd, 1)
            down_starts = np.concatenate([down_starts, d_starts])
            down_ends = np.concatenate([down_ends, d_ends])

        # Add downtime model for remainder of survey
        scheduled_downtime = ScheduledDowntimeData(
            start_time=survey_start,
        )

        # This may not be good for counting per night ..
        for ds, de in zip(scheduled_downtime.downtime["start"], scheduled_downtime.downtime["end"]):
            if ds > downtime_end and de < survey_end:
                dsm = np.floor(ds.mjd) + 0.5
                dem = np.floor(de.mjd) + 0.5
                d_starts = np.arange(dsm, dem - 1, 1)
                d_ends = np.arange(dsm + 1, dem, 1)
                down_starts = np.concatenate([down_starts, d_starts])
                down_ends = np.concatenate([down_ends, d_ends])

        # Need to check this matchup better. Setting the start_time to the
        # end of the simulated downtime placed the chunks too strongly.
        unscheduled_downtime = UnscheduledDowntimeMoreY1Data(start_time=survey_start, survey_length=3700)
        # This may not be good for counting per night ..
        for ds, de in zip(unscheduled_downtime.downtime["start"], unscheduled_downtime.downtime["end"]):
            if ds > downtime_end and de < survey_end:
                if de - ds >= TimeDelta(1, format="jd"):
                    dsm = np.floor(ds.mjd) + 0.5
                    dem = np.floor(de.mjd) + 0.5
                    d_starts = np.arange(dsm, dem - 1, 1)
                    d_ends = np.arange(dsm + 1, dem, 1)
                else:
                    d_starts = np.array([ds.mjd])
                    d_ends = np.array([de.mjd])
                down_starts = np.concatenate([down_starts, d_starts])
                down_ends = np.concatenate([down_ends, d_ends])

        # Now consolidate!
        # Sort and then remove overlaps
        sorted_order = np.argsort(down_starts)
        down_starts = down_starts[sorted_order]
        down_ends = down_ends[sorted_order]

        # Remove overlaps
        diff = down_starts[1:] - down_ends[0:-1]
        while np.min(diff) < 0:
            tts = down_starts[0:-1].copy()
            tte = down_ends[0:-1].copy()
            for i, (ds, de) in enumerate(zip(tts, tte)):
                if down_starts[i + 1] < de:
                    new_end = np.max([de, down_ends[i + 1]])
                    down_ends[i] = new_end
                    down_ends[i + 1] = new_end

            good = np.where(down_ends - np.roll(down_ends, 1) != 0)
            down_ends = down_ends[good]
            down_starts = down_starts[good]
            diff = down_starts[1:] - down_ends[0:-1]

    if add_downtime and real_downtime and visits is None:
        raise ValueError("No visits found and but real_downtime is True.")

    # Replace downtime before downtime_start, if real_downtime
    if add_downtime and real_downtime and visits is not None:

        # Identify gaps/downtime starts
        edges = np.where(np.diff(visits[obs_start_mjd_key].values) > 230 / 60 / 60 / 24)[0]
        # Every time there is a gap, a dropout "starts"
        dropout_starts = visits.iloc[edges]["obs_end_mjd"].values
        # And at the end of each gap, a dropout "ends"
        dropout_ends = visits.iloc[edges + 1][obs_start_mjd_key].values - 150 / 60 / 60 / 24

        if visits.obs_end_mjd.max() < downtime_start.mjd:
            # If the last visit ended before the downtime simulation should
            # start, it should trigger an additional dropout.
            dropout_starts = np.concatenate([dropout_starts, np.array([visits.obs_end_mjd.max()])])
            dropout_ends = np.concatenate([dropout_ends, np.array([downtime_start.mjd - 0.001])])

        d_starts = []
        d_ends = []
        for ds, de in zip(dropout_starts, dropout_ends):
            idx_s = np.where(ds >= dayobsmjd)[0][-1]
            idx_e = np.where(de >= dayobsmjd)[0][-1]
            if idx_s == idx_e:
                d_starts += [ds]
                d_ends += [de]
            else:
                idx_s = idx_s + 1
                if idx_e == idx_s:
                    d_starts += [ds]
                    d_starts += [dayobsmjd[idx_s]]
                    d_ends += [dayobsmjd[idx_e]]
                    d_ends += [de]
                else:
                    idx_e = idx_e + 1
                    d_starts += [ds]
                    d_starts += list(dayobsmjd[idx_s:idx_e])
                    d_ends += list(dayobsmjd[idx_s:idx_e])
                    d_ends += [de]

        # Use real downtime where we have that information, but continue
        # with sim downtime
        keep_starts = down_starts[np.where(down_starts >= downtime_start.mjd)]
        keep_ends = down_ends[np.where(down_starts >= downtime_start.mjd)]

        down_starts = np.concatenate([d_starts, keep_starts])
        down_ends = np.concatenate([d_ends, keep_ends])

    # Trim all of these to sunrise/sunset
    # use sunsets/sunrises over the whole survey
    alm_start = np.where(abs(almanac.sunsets["sunset"] - count_start.mjd) < 0.5)[0][0]
    alm_end = np.where(abs(almanac.sunsets["sunset"] - survey_end.mjd) < 0.5)[0][0]
    evening_twi = almanac.sunsets[alm_start:alm_end]["sun_n12_setting"]
    morning_twi = almanac.sunsets[alm_start:alm_end]["sun_n12_rising"]
    actual_sunsets = almanac.sunsets[alm_start:alm_end]["sunset"]
    actual_sunrises = almanac.sunsets[alm_start:alm_end]["sunrise"]

    downtime_starts = []
    downtime_ends = []
    eps = 0.0001
    for ds, de in zip(down_starts, down_ends):
        idx = np.where(ds >= dayobsmjd)[0][-1]
        if ds < actual_sunsets[idx]:
            downtime_starts.append(actual_sunsets[idx])
        else:
            downtime_starts.append(ds)
        idx = np.where(de > dayobsmjd)[0][-1]
        # If we ended up on a day boundary, have to back up one night
        if np.abs(de - dayobsmjd[idx]) < eps:
            idx = idx - 1
        if de > actual_sunrises[idx]:
            downtime_ends.append(actual_sunrises[idx])
        else:
            downtime_ends.append(de)

    # Turn into an array of downtimes for sim_runner
    # down_starts/ down_ends should be mjd times for ModelObservatory
    downtimes = np.array(
        list(zip(downtime_starts, downtime_ends)),
        dtype=list(zip(["start", "end"], [float, float])),
    )
    downtimes.sort(order="start")

    # Eliminate overlaps (just in case)
    diff = downtimes["start"][1:] - downtimes["end"][0:-1]
    while np.min(diff) < 0:
        logging.info("found overlap in downtimes")
        for i, dt in enumerate(downtimes[0:-1]):
            if downtimes["start"][i + 1] < dt["end"]:
                new_end = np.max([dt["end"], downtimes["end"][i + 1]])
                downtimes[i]["end"] = new_end
                downtimes[i + 1]["end"] = new_end

        good = np.where(downtimes["end"] - np.roll(downtimes["end"], 1) != 0)
        downtimes = downtimes[good]
        diff = downtimes["start"][1:] - downtimes["end"][0:-1]

    # Count up downtime within each night
    downtime_per_night = np.zeros(len(evening_twi))
    for start, end in zip(downtimes["start"], downtimes["end"]):
        idx = np.where((start >= dayobsmjd) & (end <= dayobsmjd + 1))
        if len(idx[0]) == 0:
            logging.error(
                "no index identified"
                f" {start} {end} {Time(start, format='mjd').iso} {Time(end, format='mjd').iso}",
            )
            continue
        if len(idx[0]) > 1:
            logging.error(
                "more than one index identified"
                f" {start} {end} {Time(start, format='mjd').iso} {Time(end, format='mjd').iso}",
            )
            logging.error((idx, dayobsmjd[idx], evening_twi[idx], morning_twi[idx]))
        if start < evening_twi[idx]:
            dstart = evening_twi[idx]
        else:
            dstart = start
        if end > morning_twi[idx]:
            dend = morning_twi[idx]
        else:
            dend = end
        downtime_per_night[idx] += (dend - dstart) * 24

    survey_info["downtimes"] = downtimes
    survey_info["dayobsmjd"] = dayobsmjd
    survey_info["sunrises12"] = morning_twi
    survey_info["sunsets12"] = evening_twi
    hours_in_night = (morning_twi - evening_twi) * 24
    survey_info["hours_in_night"] = hours_in_night
    survey_info["downtime_per_night"] = downtime_per_night
    survey_info["avail_per_night"] = hours_in_night - downtime_per_night
    survey_info["system_availability"] = np.nanmean(
        survey_info["avail_per_night"] / survey_info["hours_in_night"]
    )
    LOGGER.info(f"Max length of night {hours_in_night.max()} min length of night {hours_in_night.min()}")
    LOGGER.info(
        f"Total nighttime {hours_in_night.sum()}, "
        f"total downtime {downtime_per_night.sum()}, "
        f"available time {hours_in_night.sum() - downtime_per_night.sum()}"
    )
    LOGGER.info(f"Average availability {survey_info['system_availability']}")

    return survey_info


def setup_observatory_summit(
    survey_info: dict,
    seeing: float | str | None = None,
    add_clouds: bool = False,
    too_server: SimTargetooServer | None = None,
    time_setup: Time | None = None,
    expected_wait_settle: float | None = None,
    close_loop_filter_time: float | None = None,
) -> ModelObservatory:
    """Configure a model observatory matching the velocity/acceleration/jerk
    parameters at `time_setup`.

    Parameters
    ----------
    survey_info
        The survey_info dictionary returned by `survey_times`.
        Note that the survey_info carries the downtime information,
        as well as the survey_start information.
    seeing
        If specified as a float, then the constant seeing model will be used.
        If specified as string, this will be used as the seeing database path.
        If None, the standard model seeing database will be used.
    add_clouds
        If True, use our standard cloud downtime model.
        If False, use the 'ideal' cloud model resulting in no downtime.
    too_server
        A `SimTargetooServer` wrapping a list of TargetoO
        (target of opportunity) events.
    time_setup
        The Time at which to consider the observatory TMA configuration.
        If None, uses 20% velocity/acceleration/jerk settings.
    expected_wait_settle
        The current wait and/or settle time beyond TMA movement expected
        for the visit gap.
        Default `None` will use expected performance at `time_setup`.
    close_loop_filter_time
        The extra time (in seconds) to add to a filter change to account
        for close-loop iterations.
        Default `None` will use expected performance at `time_setup`.

    Returns
    -------
    model_observatory
        ModelObservatory complete with seeing model, cloud (weather downtime)
        model, and also downtimes (due to engineering).
    """
    seeing_data = None
    seeing_db = None
    if seeing is not None:
        if isinstance(seeing, (float, int)):
            seeing_data = ConstantSeeingData()
            seeing_data.fwhm_500 = seeing
        else:
            seeing_db = seeing
    # Potentially a bigger system contribution
    seeing_model = SeeingModel()

    if add_clouds:
        cloud_data = None
    else:
        cloud_data = "ideal"

    # TODO
    # create lookup table for expected_wait_settle over time
    # create lookup table for close_loop_filter_time over time
    if expected_wait_settle is None:
        expected_wait_settle = 3.0
    if close_loop_filter_time is None:
        close_loop_filter_time = 0  # 210.0

    observatory = ModelObservatory(
        nside=survey_info["nside"],
        mjd=survey_info["survey_start"].mjd,
        mjd_start=survey_info["survey_start"].mjd,
        cloud_data=cloud_data,
        seeing_data=seeing_data,
        seeing_db=seeing_db,
        wind_data=None,
        downtimes=survey_info["downtimes"],
        sim_to_o=too_server,
    )
    observatory.seeing_model = seeing_model

    if time_setup is not None:
        try:
            efd_client = InfluxQueryClient("usdf")
            tma = observatory_status.get_tma_limits(
                time_setup, time_setup + TimeDelta(1 / 24, format="jd"), efd_client
            )
            tma_performance = dict(tma.iloc[-1])
            tma_performance["settle_time"] = expected_wait_settle
            observatory.setup_telescope(**tma_performance)
            LOGGER.info(
                f"Setting up summit observatory with altitude_maxspeed {tma_performance['altitude_maxspeed']}"
            )
        except KeyError:
            LOGGER.info("USDF EFD not accessible. Using default summit-20.")
            time_setup = None
    # "summit-20 TMA" - but this is a label from the summit, not 20% in all
    if time_setup is None:
        observatory.setup_telescope(
            azimuth_maxspeed=2.0,
            azimuth_accel=2.0,
            azimuth_jerk=8.0,
            altitude_maxspeed=2.0,
            altitude_accel=2.0,
            altitude_jerk=8.0,
            settle_time=expected_wait_settle,
        )
        LOGGER.info("Setting up summit observatory as summit-20")

    # Set up camera with band changetime
    observatory.setup_camera(band_changetime=120 + close_loop_filter_time, readtime=3.07)

    return observatory


def setup_observatory_simulation(
    survey_info: dict,
    seeing: float | str | None = None,
    add_clouds: bool = False,
    too_server: SimTargetooServer | None = None,
) -> ModelObservatory:
    """Configure a `summit-10` model observatory.
    This approximates average summit performance at present.

    Parameters
    ----------
    survey_info
        The survey_info dictionary returned by `survey_times`.
        Note that the survey_info carries the downtime information,
        as well as the survey_start information.
    seeing
        If specified as a float, then the constant seeing model will be used.
        If specified as string, this will be used as the seeing database path.
        If None, the standard model seeing database will be used.
    add_clouds
        If True, use our standard cloud downtime model.
        If False, use the 'ideal' cloud model resulting in no downtime.
    too_server
        A `SimTargetooServer` wrapping a list of TargetoO
        (target of opportunity) events.

    Returns
    -------
    model_observatory
        ModelObservatory complete with seeing model, cloud (weather downtime)
        model, and also downtimes (due to engineering).
    """
    seeing_data = None
    seeing_db = None
    if seeing is not None:
        if isinstance(seeing, (float, int)):
            seeing_data = ConstantSeeingData()
            seeing_data.fwhm_500 = seeing
        else:
            seeing_db = seeing
    # Potentially a bigger system contribution
    seeing_model = SeeingModel()

    if add_clouds:
        cloud_data = None
    else:
        cloud_data = "ideal"

    observatory = ModelObservatory(
        nside=survey_info["nside"],
        mjd=survey_info["survey_start"].mjd,
        mjd_start=survey_info["survey_start"].mjd,
        cloud_data=cloud_data,
        seeing_data=seeing_data,
        seeing_db=seeing_db,
        wind_data=None,
        downtimes=survey_info["downtimes"],
        sim_to_o=too_server,
    )
    observatory.seeing_model = seeing_model
    tma_performance = tma_movement(40)
    observatory.setup_telescope(**tma_performance)
    observatory.setup_camera(band_changetime=140, readtime=3.07)
    return observatory


class SlewScatter:
    """Callable to return some scatter for the slew.

    Attempts to match the general scale of the scatter observed
    from the summit as of late Jan 2026.

    Parameters
    ----------
    seed : `int`
        Random number seed.
    slew_scale : `float`
        The scale for the scatter in the slew offest (seconds).
    """

    def __init__(self, seed: int = 42, slew_scale: float = 1.75) -> None:
        self.rng = np.random.default_rng(seed)
        self.slew_scale = slew_scale

    def __call__(self, visittime: float, slewtime: float) -> float:
        """Return a randomized offset for the visit overhead.

        Parameters
        ----------
        visittime : `float`
            The visit time (seconds).
        slewtime : `float`
            The slew time (seconds).

        Returns
        -------
        offset: `float`
            Random offset (seconds).
        """

        slew_scatter: float = np.abs(self.rng.normal(0, self.slew_scale))

        return slew_scatter


def save_opsim(
    observatory: ModelObservatory,
    observations: ObservationArray,
    initial_opsim: pd.DataFrame | None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Combine the initial (opsim formatted) visits with the observation array
    returned from the scheduler simulation.  Optionally saves the result
    to a standard opsim sqlite database.

    Parameters
    ----------
    observatory
        Model Observatory from the simulation.
        Used to save the metadata about the model observatory and site models
        to the sqlite file.
    observations
        The ObservationsArray created by the scheduler in the simulation.
    initial_opsim
        The initial opsim visits fed into the scheduler to start the
        simulation.
        Often (especially for SV) these are likely to be visits converted
        from the ConsDB. Note that extra columns are fine! We can keep
        extra consdb information available if desired (this can be useful
        for double-checking).
    filename
        If provided, this is the filename into which to save the
        (complete) opsim results. Recommend naming this something like
        `sv_{day_obs}.db` where dayobs is the integer dayobs on which the
        new sv simulation was

    Returns
    -------
    visits_df
        A DataFrame of both initial and simulated visits, in opsim format.
    """
    if len(observations) > 0:
        sim_visits = SchemaConverter().obs2opsim(observations)
        if initial_opsim is not None:
            visits_df = pd.concat([initial_opsim, sim_visits])
        else:
            visits_df = sim_visits
    else:
        visits_df = initial_opsim
    if filename is not None:
        con = sqlite3.connect(filename)
        visits_df.to_sql("observations", con, index=False, if_exists="replace")
        info = run_info_table(observatory)
        df_info = pd.DataFrame(info)
        df_info.to_sql("info", con, if_exists="replace")
        con.close()
    return visits_df
