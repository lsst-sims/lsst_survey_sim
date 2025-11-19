import argparse
import logging
import os
import pickle
import sqlite3
import subprocess
import warnings
from typing import Any

import astropy.units as u
import git
import healpy as hp
import numpy as np
import pandas as pd
import rubin_nights.dayobs_utils as rn_dayobs
import rubin_nights.rubin_sim_addons as rn_sim
from astroplan import Observer
from astropy.time import Time, TimeDelta
from rubin_nights import augment_visits, connections
from rubin_nights.influx_query import InfluxQueryClient
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler, DateSwapBandScheduler, SimpleBandSched
from rubin_scheduler.scheduler.surveys import ToOScriptedSurvey
from rubin_scheduler.scheduler.utils import ObservationArray, SchemaConverter, SimTargetooServer, TargetoO
from rubin_scheduler.utils import DEFAULT_NSIDE, Site

try:
    from rubin_sim.sim_archive import make_sim_data_dir
except ImportError:
    pass
from rubin_sim.sim_archive.make_snapshot import get_scheduler_from_config
from rubin_sim.sim_archive.prenight import AnomalousOverheadFunc

from . import lsst_support

CONFIG_SCRIPT_PATH = "ts_config_scheduler/Scheduler/feature_scheduler/maintel/fbs_config_lsst_survey.py"
CONFIG_DDF_SCRIPT_PATH = "ts_config_scheduler/Scheduler/ddf_gen/lsst_ddf_gen_block_407.py"

LOGGER = logging.getLogger(__name__)

__all__ = [
    "get_configuration",
    "fetch_previous_visits",
    "fetch_too_events",
    "setup_scheduler",
    "setup_band_scheduler",
    "setup_observatory",
    "run_sim",
    "simple_sim",
    "fetch_lsst_visits_cli",
    "make_lsst_scheduler_cli",
    "make_model_observatory_cli",
    "make_band_scheduler_cli",
    "run_lsst_sim_cli",
]


def get_configuration(ts_config_scheduler_commit: str, clone_path: str = "ts_config_scheduler") -> None:
    """Git checkout ts_config_scheduler and set it to the desired commit.

    `ts_config_scheduler_commit` is fetchable from
    `lsst.sal.Scheduler.logevent_configurationApplied`
    or use tip of the run branch, fetchable from `lsst.obsenv.run_branch`.

    Parameters
    ----------
    ts_config_scheduler_commit : `str`
        Git branch .. could be a hash?
    clone_path : `str` or None
        The location to clone the repo.
        Join with the remainder of the path to the configuration scripts.
    """
    repo_url = "https://github.com/lsst-ts/ts_config_scheduler"

    # Clone new or use existing repo.
    if not os.path.isdir(clone_path):
        repo = git.Repo.clone_from(repo_url, clone_path)
        LOGGER.info(f"ts_config_scheduler repository cloned to: {clone_path}.")
    else:
        repo = git.Repo(clone_path)
        LOGGER.info(f"Directory {clone_path} already exists. Let's assume it's the repo.")
        repo.git.checkout("develop")
        repo.git.pull()
    # Check out the commit
    repo.git.checkout(ts_config_scheduler_commit)
    LOGGER.info(f"Checked out commit {ts_config_scheduler_commit} in {clone_path}.")


def fetch_previous_visits(
    day_obs: int, tokenfile: str | None = None, site: str = "usdf", convert_to_opsim: bool = True
) -> pd.DataFrame:
    """Fetch relevant visits from the Consdb.

    Parameters
    ----------
    day_obs
        The day_obs (integer) of the day on which to start the simulation.
        Will fetch all science visits *up to* (<) this day_obs.
    tokenfile
        Path to the RSP tokenfile.
        See also `rubin_nights.connections.get_access_token`.
        Default None will use `ACCESS_TOKEN` environment variable.
    site
        The site (`usdf`, `usdf-dev`, `summit` ..) location at
        which to query services. Must match tokenfile origin.
    convert_to_opsim
        If True, convert to opsim format with rubin_nights.consdb_to_opsim.
        If False, keep in consdb format.

    Returns
    -------
    visits : `pd.DataFrame`
        DataFrame containing (optionally) opsim-formatted visit information
        from the consdb for the LSST visits up to day_obs.
        Is None if no visits available.
    """
    # Get the survey visits from the ConsDB.
    endpoints = connections.get_clients(tokenfile=tokenfile, site=site)
    consdb = endpoints["consdb"]

    instrument = "lsstcam"
    query = (
        f"select v.*, q.* from cdb_{instrument}.visit1 as v "
        f"left join cdb_{instrument}.visit1_quicklook as q "
        f"on v.visit_id = q.visit_id "
        f"where v.day_obs < {day_obs} "
        f"and v.science_program = 'BLOCK-407' or v.science_program = 'BLOCK-408'"
    )
    visits = consdb.query(query)
    if len(visits) > 0:
        # Augment visits adds some additional columns.
        visits = augment_visits.augment_visits(visits, instrument)
        # Remove known bad visits.
        bad_visit_ids = augment_visits.fetch_excluded_visits("lsstcam")
        visits = augment_visits.exclude_visits(visits, bad_visit_ids)
        # Replace Nans in measured seeing if any remain.
        fill_value = 3.0
        fwhm = np.where(np.isnan(visits.fwhm_eff.values), fill_value, visits.fwhm_eff.values)
        visits.loc[:, "fwhm_eff"] = fwhm
        if convert_to_opsim:
            # Convert consdb visits to opsim visits
            visits = rn_sim.consdb_to_opsim(visits)
            visits.loc[:, "note"] = visits.loc[:, "scheduler_note"].copy()
        LOGGER.info(f"Fetched {len(visits)} good visits.")
    else:
        visits = None
    return visits


def fetch_too_events(t_start: Time, t_end: Time, site: str = "base") -> list[TargetoO] | None:
    """Fetch ToO triggers from EFD and convert to TargetoO events.

    For success, this requires access to `base`.
    (i.e. summit VPN or running on base-lsp.lsst.codes).

    Parameters
    ----------
    t_start, t_end
        Time period to check the EFD for too_alerts.
    site
        The name of the EFD to check for alerts.
        Default is `base`, but `summit` may be valid.

    Returns
    -------
    toos : `list` [ `TargetoO` ] or None
        List of rubin_scheduler.scheduler.utils.TargetoO objects.
        Add to conditions (or ModelObservatory).
        Returns None if no ToOs were within t_start to t_end or
        if there was an error connecting or processing the ToOs.
    """
    too_client = InfluxQueryClient(site, db_name="lsst.scimma")

    # Do we have a usable connection to this EFD?
    topics = too_client.get_topics()
    if len(topics) == 0:
        LOGGER.error(f"No connection to {site} efd for alerts")
        return None

    alerts = too_client.select_time_series("lsst.scimma.too_alert", "*", t_start, t_end)
    if len(alerts) > 0:
        # Find columns with reward_maskXXX and sort them in order
        reward_map_idxs = sorted(
            [int(c.split("reward_map")[-1]) for c in alerts.columns if "reward_map" in c and "nside" not in c]
        )
        reward_map_cols = [f"reward_map{c}" for c in reward_map_idxs]
        ra, dec = np.radians(hp.pix2ang(32, reward_map_idxs, nest=False, lonlat=True))

        # (note that the reward_map cols are in NEST order and FBS expects RING
        def _alert_to_too(a: pd.Series) -> TargetoO:
            reward_map = np.zeros(hp.nside2npix(a.reward_map_nside), bool)
            reward_map[reward_map_idxs] = a[reward_map_cols].values
            reward_map = hp.reorder(reward_map, n2r=True)
            ra_rad_center = ra[reward_map].mean()
            dec_rad_center = dec[reward_map].mean()
            try:
                mjd_time = Time(a.event_trigger_timestamp, format="isot", scale="utc").tai.mjd
            except ValueError:
                LOGGER.error(f"Could not convert timestamp {a.event_trigger_timestamp}")
                return None
            too = TargetoO(
                # tooid=a.source, # is what I'd like to do
                tooid=a.counter,
                footprint=reward_map,
                ra_rad_center=ra_rad_center,
                dec_rad_center=dec_rad_center,
                mjd_start=mjd_time,
                # We will need to update duration in ts_scheduler too
                duration=10.0,
                too_type=a.alert_type,
                posterior_distance=None,
            )
            return too

        alerts["counter"] = np.arange(1, len(alerts) + 1, 1)
        toos = alerts.apply(_alert_to_too, axis=1)
        toos = [too for too in toos if too is not None]
    else:
        LOGGER.info(f"No alerts found at {site} efd")
        toos = None
    return toos


def setup_scheduler(
    config_script_path: str,
    config_ddf_script_path: str | None = None,
    day_obs: int | None = None,
    too_server: SimTargetooServer | None = None,
    band_scheduler: DateSwapBandScheduler | None = None,
    initial_opsim: pd.DataFrame | None = None,
    opsim_filename: str | None = None,
) -> tuple[CoreScheduler, pd.DataFrame, int]:
    """Set up the survey scheduler.
     Read previous visits into scheduler for startup.

    Parameters
    ----------
    config_script_path
        The path to the scheduler configuration file.
    config_ddf_script_path
        The path to the associated DDF configuration file.
        If provided, this script will be run in order to generate the
        pre-generated DDF observations. If the pre-gen array already
        exists, the current lsst_ddf_gen scripts exit quickly.
    day_obs
        The day_obs (integer) of the day on which to start the simulation.
        Will fetch all visits *up to* this day_obs.
        If initial_opsim is passed, this can be ignored.
        If None, and no initial_opsim and no opsim_filename, will use "today".
    too_server
        A `SimTargetooServer` wrapping a list of TargetoO
        (target of opportunity) events.
        This is needed for setting up obs_wanted for the ToO surveys.
        If None, scheduler will be activated without any ToO knowledge.
    band_scheduler
        The BandScheduler to determine current mounted filters.
        This is needed for setting up obs_wanted for the ToO surveys.
    initial_opsim
        If initial_opsim is not None, use this dataframe instead of
        reading from disk. These should be *opsim* formatted visits.
        If not None, this will ignore all other kwargs.
    opsim_filename
        Get previous visits from a file, instead of directly from
        the ConsDB. If set, then consdb will not be queried.

    Returns
    -------
    scheduler : `CoreScheduler`
    initial_opsim : `pd.DataFrame`
    nside : `int`
    """
    # Run the ddf config
    if config_ddf_script_path is not None:
        # Run the DDF configuration
        result = subprocess.run(config_ddf_script_path, capture_output=True)
        LOGGER.info(result.stdout)
        if len(result.stderr) > 0:
            LOGGER.error(result.stderr)

    # Set up the scheduler from the config file from ts_config_ocs.
    nside, scheduler = get_scheduler_from_config(config_script_path)

    if day_obs is None:
        day_obs = rn_dayobs.today_day_obs()
    day_obs_time = rn_dayobs.day_obs_to_time(day_obs)

    # Set up ToOScriptedSurveys to be ready for ToOs.
    if too_server is not None and band_scheduler is not None:
        too_objs = too_server(day_obs_time.mjd)
        # This next bit is a total hack because mjd0 in ToOScriptedSurvey
        # should probably be the time of the ToO mjd-start, NOT the
        # current conditions.mjd ...
        if too_objs is not None:
            LOGGER.info(f"Adding {len(too_objs)} ToO to FBS setup.")
            for too in too_objs:
                # Do this one at a time right now because of too.mjd
                conditions = Conditions(nside=nside, mjd=too.mjd_start)
                conditions.mounted_bands = band_scheduler(conditions)
                conditions.current_band = "r"
                conditions.targets_of_opportunity = [too]
                # Update conditions in surveys so they set up obs_wanted.
                for surveys in scheduler.survey_lists:
                    for survey in surveys:
                        survey.update_conditions(conditions)
        else:
            LOGGER.info("Received too_server but no active ToOs.")

    # Add previous observations.
    if initial_opsim is None:
        if opsim_filename is not None:
            # Read from the datafile `filename`.
            con = sqlite3.connect(opsim_filename)
            initial_opsim = pd.read_sql("select * from observations;", con)
        else:
            LOGGER.info("Starting without any initial_opsim visits.")

    # Convert opsim visits to ObservationArray and feed the scheduler.
    if initial_opsim is not None and len(initial_opsim) > 0:
        sch_obs = SchemaConverter().opsimdf2obs(initial_opsim)
        scheduler.add_observations_array(sch_obs)
    return scheduler, initial_opsim, nside


def setup_band_scheduler() -> DateSwapBandScheduler:
    # Set up the filter scheduler.
    # Might be reasonable to add kwargs for upcoming filter swaps,
    # but it's also pretty easy to modify this file to match summit schedule.
    # https://rubinobs.atlassian.net/wiki/spaces/CAM
    # #  /pages/702939386/Filter+Swap+planning (confluence).
    # See also slack, #cam-filter-exchange
    upcoming_schedule = {
        "2025-10-22": ["g", "r", "i", "z"],
        "2025-10-25": ["u", "g", "r", "i", "z"],
        "2025-10-28": ["g", "r", "i", "z", "y"],
        "2025-11-11": ["u", "g", "r", "i", "z"],
        "2025-11-27": ["g", "r", "i", "z", "y"],
        "2025-12-10": ["u", "g", "r", "i", "z"],
        "2025-12-24": ["g", "r", "i", "z", "y"],
        "2026-01-12": ["u", "g", "r", "i", "z"],
        "2026-01-26": ["g", "r", "i", "z", "y"],
        "2026-02-10": ["u", "g", "r", "i", "z"],
        "2026-02-24": ["g", "r", "i", "z", "y"],
        "2026-03-11": ["u", "g", "r", "i", "z"],
        "2026-03-25": ["g", "r", "i", "z", "y"],
    }
    end_date = Time("2026-03-30T12:00:00")
    band_scheduler = DateSwapBandScheduler(
        swap_schedule=upcoming_schedule,
        end_date=end_date,
        backup_band_scheduler=SimpleBandSched(illum_limit=40),
    )
    return band_scheduler


def setup_observatory(
    day_obs: int,
    nside: int = DEFAULT_NSIDE,
    add_downtime: bool = False,
    add_clouds: bool = False,
    seeing: float | None = None,
    real_downtime: bool = False,
    initial_opsim: pd.DataFrame | None = None,
    too_server: SimTargetooServer | None = None,
) -> tuple[ModelObservatory, dict]:
    """Set up the model observatory.

    Parameters
    ----------
    day_obs
        The day_obs to start generating (extra) downtime for the observatory.
    nside
        The HEALpix nside value for the model observatory and scheduler.
    add_downtime
        If False, will not add any random downtime to the model observatory.
        If True, then scheduled and unscheduled downtime will be added.
        For prenight simulations, this should probably be True.
        For full simulations, this should be False.
    add_clouds
        If True, will add cloud downtime to the model observatory.
        If False, will not add cloud downtime and use 'ideal' (no) clouds.
        For prenight simulations, this should be False.
        For full simulations, this should probably be True.
    seeing
        Set the seeing to a single value (float) or use seeing distribution
        (if None).
        For full simulations, this should be None.
        For prenight simulations - it depends. If we have an estimate
        of the seeing expected for the night, it may be useful to set a value.
    opsim_filename
        Get previous visits from a file, instead of directly from
        the ConsDB. If set, then consdb will not be queried.
    tokenfile
        Path to the RSP tokenfile.
        See also `rubin_nights.connections.get_access_token`.
        Default None will use `ACCESS_TOKEN` environment variable.
    site
        The site (`usdf`, `usdf-dev`, `summit` ..) location at
        which to query services. Must match tokenfile origin.
    real_downtime
        A boolean flag to determine whether to rewrite the downtime
        within the range of initial_opsim into the actual uptime for visits
        or not. If True, initial_opsim must not be None.
    initial_opsim
        If initial_opsim is not None, use these visits instead of fetching or
        reading from disk. These should be *opsim* formatted visits.
    too_server
        A `SimTargetooServer` wrapping a list of TargetoO
        (target of opportunity) events.

    Returns
    -------
    observatory : `ModelObservatory`
    survey_info : `dict`
    """
    # Find the survey information - survey start, downtime simulation ..
    if real_downtime:
        if initial_opsim is None:
            raise ValueError("If real_downtime is True, initial_opsim must be provided.")

    survey_info = lsst_support.survey_times(
        day_obs=day_obs,
        add_downtime=add_downtime,
        real_downtime=real_downtime,
        visits=initial_opsim,
    )
    survey_info["nside"] = nside

    # This isn't strictly necessary for survey_info but adds useful
    # potential footprint information for plotting purposes
    survey_info.update(lsst_support.survey_footprint(nside=nside))

    # Now that we have downtime, set up model observatory.
    observatory = lsst_support.setup_observatory_summit(
        survey_info, seeing=seeing, add_clouds=add_clouds, too_server=too_server
    )
    return observatory, survey_info


def run_sim(
    scheduler: CoreScheduler,
    band_scheduler: DateSwapBandScheduler,
    observatory: ModelObservatory,
    survey_info: dict,
    day_obs: int,
    sim_nights: int | None = None,
    anomalous_overhead_func: Any | None = None,
    keep_rewards: bool = False,
    delay: float = 0,
) -> tuple[np.recarray, CoreScheduler, ModelObservatory, pd.DataFrame, pd.DataFrame, dict]:
    """Run the (already set up) scheduler and observatory for
    the appropriate length of time.

    Parameters
    ----------
    scheduler
        The CoreScheduler, ready to run the simulation with previous visits,
        if applicable.
    band_scheduler
        The Band swap scheduler.
    observatory
        The ModelObservatory, configured to run the simulation.
    survey_info
        Dictionary containing survey_start and (potentially) survey_end
        astropy Time dates. Returned from `sv_support.survey_times`.
    day_obs
        The integer day_obs on which to start the simulation.
    sim_nights
        The number of nights to run the simulation. If None, then run
        to the end of survey specified in survey_info.
    anomalous_overhead_func
        A function or callable object that takes the visit time and slew time
        (in seconds) as argument, and returns and additional offset (also
        in seconds) to be applied as additional overhead between exposures.
        Defaults to None.
    keep_rewards
        If True, will compute and return rewards.
    delay
        Number of minutes by which simulated observing should be delayed.

    Returns
    -------
    sim_observations, scheduler, observatory, rewards, obs_rewards, survey_info

    """
    # Start at dayobs sunset minus a tiny bit of time
    # (ensure band scheduler changes if needed and that we start on time)
    day_obs_str = rn_dayobs.day_obs_int_to_str(day_obs)
    day_obs_time = Time(f"{day_obs_str}T12:00:00", format="isot", scale="utc")

    observer = Observer(Site("LSST").to_earth_location())
    sunset = Time(
        observer.sun_set_time(day_obs_time, which="next", horizon=-6 * u.deg),
        format="jd",
    )

    # If a delay is requested, set the delay relative to 12 degree twilight.
    # This might not always be correct. Ideally, we might need to start with a
    # mini-simulation to test where the first visit comes out without a delay,
    # then follow it with a second sim starting delayed relative to that.
    # Note that downtimes may have a built-in delay to match observed starts.
    if delay > 0:
        nominal_start = Time(
            observer.sun_set_time(day_obs_time, which="next", horizon=-12 * u.deg),
            format="jd",
        ).mjd
        sim_start = nominal_start + delay / (24.0 * 60.0)
    else:
        sim_start = sunset.mjd - 15 / 60 / 24

    if sim_nights is not None:
        # end at sunrise after sim_nights
        end_time = day_obs_time + TimeDelta(sim_nights, format="jd")
        sunrise = Time(
            observer.sun_rise_time(end_time, which="previous", horizon=-6 * u.deg),
            format="jd",
        )
        sim_end = sunrise.mjd
    else:
        # end at end of survey
        sim_end = survey_info["survey_end"].mjd

    # Set observatory MJD
    observatory.mjd = sim_start

    # The scheduler is noisy.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vals = sim_runner(
            observatory,
            scheduler,
            band_scheduler=band_scheduler,
            sim_start_mjd=sim_start,
            sim_duration=sim_end - sim_start,
            record_rewards=keep_rewards,
            verbose=True,
            anomalous_overhead_func=anomalous_overhead_func,
        )
    # Separate outputs.
    observatory = vals[0]
    scheduler = vals[1]
    sim_observations = vals[2]
    if len(vals) == 5:
        rewards = vals[3]
        obs_rewards = vals[4]
    else:
        rewards = []
        obs_rewards = []

    return sim_observations, scheduler, observatory, rewards, obs_rewards, survey_info


def simple_sim(
    day_obs: int,
    sim_nights: int,
    tokenfile: str | None = None,
    site: str = "usdf",
) -> tuple[pd.DataFrame, dict]:
    """Run a basic simulation starting at day_obs, running for sim_nights.
    With downtime and weather.

    Parameters
    ----------
    day_obs
        The integer day_obs on which to start the simulation.
    sim_nights
        The number of nights to run the simulation. If None, then run
        to the end of survey specified in survey_info.
    tokenfile
        Path to the RSP tokenfile.
        See also `rubin_nights.connections.get_access_token`.
        Default None will use `ACCESS_TOKEN` environment variable.
    site
        The site (`usdf`, `usdf-dev`, `summit` ..) location at
        which to query services. Must match tokenfile origin.

    Returns
    -------
    visits, survey_info : `pd.DataFrame`, `dict`
    """
    initial_opsim = fetch_previous_visits(
        day_obs=day_obs, tokenfile=tokenfile, site=site, convert_to_opsim=True
    )
    scheduler, initial_opsim, nside = setup_scheduler(
        config_script_path=CONFIG_SCRIPT_PATH,
        config_ddf_script_path=CONFIG_DDF_SCRIPT_PATH,
        day_obs=day_obs,
        initial_opsim=initial_opsim,
    )

    band_scheduler = setup_band_scheduler()

    observatory, survey_info = setup_observatory(
        day_obs=day_obs,
        nside=nside,
        add_downtime=True,
        add_clouds=True,
        seeing=None,
        real_downtime=True,
        initial_opsim=initial_opsim,
    )

    sim_observations, scheduler, observatory, rewards, obs_rewards, survey_info = run_sim(
        scheduler=scheduler,
        band_scheduler=band_scheduler,
        observatory=observatory,
        survey_info=survey_info,
        day_obs=day_obs,
        sim_nights=sim_nights,
        keep_rewards=False,
    )

    filename = f"lsst_{day_obs}_{sim_nights}.db"
    visits = lsst_support.save_opsim(observatory, sim_observations, initial_opsim, filename)

    return visits, survey_info


def fetch_lsst_visits_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Query the consdb for completed LSST visits")
    parser.add_argument("day_obs", type=int, help="Day_obs before which to query.")
    parser.add_argument("file_name", type=str, help="Name of opsim db file to write.")
    parser.add_argument("token_file", type=str, help="files with USDF access token")
    parser.add_argument(
        "--site", type=str, default="usdf", help="site of consdb to query (usdf, usdf-dev, or summit)"
    )
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    day_obs = args.day_obs
    file_name = args.file_name
    token_file = args.token_file
    site = args.site

    visits = fetch_previous_visits(day_obs, token_file, site=site)
    if visits is None:
        # Make an empty pd.DataFrame with opsim column names and types.
        visits = SchemaConverter().obs2opsim(ObservationArray()[0:0])

    with sqlite3.connect(file_name) as connection:
        visits.to_sql("observations", connection, index=False)

    return 0


def make_lsst_scheduler_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Create a pickle of an LSST scheduler")
    parser.add_argument("file_name", type=str, help="Name of pickle file to write.")
    parser.add_argument("--opsim", type=str, default="", help="Name of opsim visits file to load.")
    parser.add_argument(
        "--config_script",
        type=str,
        default=CONFIG_SCRIPT_PATH,
        help="Path to the config script for the scheduler.",
    )
    parser.add_argument(
        "--config_ddf_script",
        type=str,
        default=CONFIG_DDF_SCRIPT_PATH,
        help="Path to the config script for the DDF observations for this scheduler config.",
    )
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)
    opsim_fname = args.opsim
    scheduler_fname = args.file_name
    scheduler_config_script = args.config_script
    scheduler_ddf_config_script = args.config_ddf_script

    # The argparse argument of config_ddf_script cannot
    # be set to None on the command line, so use
    # the empty string as a sentinel value.
    if len(scheduler_ddf_config_script) == 0:
        scheduler_ddf_config_script = None

    scheduler, initial_opsim, nside = setup_scheduler(
        config_script_path=scheduler_config_script,
        config_ddf_script_path=scheduler_ddf_config_script,
        day_obs=None,
        opsim_filename=opsim_fname,
    )

    print("NSIDE: ", nside)

    # Save to a pickle
    with open(scheduler_fname, "wb") as sched_io:
        pickle.dump(scheduler, sched_io)

    return 0


def make_model_observatory_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Create a pickle of a model observatory")
    parser.add_argument("file_name", type=str, help="Name of pickle file to write.")
    parser.add_argument("--day_obs", type=int, default=None, help="day_obs for simulation start")
    parser.add_argument("--nside", type=int, default=32, help="nside for the model observatory.")
    parser.add_argument(
        "--include-downtime", action="store_true", dest="include_downtime", help="Include scheduled downtime"
    )
    parser.add_argument("--seeing", type=float, default=0, help="Seeing to use")
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    if args.day_obs is None:
        day_obs = rn_dayobs.today_day_obs()
    else:
        day_obs = args.day_obs

    observatory_fname = args.file_name
    nside = args.nside
    if args.include_downtime:
        add_downtime = True
    else:
        add_downtime = False
    seeing = None if args.seeing == 0 else args.seeing

    observatory, survey_info = setup_observatory(
        day_obs=day_obs,
        nside=nside,
        add_downtime=add_downtime,
        add_clouds=False,
        seeing=seeing,
        real_downtime=False,
    )

    # Save to a pickle
    with open(observatory_fname, "wb") as observatory_io:
        pickle.dump(observatory, observatory_io)

    return 0


def make_band_scheduler_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Create a pickle of a band scheduler")
    parser.add_argument("file_name", type=str, help="Name of pickle file to write.")
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)
    file_name = args.file_name

    band_scheduler = setup_band_scheduler()

    with open(file_name, "wb") as bs_io:
        pickle.dump(band_scheduler, bs_io)

    return 0


def run_lsst_sim_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Run an SV simulation.")
    parser.add_argument("scheduler", type=str, help="scheduler pickle file.")
    parser.add_argument("observatory", type=str, help="model observatory pickle file.")
    parser.add_argument("initial_opsim", type=str, help="initial opsim database.")
    parser.add_argument("day_obs", type=int, help="start day obs.")
    parser.add_argument("sim_nights", type=int, help="number of nights to run.")
    parser.add_argument("run_name", type=str, help="Run (also db output) name.")
    parser.add_argument("--keep_rewards", action="store_true", help="Compute rewards data.")
    parser.add_argument("--telescope", type=str, default="simonyi", help="The telescope simulated.")
    parser.add_argument("--label", type=str, default="", help="The tags on the simulation.")
    parser.add_argument("--delay", type=float, default=0.0, help="Minutes after nominal to start.")
    parser.add_argument("--anom_overhead_scale", type=float, default=0.0, help="scale of scatter in the slew")
    parser.add_argument(
        "--anom_overhead_seed",
        type=int,
        default=1,
        help="random number seed for anomalous scatter in overhead",
    )
    parser.add_argument("--tags", type=str, default=[], nargs="*", help="The tags on the simulation.")
    parser.add_argument("--results", type=str, default="", help="Results directory.")
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    with open(args.scheduler, "rb") as sched_io:
        scheduler = pickle.load(sched_io)

    with open(args.observatory, "rb") as obsv_io:
        observatory = pickle.load(obsv_io)

    band_scheduler = setup_band_scheduler()

    initial_opsim = None
    if len(args.initial_opsim) > 0:
        converter = SchemaConverter()
        initial_obs = converter.opsim2obs(args.initial_opsim)
        initial_opsim = converter.obs2opsim(initial_obs)

    day_obs = args.day_obs
    sim_nights = args.sim_nights
    run_name = args.run_name
    results_dir = args.results
    keep_rewards = args.keep_rewards
    tags = args.tags
    label = args.label
    telescope = args.telescope
    delay = args.delay
    anom_overhead_scale = args.anom_overhead_scale
    anom_overhead_seed = args.anom_overhead_seed

    if anom_overhead_scale > 0:
        anomalous_overhead_func = AnomalousOverheadFunc(anom_overhead_seed, anom_overhead_scale)
    else:
        anomalous_overhead_func = None

    if keep_rewards:
        scheduler.keep_rewards = keep_rewards

    survey_info = lsst_support.survey_times(add_downtime=False, day_obs=day_obs)

    LOGGER.info("Starting simulation")
    observations, scheduler, observatory, rewards, obs_rewards, survey_info = run_sim(
        scheduler=scheduler,
        band_scheduler=band_scheduler,
        observatory=observatory,
        survey_info=survey_info,
        day_obs=day_obs,
        sim_nights=sim_nights,
        anomalous_overhead_func=anomalous_overhead_func,
        keep_rewards=keep_rewards,
        delay=delay,
    )
    LOGGER.info("Simulation complete.")

    if len(results_dir) > 0:
        data_path = make_sim_data_dir(
            observations,
            rewards if keep_rewards else None,
            obs_rewards if keep_rewards else None,
            in_files={"scheduler": args.scheduler, "observatory": args.observatory},
            tags=tags,
            label=label,
            opsim_metadata={"telescope": telescope},
            data_path=results_dir,
        )
        LOGGER.info(f"Wrote results to directory: {data_path.name}")

    else:
        _ = lsst_support.save_opsim(observatory, observations, initial_opsim, run_name + ".db")

    return 0
