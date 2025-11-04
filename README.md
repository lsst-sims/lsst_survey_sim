# lsst_survey_sim
Support for running simulations of the main LSST survey, primarily for prenight or progress evaluations.

There is a pyproject.toml file here, but ts_fbs_utils is not packaged outside
of the lsst-ts conda channel, and ts_config_scheduler is only available via git clone.

One option for installation of this package is as follows (note directions for handling $RUBIN_SIM_DATA_DIR can be found in [rubin_scheduler](https://rubin-scheduler.lsst.io/data-download.html#data-download) and [rubin_sim](https://rubin-sim.lsst.io/data-download.html#data-download) data download directions): 
```
pip install git+https://github.com/lsst-sims/lsst_survey_sim
scheduler_download_data --update
rs_download_data --update --dirs sim_baseline
```

The advantage of the above is that you will have the appropriate versions of ts_fbs_utils and rubin_scheduler that should be in use at the summit.
However, installing this way in the RSP environment is a problem, as almost all of the necessary packages are already installed in the RSP environment. 

In the RSP, 
```
git clone git@github.com:/lsst-sims/lsst_survey_sim.git
cd lsst_survey_sim
pip install -e . --no-deps
```
is a better choice. In the RSP, setting 
```
os.environ["RUBIN_SIM_DATA_DIR"] = "/sdf/data/rubin/shared/rubin_sim_data"
```
will provide access to the necessary scheduler and rubin-sim data.

Then ts_fbs_utils will have to be installed. In general, installing and using `develop`
of ts-fbs-utils is fairly safe for new simulations, but checking the version in the lsst-survey-sims dependencies can be helpful. Either
```
pip install git+https://github.com/lsst-ts/ts_fbs_utils
```
or 
```
git clone git@github.com:lsst-ts/ts_fbs_utils.git
cd ts_fbs_utils
pip install -e . --no-deps
```
are suitable.

The configurations for the FBS are kept in [ts_config_scheduler](https://github.com/lsst-ts/ts_config_scheduler) which can be independently cloned
```
git clone git@github.com:lsst-ts/ts_config_scheduler.git
```
or will be cloned for you when running the demo notebook. 
In general, ts-config-scheduler should be poitned to the current run branch, which changes every few weeks.
The run branch can be found in JIRA with a query like: 
"Summary ~ "Support Summit Observing Weeks" and status not in (DONE, Invalid) order by duedate ASC"
(and often looks like the ticket branch starting with DM- that has the highest number)
or is tracked in the lsst.obsenv.run_branch topic. 

An example of running a simulation is shown in `notebooks/lsst_eval.ipynb`.

