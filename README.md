# lsst_survey_sim
Support for running simulations of the main LSST survey, primarily for prenight or progress evaluations.

There is a pyproject.toml file here, but ts_fbs_utils is not packaged outside
of the lsst-ts conda channel, and ts_config_scheduler is only available via git clone.

One option for installation of this package is as follows (note directions for handling $RUBIN_SIM_DATA_DIR can be found in [rubin_scheduler](https://rubin-scheduler.lsst.io/data-download.html#data-download) and [rubin_sim](https://rubin-sim.lsst.io/data-download.html#data-download) data download directions): 
```
pip install git+https://github.com/lsst-sims/lsst_survey_sim
scheduler_download_data --update
```

The advantage of the above is that you will have the appropriate versions of ts_fbs_utils and rubin_scheduler that should be in use at the summit and it's very relevant for running command-line simulations.
However, installing this way in the RSP environment is a problem, as almost all of the necessary packages are already installed in the RSP environment and it may not work easily to change the package version. It's also note so suitable for running the notebooks in the 'notebook' directory (a git clone works better for this).

In the RSP or for running with the notebooks:
```
git clone git@github.com:/lsst-sims/lsst_survey_sim.git
cd lsst_survey_sim
pip install -e . --no-deps --no-build-isolation
```
And then at the USDF RSP:
```
os.environ["RUBIN_SIM_DATA_DIR"] = "/sdf/data/rubin/shared/rubin_sim_data"
```
or follow the instructions to download the relevant rubin_scheduler and rubin_sim data as above.

The only required packages for lsst_survey_sims that are not provided in the RSP are ts_fbs_utils.  In general, installing and using `develop`
of ts-fbs-utils is fairly safe for new simulations, but checking the version in the lsst-survey-sims dependencies can be helpful. Either of
```
pip install --user git+https://github.com/lsst-ts/ts_fbs_utils
```
or 
```
git clone git@github.com:lsst-ts/ts_fbs_utils.git
cd ts_fbs_utils
pip install -e . --no-deps --no-build-isolation
```
are suitable.

The configurations for the FBS are kept in [ts_config_scheduler](https://github.com/lsst-ts/ts_config_scheduler) which can be independently cloned
```
git clone git@github.com:lsst-ts/ts_config_scheduler.git
```
or will be cloned for you when running the demo notebook. 
In general, ts_config_scheduler can be pointed to the tip of develop; sometimes a temporary run branch is deployed for patches during the night. 

An example of running a simulation is shown in `notebooks/lsst_eval.ipynb`.
A notebook similar to this is available in [Times Square](https://usdf-rsp.slac.stanford.edu/times-square/github/lsst/schedview_notebooks/prenight/run_sim).
