# lsst_survey_sim
Support for running simulations of the main LSST survey, primarily for prenight or progress evaluations.

Although there is a pyproject.toml file here, not all of the 
requirements will install properly (ts-config-scheduler, ts-fbs-utils). 

Installing as 
`pip install -e . --no-deps` seems like a good starting point for now. 

For ts-fbs-utils: 
`git clone git@github.com:lsst-ts/ts_fbs_utils.git`
Running from the develop branch seems safe, but running from the last tag is likely better.
`pip install -e . --no-deps` 

For ts-config-scheduler: 
`git clone git@github.com:lsst-ts/ts_config_scheduler.git`

This has to be run from the current run branch, latest commit. 
The run branch changes every few weeks. 
It can be found in JIRA with a query like: 
"Summary ~ "Support Summit Observing Weeks" and status not in (DONE, Invalid) order by duedate ASC"
(and often looks like the ticket branch starting with DM- that has the highest number)
You can either add 
`ts_config_scheduler/Scheduler/feature_scheduler/maintel/fbs_config_lsst_survey.py`
to your python path (`sys.path.insert(0, ts_config_ocs/Scheduler/feature_scheduler/maintel/fbs_config_lsst_survey.py)`) 
or symlink that file to your simulation working directory.

