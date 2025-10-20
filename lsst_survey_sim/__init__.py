from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lsst_survey_sim")
except PackageNotFoundError:
    # package is not installed
    pass
