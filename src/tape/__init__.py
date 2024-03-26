import dask
import warnings


QUERY_PLANNING_ON = dask.config.get("dataframe.query-planning")
# Force the use of dask-expressions backends
if QUERY_PLANNING_ON is False:
    warnings.warn("This version of tape (v0.4.0+) requires dataframe query-planning, which has been enabled.")
    dask.config.set({"dataframe.query-planning": True})

from .analysis import *  # noqa
from .ensemble import *  # noqa
from .ensemble_frame import *  # noqa
from .timeseries import *  # noqa
from .ensemble_readers import *  # noqa
from ._version import __version__  # noqa
from .ensemble_frame import *  # noqa
