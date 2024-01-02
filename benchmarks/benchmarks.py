"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import os
import numpy as np
import pandas as pd
import tape
from tape.ensemble import Ensemble


TESTDATA_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "tape_tests", "data")


def load_parquet_data():
    return tape.read_parquet(
        source_file=os.path.join(TESTDATA_PATH, "source", "test_source.parquet"),
        object_file=os.path.join(TESTDATA_PATH, "object", "test_object.parquet"),
        dask_client=False,
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )


def time_batch():
    """Time a simple batch command"""
    ens = load_parquet_data()

    res = ens.batch(np.mean, "psFlux")
    res.compute()


def time_prune_sync_workflow():
    """Test a filter (using prune) -> sync workflow"""
    ens = load_parquet_data()

    ens.prune(50)  # calc nobs -> cut any object with nobs<50
    ens.source.head(5)  # should call sync implicitly


def time_basic_workflow():
    np.random.seed(1)

    # Generate 10 astronomical objects
    n_obj = 10
    ids = 8000 + np.arange(n_obj)
    names = ids.astype(str)
    object_table = pd.DataFrame(
        {
            "id": ids,
            "name": names,
            "ddf_bool": np.random.randint(0, 2, n_obj), # 0 if from deep drilling field, 1 otherwise
            "libid_cadence": np.random.randint(1, 130, n_obj),
        }
    )

    # Create 1000 lightcurves with 100 measurements each
    lc_len = 100
    num_points = 1000
    all_bands = np.array(["r", "g", "b", "i"])
    source_table = pd.DataFrame(
        {
            "id": 8000 + (np.arange(num_points) % n_obj),
            "time": np.arange(num_points),
            "flux":  np.random.random_sample(size=num_points)*10,
            "band": np.repeat(all_bands, num_points / len(all_bands)),
            "error": np.random.random_sample(size=num_points),
            "count": np.arange(num_points),
        },
    )

    ens = Ensemble()  # initialize an ensemble object

    # Read in the generated lightcurve data
    ens.from_pandas(
        source_frame=source_table,
        object_frame=object_table,
        id_col="id",
        time_col="time",
        flux_col="flux",
        err_col="error",
        band_col="band",
        npartitions=1)
