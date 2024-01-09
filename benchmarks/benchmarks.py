"""A suite of TAPE Benchmarks.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import os
import numpy as np
import tape


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
