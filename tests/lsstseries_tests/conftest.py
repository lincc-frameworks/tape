"""Test fixtures for ensemble manipulations"""
import pytest
from dask.distributed import Client

from lsstseries import ensemble


@pytest.fixture(scope="package", name="dask_client")
def dask_client():
    """Create a single client for use by all unit test cases."""
    client = Client()
    yield client
    client.close()


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble(dask_client):
    """Create an ensemble from parquet data."""
    ens = ensemble(client=dask_client)
    ens.from_parquet(
        "tests/lsstseries_tests/data/test_subset.parquet",
        id_col="ps1_objid",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )

    return ens
