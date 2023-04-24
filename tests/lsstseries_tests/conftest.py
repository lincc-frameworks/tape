"""Test fixtures for Ensemble manipulations"""
import pytest
from dask.distributed import Client

from lsstseries import Ensemble


@pytest.fixture(scope="package", name="dask_client")
def dask_client():
    """Create a single client for use by all unit test cases."""
    client = Client()
    yield client
    client.close()


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble(dask_client):
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=dask_client)
    ens.from_parquet(
        "tests/lsstseries_tests/data/source/test_source.parquet",
        "tests/lsstseries_tests/data/object/test_object.parquet",
        id_col="ps1_objid",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_from_source(dask_client):
    """Create an Ensemble from parquet data, with object file withheld."""
    ens = Ensemble(client=dask_client)
    ens.from_parquet(
        "tests/lsstseries_tests/data/source/test_source.parquet",
        id_col="ps1_objid",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_from_hipscat(dask_client):
    """Create an Ensemble from a hipscat/hive-style directory."""
    ens = Ensemble(client=dask_client)
    ens.from_hipscat(
        "tests/lsstseries_tests/data",
        id_col="ps1_objid",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )

    return ens
