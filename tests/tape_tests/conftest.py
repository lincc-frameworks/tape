"""Test fixtures for Ensemble manipulations"""
import pytest
from dask.distributed import Client

from tape import Ensemble
from tape.utils import ColumnMapper


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
        "tests/tape_tests/data/source/test_source.parquet",
        "tests/tape_tests/data/object/test_object.parquet",
        id_col="ps1_objid",
        time_col="midPointTai",
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
        "tests/tape_tests/data/source/test_source.parquet",
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_with_column_mapper(dask_client):
    """Create an Ensemble from parquet data, with object file withheld."""
    ens = Ensemble(client=dask_client)

    colmap = ColumnMapper().assign(
        id_col="ps1_objid",
        time_col="midPointTai",
        flux_col="psFlux",
        err_col="psFluxErr",
        band_col="filterName",
    )
    ens.from_parquet(
        "tests/tape_tests/data/source/test_source.parquet",
        column_mapper=colmap,
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_with_known_column_mapper(dask_client):
    """Create an Ensemble from parquet data, with object file withheld."""
    ens = Ensemble(client=dask_client)

    colmap = ColumnMapper().use_known_map("ZTF")
    ens.from_parquet(
        "tests/tape_tests/data/source/test_source.parquet",
        column_mapper=colmap,
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_from_hipscat(dask_client):
    """Create an Ensemble from a hipscat/hive-style directory."""
    ens = Ensemble(client=dask_client)
    ens.from_hipscat(
        "tests/tape_tests/data",
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )

    return ens


@pytest.fixture(scope="package", name="dask_bench_client")
def dask_bench_client():
    """Create a single client for use by all unit test cases."""
    client = Client(n_workers=1, threads_per_worker=1)
    yield client
    client.close()


# pylint: disable=redefined-outer-name
@pytest.fixture
def bench_ensemble(dask_bench_client):
    """Create an Ensemble from parquet data for benchmarking."""
    ens = Ensemble(client=dask_bench_client)
    ens.from_parquet(
        "tests/tape_tests/data/source/test_source.parquet",
        "tests/tape_tests/data/object/test_object.parquet",
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )

    return ens
