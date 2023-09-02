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
def parquet_ensemble_without_client():
    """Create an Ensemble from parquet data without a dask client."""
    ens = Ensemble(client=False)
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

@pytest.fixture
def parquet_files_and_ensemble_without_client():
    """Create an Ensemble from parquet data without a dask client."""
    ens = Ensemble(client=False)
    source_file = "tests/tape_tests/data/source/test_source.parquet"
    object_file = "tests/tape_tests/data/object/test_object.parquet"
    colmap = ColumnMapper().assign(
        id_col="ps1_objid",
        time_col="midPointTai",
        flux_col="psFlux",
        err_col="psFluxErr",
        band_col="filterName",
    )
    ens.from_parquet(
        source_file,
        object_file,
        column_mapper=colmap
    )

    return ens, source_file, object_file, colmap

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

# pylint: disable=redefined-outer-name
@pytest.fixture
def ensemble_from_source_dict(dask_client):
    """Create an Ensemble from a source dict, returning the ensemble and the source dict."""
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # a few time steps, flux, and data for zero point calculations.
    source_dict = {
        "id": [8001, 8001, 8002, 8002, 8002],
        "time": [1, 2, 3, 4, 5],
        "flux": [30.5, 70, 80.6, 30.2, 60.3],
        "zp_mag": [25.0, 25.0, 25.0, 25.0, 25.0],
        "zp_flux": [10**10, 10**10, 10**10, 10**10, 10**10],
        "error": [10, 10, 10, 10, 10],
        "band": ["g", "g", "b", "b", "b"],
    }
    # map flux_col to one of the flux columns at the start
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="error", band_col="band")
    ens.from_source_dict(source_dict, column_mapper=cmap)

    return ens, source_dict