"""Test fixtures for Ensemble manipulations"""
import numpy as np
import pandas as pd
import dask.dataframe as dd

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
def dask_dataframe_ensemble(dask_client):
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=dask_client)

    num_points = 1000
    all_bands = np.array(["r", "g", "b", "i"])
    rows = {
        "id": 8000 + (np.arange(num_points) % 5),
        "time": np.arange(num_points),
        "flux": np.arange(num_points) % len(all_bands),
        "band": np.repeat(all_bands, num_points / len(all_bands)),
        "err": 0.1 * (np.arange(num_points) % 10),
        "count": np.arange(num_points),
        "something_else": np.full(num_points, None),
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")

    ens.from_dask_dataframe(
        source_frame=dd.from_dict(rows, npartitions=1),
        column_mapper=cmap,
    )

    return ens

# pylint: disable=redefined-outer-name
@pytest.fixture
def dask_dataframe_with_object_ensemble(dask_client):
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=dask_client)

    n_obj = 5
    id = 8000 + np.arange(n_obj)
    name = id.astype(str)
    object_table = dd.from_dict(
        dict(id=id, name=name),
        npartitions=1,
    )

    num_points = 1000
    all_bands = np.array(["r", "g", "b", "i"])
    source_table = dd.from_dict(
        {
            "id": 8000 + (np.arange(num_points) % n_obj),
            "time": np.arange(num_points),
            "flux": np.arange(num_points) % len(all_bands),
            "band": np.repeat(all_bands, num_points / len(all_bands)),
            "err": 0.1 * (np.arange(num_points) % 10),
            "count": np.arange(num_points),
            "something_else": np.full(num_points, None),
        },
        npartitions=1,
    )
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")

    ens.from_dask_dataframe(
        source_frame=source_table,
        object_frame=object_table,
        column_mapper=cmap,
    )

    return ens

# pylint: disable=redefined-outer-name
@pytest.fixture
def pandas_ensemble(dask_client):
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=dask_client)

    num_points = 1000
    all_bands = np.array(["r", "g", "b", "i"])
    rows = {
        "id": 8000 + (np.arange(num_points) % 5),
        "time": np.arange(num_points),
        "flux": np.arange(num_points) % len(all_bands),
        "band": np.repeat(all_bands, num_points / len(all_bands)),
        "err": 0.1 * (np.arange(num_points) % 10),
        "count": np.arange(num_points),
        "something_else": np.full(num_points, None),
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")

    ens.from_pandas(
        pd.DataFrame(rows),
        column_mapper=cmap,
        npartitions=1,
    )

    return ens

# pylint: disable=redefined-outer-name
@pytest.fixture
def pandas_with_object_ensemble(dask_client):
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=dask_client)

    n_obj = 5
    id = 8000 + np.arange(n_obj)
    name = id.astype(str)
    object_table = pd.DataFrame(
        dict(id=id, name=name),
    )

    num_points = 1000
    all_bands = np.array(["r", "g", "b", "i"])
    source_table =pd.DataFrame(
        {
            "id": 8000 + (np.arange(num_points) % n_obj),
            "time": np.arange(num_points),
            "flux": np.arange(num_points) % len(all_bands),
            "band": np.repeat(all_bands, num_points / len(all_bands)),
            "err": 0.1 * (np.arange(num_points) % 10),
            "count": np.arange(num_points),
            "something_else": np.full(num_points, None),
        },
    )
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")

    ens.from_pandas(
        source_frame=source_table,
        object_frame=object_table,
        column_mapper=cmap,
        npartitions=1,
    )

    return ens
