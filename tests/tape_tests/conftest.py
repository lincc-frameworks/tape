"""Test fixtures for Ensemble manipulations"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import pytest
import tape
import lsdb

from dask.distributed import Client

from tape import Ensemble
from tape.utils import ColumnMapper


@pytest.fixture
def create_test_rows():
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

    return rows


@pytest.fixture
def create_test_column_mapper():
    return ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")


@pytest.fixture
@pytest.mark.parametrize("create_test_rows", [("create_test_rows")])
def create_test_source_table(create_test_rows, npartitions=1):
    return dd.from_dict(create_test_rows, npartitions)


@pytest.fixture
def create_test_object_table(npartitions=1):
    n_obj = 5
    id = 8000 + np.arange(n_obj)
    name = id.astype(str)
    return dd.from_dict(dict(id=id, name=name), npartitions)


# pylint: disable=redefined-outer-name
@pytest.fixture
@pytest.mark.parametrize(
    "create_test_source_table, create_test_column_mapper",
    [("create_test_source_table", "create_test_column_mapper")],
)
def read_dask_dataframe_ensemble(create_test_source_table, create_test_column_mapper):
    return tape.read_dask_dataframe(
        dask_client=False,
        source_frame=create_test_source_table,
        column_mapper=create_test_column_mapper,
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
@pytest.mark.parametrize(
    "create_test_source_table, create_test_object_table, create_test_column_mapper",
    [("create_test_source_table", "create_test_object_table", "create_test_column_mapper")],
)
def read_dask_dataframe_with_object_ensemble(
    create_test_source_table, create_test_object_table, create_test_column_mapper
):
    return tape.read_dask_dataframe(
        source_frame=create_test_source_table,
        object_frame=create_test_object_table,
        dask_client=False,
        column_mapper=create_test_column_mapper,
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
@pytest.mark.parametrize(
    "create_test_rows, create_test_column_mapper", [("create_test_rows", "create_test_column_mapper")]
)
def read_pandas_ensemble(create_test_rows, create_test_column_mapper):
    return tape.read_pandas_dataframe(
        source_frame=pd.DataFrame(create_test_rows),
        column_mapper=create_test_column_mapper,
        dask_client=False,
        npartitions=1,
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
@pytest.mark.parametrize(
    "create_test_rows, create_test_column_mapper", [("create_test_rows", "create_test_column_mapper")]
)
def read_pandas_with_object_ensemble(create_test_rows, create_test_column_mapper):
    n_obj = 5
    id = 8000 + np.arange(n_obj)
    name = id.astype(str)
    object_table = pd.DataFrame(dict(id=id, name=name))

    """Create an Ensemble from pandas dataframes."""
    return tape.read_pandas_dataframe(
        dask_client=False,
        source_frame=pd.DataFrame(create_test_rows),
        object_frame=object_table,
        column_mapper=create_test_column_mapper,
        npartitions=1,
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
def read_parquet_ensemble_with_client(dask_client):
    """Create an Ensemble from parquet data with a dask client."""
    return tape.read_parquet(
        source_file="tests/tape_tests/data/source/test_source.parquet",
        object_file="tests/tape_tests/data/object/test_object.parquet",
        dask_client=dask_client,
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
def read_parquet_ensemble():
    """Create an Ensemble from parquet data."""
    return tape.read_parquet(
        source_file="tests/tape_tests/data/source/test_source.parquet",
        object_file="tests/tape_tests/data/object/test_object.parquet",
        dask_client=False,
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
def read_parquet_ensemble_from_source():
    """Create an Ensemble from parquet data, with object file withheld."""
    return tape.read_parquet(
        source_file="tests/tape_tests/data/source/test_source.parquet",
        dask_client=False,
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
def read_parquet_ensemble_with_column_mapper():
    """Create an Ensemble from parquet data, with object file withheld."""
    colmap = ColumnMapper().assign(
        id_col="ps1_objid",
        time_col="midPointTai",
        flux_col="psFlux",
        err_col="psFluxErr",
        band_col="filterName",
    )

    return tape.read_parquet(
        source_file="tests/tape_tests/data/source/test_source.parquet",
        column_mapper=colmap,
        dask_client=False,
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
def read_parquet_ensemble_with_known_column_mapper():
    """Create an Ensemble from parquet data, with object file withheld."""
    colmap = ColumnMapper().use_known_map("ZTF")

    return tape.read_parquet(
        source_file="tests/tape_tests/data/source/test_source.parquet",
        column_mapper=colmap,
        dask_client=False,
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
def read_parquet_ensemble_from_hipscat():
    """Create an Ensemble from a hipscat/hive-style directory."""

    colmap = ColumnMapper(
        id_col="_hipscat_index",
        time_col="mjd",
        flux_col="mag",
        err_col="Norder",  # no error column...
        band_col="band",
    )

    return tape.read_hipscat(
        "tests/tape_tests/data/small_sky_hipscat/small_sky_source_catalog",
        "tests/tape_tests/data/small_sky_hipscat/small_sky_object_catalog",
        column_mapper=colmap,
        object_index="id",
        source_index="object_id",
        dask_client=False,
    )


@pytest.fixture(scope="package", name="dask_client")
def dask_client():
    """Create a single client for use by all unit test cases."""
    client = Client()
    yield client
    client.close()


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_with_client(dask_client):
    """Create an Ensemble from parquet data without a dask client."""
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


@pytest.fixture
def parquet_files_and_ensemble_with_client(dask_client):
    """Create an Ensemble from parquet data without a dask client."""
    ens = Ensemble(client=dask_client)
    source_file = "tests/tape_tests/data/source/test_source.parquet"
    object_file = "tests/tape_tests/data/object/test_object.parquet"
    colmap = ColumnMapper().assign(
        id_col="ps1_objid",
        time_col="midPointTai",
        flux_col="psFlux",
        err_col="psFluxErr",
        band_col="filterName",
    )
    ens = ens.from_parquet(source_file, object_file, column_mapper=colmap)
    return ens, source_file, object_file, colmap


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble():
    """Create an Ensemble from parquet data."""
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
def parquet_ensemble_partition_size():
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=False)
    ens.from_parquet(
        "tests/tape_tests/data/source/test_source.parquet",
        "tests/tape_tests/data/object/test_object.parquet",
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
        partition_size="1MB",
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_with_divisions():
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=False)
    ens.from_parquet(
        "tests/tape_tests/data/source/test_source.parquet",
        "tests/tape_tests/data/object/test_object.parquet",
        id_col="ps1_objid",
        time_col="midPointTai",
        band_col="filterName",
        flux_col="psFlux",
        err_col="psFluxErr",
        sort=True,
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def parquet_ensemble_from_source():
    """Create an Ensemble from parquet data, with object file withheld."""
    ens = Ensemble(client=False)
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
def parquet_ensemble_with_column_mapper():
    """Create an Ensemble from parquet data, with object file withheld."""
    ens = Ensemble(client=False)

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
def parquet_ensemble_from_hipscat():
    """Create an Ensemble from a hipscat/hive-style directory."""
    ens = Ensemble(client=False)

    colmap = ColumnMapper(
        id_col="_hipscat_index",
        time_col="mjd",
        flux_col="mag",
        err_col="Norder",  # no error column...
        band_col="band",
    )

    ens.from_hipscat(
        "tests/tape_tests/data/small_sky_hipscat/small_sky_source_catalog",
        "tests/tape_tests/data/small_sky_hipscat/small_sky_object_catalog",
        column_mapper=colmap,
        object_index="id",
        source_index="object_id",
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def ensemble_from_lsdb():
    """Create a dask dataframe from LSDB catalogs"""
    object_cat = lsdb.read_hipscat("tests/tape_tests/data/small_sky_hipscat/small_sky_object_catalog")
    source_cat = lsdb.read_hipscat("tests/tape_tests/data/small_sky_hipscat/small_sky_source_catalog")

    # Pain points: Suffixes here are a bit annoying, and I'd ideally want just the source columns (especially at scale)
    # We do this to get the source catalog indexed by the objects hipscat index
    joined_source_cat = object_cat.join(
        source_cat, left_on="id", right_on="object_id", suffixes=("_object", "")
    )

    colmap = ColumnMapper(
        id_col="_hipscat_index",
        time_col="mjd",
        flux_col="mag",
        err_col="Norder",  # no error column...
        band_col="band",
    )

    ens = Ensemble(client=False)

    # We just avoid needing to invoke the ._ddf property from the catalogs
    ens.from_lsdb(joined_source_cat, object_cat, column_mapper=colmap)

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def read_ensemble_from_lsdb():
    """Create a dask dataframe from LSDB catalogs"""
    object_cat = lsdb.read_hipscat("tests/tape_tests/data/small_sky_hipscat/small_sky_object_catalog")
    source_cat = lsdb.read_hipscat("tests/tape_tests/data/small_sky_hipscat/small_sky_source_catalog")

    # Pain points: Suffixes here are a bit annoying, and I'd ideally want just the source columns (especially at scale)
    # We do this to get the source catalog indexed by the objects hipscat index
    joined_source_cat = object_cat.join(
        source_cat, left_on="id", right_on="object_id", suffixes=("_object", "")
    )

    colmap = ColumnMapper(
        id_col="_hipscat_index",
        time_col="mjd",
        flux_col="mag",
        err_col="Norder",  # no error column...
        band_col="band",
    )

    # We just avoid needing to invoke the ._ddf property from the catalogs
    ens = tape.read_lsdb(joined_source_cat, object_cat, column_mapper=colmap, dask_client=False)

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def dask_dataframe_ensemble():
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=False)

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
def dask_dataframe_ensemble_partition_size():
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=False)

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
        partition_size="1MB",
    )

    return ens


# pylint: disable=redefined-outer-name
@pytest.fixture
def dask_dataframe_with_object_ensemble():
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=False)

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
def pandas_ensemble():
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=False)

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
def pandas_with_object_ensemble():
    """Create an Ensemble from parquet data."""
    ens = Ensemble(client=False)

    n_obj = 5
    id = 8000 + np.arange(n_obj)
    name = id.astype(str)
    object_table = pd.DataFrame(
        dict(id=id, name=name),
    )

    num_points = 1000
    all_bands = np.array(["r", "g", "b", "i"])
    source_table = pd.DataFrame(
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


# pylint: disable=redefined-outer-name
@pytest.fixture
def ensemble_from_source_dict():
    """Create an Ensemble from a source dict, returning the ensemble and the source dict."""
    ens = Ensemble(client=False)

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
