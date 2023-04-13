"""Test ensemble manipulations"""
import copy
import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytest

from lsstseries import Ensemble
from lsstseries.analysis.stetsonj import calc_stetson_J
from lsstseries.analysis.structurefunction2 import calc_sf2


# pylint: disable=protected-access
def test_with():
    """Test that we open and close a client on enter and exit."""
    with Ensemble() as ens:
        ens.from_parquet(
            "tests/lsstseries_tests/data/test_subset.parquet",
            id_col="ps1_objid",
            band_col="filterName",
            flux_col="psFlux",
            err_col="psFluxErr",
        )
        assert ens._data is not None


def test_from_parquet(parquet_ensemble):
    """
    Test that ensemble.from_parquet() successfully loads a parquet file
    """
    # Check to make sure the source and object tables were created
    assert parquet_ensemble._source is not None
    assert parquet_ensemble._object is not None

    # Check that the data is not empty.
    (_, parquet_ensemble._source) = parquet_ensemble.compute()
    assert parquet_ensemble._source.size > 0

    # Check the we loaded the correct columns.
    for col in [
        parquet_ensemble._time_col,
        parquet_ensemble._flux_col,
        parquet_ensemble._err_col,
        parquet_ensemble._band_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert parquet_ensemble._source[col] is not None


def test_insert(parquet_ensemble):
    num_partitions = parquet_ensemble._source.npartitions
    (old_object, old_source) = parquet_ensemble.compute()
    old_size = old_source.shape[0]

    # Save the column names to shorter strings
    id_col = parquet_ensemble._id_col
    time_col = parquet_ensemble._time_col
    flux_col = parquet_ensemble._flux_col
    err_col = parquet_ensemble._err_col
    band_col = parquet_ensemble._band_col

    # Test an insertion of 5 observations.
    new_inds = [2, 1, 100, 110, 111]
    new_bands = ["g", "r", "sky_blue", "b", "r"]
    new_times = [1.0, 1.1, 1.2, 1.3, 1.4]
    new_fluxes = [2.0, 2.5, 3.0, 3.5, 4.0]
    new_errs = [0.1, 0.05, 0.01, 0.05, 0.01]
    parquet_ensemble.insert_sources(
        new_inds, new_bands, new_times, new_fluxes, new_errs, force_repartition=True
    )

    # Check we did not increase the number of partitions.
    assert parquet_ensemble._source.npartitions == num_partitions

    # Check that all the new data points are in there. The order may be different
    # due to the repartitioning.
    (new_obj, new_source) = parquet_ensemble.compute()
    assert new_source.shape[0] == old_size + 5
    for i in range(5):
        assert new_source.loc[new_inds[i]][time_col] == new_times[i]
        assert new_source.loc[new_inds[i]][flux_col] == new_fluxes[i]
        assert new_source.loc[new_inds[i]][err_col] == new_errs[i]
        assert new_source.loc[new_inds[i]][band_col] == new_bands[i]

    # Check that all of the old data is still in there.
    obj_ids = old_source.index.unique()
    for idx in obj_ids:
        assert old_source.loc[idx].shape[0] == new_source.loc[idx].shape[0]

    # Insert a few more observations without repartitioning.
    new_inds2 = [2, 1, 100, 110, 111]
    new_bands2 = ["r", "g", "b", "r", "g"]
    new_times2 = [10.0, 10.1, 10.2, 10.3, 10.4]
    new_fluxes2 = [2.0, 2.5, 3.0, 3.5, 4.0]
    new_errs2 = [0.1, 0.05, 0.01, 0.05, 0.01]
    parquet_ensemble.insert_sources(
        new_inds2, new_bands2, new_times2, new_fluxes2, new_errs2, force_repartition=False
    )

    # Check we *did* increase the number of partitions and the size increased.
    assert parquet_ensemble._source.npartitions != num_partitions
    (new_obj, new_source) = parquet_ensemble.compute()
    assert new_source.shape[0] == old_size + 10


def test_insert_paritioned(dask_client):
    ens = Ensemble(client=dask_client)

    # Create all fake source data with known divisions.
    num_points = 1000
    all_bands = ["r", "g", "b", "i"]
    rows = {
        ens._id_col: [8000 + 2 * i for i in range(num_points)],
        ens._time_col: [float(i) for i in range(num_points)],
        ens._flux_col: [0.5 * float(i) for i in range(num_points)],
        ens._band_col: [all_bands[i % 4] for i in range(num_points)],
    }
    ddf = dd.DataFrame.from_dict(rows, npartitions=4)
    ddf = ddf.set_index(ens._id_col, drop=True)
    assert ddf.known_divisions

    # Save the old data for comparison.
    old_data = ddf.compute()
    old_div = copy.copy(ddf.divisions)
    old_sizes = [len(ddf.partitions[i]) for i in range(4)]
    assert old_data.shape[0] == num_points

    # Directly set the dask data set.
    ens._source = ddf
    ens._object = ens._generate_object_table()

    # Test an insertion of 5 observations.
    new_inds = [8001, 8003, 8005, 9005, 9007]
    new_bands = ["g", "r", "sky_blue", "b", "r"]
    new_times = [1.0, 1.1, 1.2, 1.3, 1.4]
    new_fluxes = [2.0, 2.5, 3.0, 3.5, 4.0]
    new_errs = [0.1, 0.05, 0.01, 0.05, 0.01]
    ens.insert_sources(new_inds, new_bands, new_times, new_fluxes, new_errs, force_repartition=True)

    # Check we did not increase the number of partitions and the points
    # were placed in the correct partitions.
    assert ens._source.npartitions == 4
    assert ens._source.divisions == old_div
    assert len(ens._source.partitions[0]) == old_sizes[0] + 3
    assert len(ens._source.partitions[1]) == old_sizes[1]
    assert len(ens._source.partitions[2]) == old_sizes[2] + 2
    assert len(ens._source.partitions[3]) == old_sizes[3]

    # Check that all the new data points are in there. The order may be different
    # due to the repartitioning.
    (new_obj, new_source) = ens.compute()
    assert new_source.shape[0] == num_points + 5
    for i in range(5):
        assert new_source.loc[new_inds[i]][ens._time_col] == new_times[i]
        assert new_source.loc[new_inds[i]][ens._flux_col] == new_fluxes[i]
        assert new_source.loc[new_inds[i]][ens._err_col] == new_errs[i]
        assert new_source.loc[new_inds[i]][ens._band_col] == new_bands[i]

    # Insert a bunch of points into the second partition.
    new_inds = [8804, 8804, 8804, 8804, 8804]
    ens.insert_sources(new_inds, new_bands, new_times, new_fluxes, new_errs, force_repartition=True)

    # Check we did not increase the number of partitions and the points
    # were placed in the correct partitions.
    assert ens._source.npartitions == 4
    assert ens._source.divisions == old_div
    assert len(ens._source.partitions[0]) == old_sizes[0] + 3
    assert len(ens._source.partitions[1]) == old_sizes[1] + 5
    assert len(ens._source.partitions[2]) == old_sizes[2] + 2
    assert len(ens._source.partitions[3]) == old_sizes[3]


def test_core_wrappers(parquet_ensemble):
    """
    Test that the core wrapper functions execute without errors
    """
    # Just test if these execute successfully
    parquet_ensemble.client_info()
    parquet_ensemble.info()
    parquet_ensemble.columns()
    parquet_ensemble.head(n=5)
    parquet_ensemble.tail(n=5)
    parquet_ensemble.compute()


def test_sync_tables(parquet_ensemble):
    """
    Test that _table_sync works as expected
    """

    assert len(parquet_ensemble.compute("object")) == 15
    assert len(parquet_ensemble.compute("source")) == 2000

    parquet_ensemble.prune(50, col_name="nobs_r").prune(50, col_name="nobs_g")
    assert parquet_ensemble._object_dirty  # Prune should set the object dirty flag

    parquet_ensemble.dropna(1)
    assert parquet_ensemble._source_dirty  # Dropna should set the source dirty flag

    parquet_ensemble._sync_tables()

    # both tables should have the expected number of rows after a sync
    assert len(parquet_ensemble.compute("object")) == 5
    assert len(parquet_ensemble.compute("source")) == 1562

    # dirty flags should be unset after sync
    assert not parquet_ensemble._object_dirty
    assert not parquet_ensemble._source_dirty


def test_dropna(parquet_ensemble):
    pdf = parquet_ensemble._source.compute()
    parquet_length = len(pdf.index)

    # Try dropping NaNs and confirm nothing is dropped (there are no NaNs).
    parquet_ensemble.dropna(1)
    assert len(parquet_ensemble._source.compute().index) == parquet_length

    # Get a valid ID to use and count its occurrences.
    valid_id = pdf.index.values[1]
    occurrences = len(pdf.loc[valid_id].values)

    # Set the psFlux values for one object to NaN so we can drop it.
    # We do this on the instantiated object (pdf) and convert it back into a
    # Dask DataFrame.
    pdf.loc[valid_id, parquet_ensemble._flux_col] = pd.NA
    parquet_ensemble._source = dd.from_pandas(pdf, npartitions=1)

    # Try dropping NaNs and confirm that we did.
    parquet_ensemble.dropna(1)
    assert len(parquet_ensemble._source.compute().index) == parquet_length - occurrences


def test_prune(parquet_ensemble):
    """
    Test that ensemble.prune() appropriately filters the dataframe
    """
    threshold = 10
    parquet_ensemble.prune(threshold)

    assert not np.any(parquet_ensemble._object["nobs_total"].values < threshold)


@pytest.mark.parametrize("use_map", [True, False])
def test_batch(parquet_ensemble, use_map):
    """
    Test that ensemble.batch() returns the correct values of the first result
    """
    result = parquet_ensemble.prune(10).dropna(1).batch(calc_stetson_J, use_map=use_map, band_to_calc=None)

    assert pytest.approx(result.values[0]["g"], 0.001) == -0.04174282
    assert pytest.approx(result.values[0]["r"], 0.001) == 0.6075282


def test_to_timeseries(parquet_ensemble):
    """
    Test that ensemble.to_timeseries() runs and assigns the correct metadata
    """
    ts = parquet_ensemble.to_timeseries(88480000290704349)

    assert ts.meta["id"] == 88480000290704349


def test_build_index(dask_client):
    """
    Test that ensemble indexing returns expected behavior
    """

    obj_ids = [1, 1, 1, 2, 1, 2, 2]
    bands = ["u", "u", "u", "g", "g", "u", "u"]

    ens = Ensemble(client=dask_client)
    result = ens._build_index(obj_ids, bands)
    assert len(result.levels) == 3

    result_ids = list(result.get_level_values(0))
    assert result_ids == obj_ids

    result_bands = list(result.get_level_values(1))
    assert result_bands == bands

    result_ids = list(result.get_level_values(2))
    target = [0, 1, 2, 0, 0, 0, 1]
    assert result_ids == target


@pytest.mark.parametrize("method", ["size", "length", "loglength"])
@pytest.mark.parametrize("combine", [True, False])
@pytest.mark.parametrize("sthresh", [50, 100])
def test_sf2(parquet_ensemble, method, combine, sthresh, use_map=False):
    """
    Test calling sf2 from the ensemble
    """

    res_sf2 = parquet_ensemble.sf2(combine=combine, method=method, sthresh=sthresh, use_map=use_map)
    res_batch = parquet_ensemble.batch(
        calc_sf2, use_map=use_map, combine=combine, method=method, sthresh=sthresh
    )

    if combine:
        assert not res_sf2.equals(res_batch)  # output should be different
    else:
        assert res_sf2.equals(res_batch)  # output should be identical
