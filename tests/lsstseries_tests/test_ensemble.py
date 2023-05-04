"""Test ensemble manipulations"""
import copy

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from lsstseries import Ensemble
from lsstseries.analysis.stetsonj import calc_stetson_J
from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
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


def test_from_source_dict(dask_client):
    """
    Test that ensemble.from_source_dict() successfully creates data from a dictionary.
    """
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps. Leave out the flux data initially.
    rows = {
        ens._id_col: [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        ens._time_col: [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
        ens._band_col: ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        ens._err_col: [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }

    # We get an error without all of the required rows.
    with pytest.raises(ValueError):
        ens.from_source_dict(rows)

    # Add the last row and build the ensemble.
    rows[ens._flux_col] = [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ens.from_source_dict(rows)
    (obj_table, src_table) = ens.compute()

    # Check that the loaded source table is correct.
    assert src_table.shape[0] == 9
    for i in range(9):
        assert src_table.iloc[i][ens._flux_col] == rows[ens._flux_col][i]
        assert src_table.iloc[i][ens._time_col] == rows[ens._time_col][i]
        assert src_table.iloc[i][ens._band_col] == rows[ens._band_col][i]
        assert src_table.iloc[i][ens._err_col] == rows[ens._err_col][i]

    # Check that the derived object table is correct.
    assert obj_table.shape[0] == 2
    assert obj_table.iloc[0][ens._nobs_col] == 4
    assert obj_table.iloc[1][ens._nobs_col] == 5


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
    ens.from_source_dict(rows, npartitions=4)

    # Save the old data for comparison.
    old_data = ens.compute("source")
    old_div = copy.copy(ens._source.divisions)
    old_sizes = [len(ens._source.partitions[i]) for i in range(4)]
    assert old_data.shape[0] == num_points

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
    old_objects_pdf = parquet_ensemble._object.compute()
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

    # Sync the table and check that the number of objects decreased.
    parquet_ensemble._sync_tables()

    new_objects_pdf = parquet_ensemble._object.compute()
    assert len(new_objects_pdf.index) == len(old_objects_pdf.index) - 1

    # Assert the filtered ID is no longer in the objects.
    assert not valid_id in new_objects_pdf.index.values

    # Check that none of the other counts have changed.
    for i in new_objects_pdf.index.values:
        for c in new_objects_pdf.columns.values:
            assert new_objects_pdf.loc[i, c] == old_objects_pdf.loc[i, c]


def test_keep_zeros(parquet_ensemble):
    """Test that we can sync the tables and keep objects with zero sources."""
    parquet_ensemble.keep_empty_objects = True

    prev_npartitions = parquet_ensemble._object.npartitions
    old_objects_pdf = parquet_ensemble._object.compute()
    pdf = parquet_ensemble._source.compute()

    # Set the psFlux values for one object to NaN so we can drop it.
    # We do this on the instantiated object (pdf) and convert it back into a
    # Dask DataFrame.
    valid_id = pdf.index.values[1]
    pdf.loc[valid_id, parquet_ensemble._flux_col] = pd.NA
    parquet_ensemble._source = dd.from_pandas(pdf, npartitions=1)

    # Sync the table and check that the number of objects decreased.
    parquet_ensemble.dropna(1)
    parquet_ensemble._sync_tables()

    new_objects_pdf = parquet_ensemble._object.compute()
    assert len(new_objects_pdf.index) == len(old_objects_pdf.index)
    assert parquet_ensemble._object.npartitions == prev_npartitions

    # Check that all counts have stayed the same except the filtered index,
    # which should now be all zeros.
    for i in old_objects_pdf.index.values:
        for c in new_objects_pdf.columns.values:
            if i == valid_id:
                assert new_objects_pdf.loc[i, c] == 0
            else:
                assert new_objects_pdf.loc[i, c] == old_objects_pdf.loc[i, c]


def test_prune(parquet_ensemble):
    """
    Test that ensemble.prune() appropriately filters the dataframe
    """
    threshold = 10
    parquet_ensemble.prune(threshold)

    assert not np.any(parquet_ensemble._object["nobs_total"].values < threshold)


def test_find_day_gap_offset(dask_client):
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps.
    rows = {
        ens._id_col: [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        ens._time_col: [10.1, 10.2, 10.2, 11.1, 11.2, 10.9, 11.1, 15.0, 15.1],
        ens._flux_col: [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ens._band_col: ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        ens._err_col: [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    ens.from_source_dict(rows)
    gap_time = ens.find_day_gap_offset()
    assert abs(gap_time - 13.0 / 24.0) < 1e-6

    # Create fake observations covering all times
    rows = {
        ens._id_col: [8001] * 100,
        ens._time_col: [24.0 * (float(i) / 100.0) for i in range(100)],
        ens._flux_col: [1.0] * 100,
        ens._band_col: ["g"] * 100,
    }
    ens.from_source_dict(rows)
    assert ens.find_day_gap_offset() == -1


def test_bin_sources_day(dask_client):
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps.
    rows = {
        ens._id_col: [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        ens._time_col: [10.1, 10.2, 10.2, 11.1, 11.2, 10.9, 11.1, 15.0, 15.1],
        ens._flux_col: [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ens._band_col: ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        ens._err_col: [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    ens.from_source_dict(rows)

    # Check that the source table has 9 rows.
    old_source = ens.compute("source")
    assert old_source.shape[0] == 9

    # Bin the sources and check that we now have 6 rows.
    # This should throw a warning because we are overwriting the aggregation
    # for the time column.
    with pytest.warns():
        ens.bin_sources(
            time_window=1.0,
            offset=0.5,
            custom_aggr={ens._time_col: "min"},
            count_col="aggregated_bin_count",
        )
    new_source = ens.compute("source")
    assert new_source.shape[0] == 6
    assert new_source.shape[1] == 5

    # Check the results.
    list_to_check = [(8001, 0), (8001, 1), (8001, 2), (8002, 0), (8002, 1), (8002, 2)]
    expected_flux = [1.5, 5.0, 3.0, 1.0, 2.5, 4.5]
    expected_time = [10.1, 10.2, 11.1, 11.2, 10.9, 15.0]
    expected_band = ["g", "b", "g", "b", "g", "g"]
    expected_error = [1.118033988749895, 1.0, 3.0, 2.0, 2.5, 3.905124837953327]
    expected_count = [2, 1, 1, 1, 2, 2]

    for i in range(6):
        res = new_source.loc[list_to_check[i][0]].iloc[list_to_check[i][1]]
        assert abs(res[ens._flux_col] - expected_flux[i]) < 1e-6
        assert abs(res[ens._time_col] - expected_time[i]) < 1e-6
        assert abs(res[ens._err_col] - expected_error[i]) < 1e-6
        assert res[ens._band_col] == expected_band[i]
        assert res[ens._bin_count_col] == expected_count[i]


def test_bin_sources_two_days(dask_client):
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps.
    rows = {
        ens._id_col: [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002, 8002],
        ens._time_col: [10.1, 10.2, 10.2, 11.1, 11.2, 10.9, 11.1, 15.0, 15.1, 14.0],
        ens._flux_col: [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
        ens._band_col: ["g", "g", "b", "g", "b", "g", "g", "g", "g", "g"],
        ens._err_col: [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0],
    }
    ens.from_source_dict(rows)

    # Check that the source table has 10 rows.
    old_source = ens.compute("source")
    assert old_source.shape[0] == 10

    # Bin the sources and check that we now have 5 rows.
    # This should throw a warning because we are overwriting the aggregation
    # for the time column.
    ens.bin_sources(time_window=2.0, offset=0.5)
    new_source = ens.compute("source")
    assert new_source.shape[0] == 5

    # Check the results.
    list_to_check = [(8001, 0), (8001, 1), (8002, 0), (8002, 1), (8002, 2)]
    expected_flux = [2.0, 5.0, 1.0, 2.5, 4.666666]
    expected_time = [10.46666, 10.2, 11.2, 11.0, 14.70]
    expected_band = ["g", "b", "b", "g", "g"]

    for i in range(5):
        res = new_source.loc[list_to_check[i][0]].iloc[list_to_check[i][1]]
        assert abs(res[ens._flux_col] - expected_flux[i]) < 1e-3
        assert abs(res[ens._time_col] - expected_time[i]) < 1e-3
        assert res[ens._band_col] == expected_band[i]


@pytest.mark.parametrize("use_map", [True, False])
@pytest.mark.parametrize("on", [None, ["ps1_objid", "filterName"], ["nobs_total", "ps1_objid"]])
def test_batch(parquet_ensemble, use_map, on):
    """
    Test that ensemble.batch() returns the correct values of the first result
    """
    result = (
        parquet_ensemble.prune(10).dropna(1).batch(calc_stetson_J, use_map=use_map, on=on, band_to_calc=None)
    )

    if on is None:
        assert pytest.approx(result.values[0]["g"], 0.001) == -0.04174282
        assert pytest.approx(result.values[0]["r"], 0.001) == 0.6075282
    elif on is ["ps1_objid", "filterName"]:  # In case where we group on id and band, the structure changes
        assert pytest.approx(result.values[1]["r"], 0.001) == 0.6075282
        assert pytest.approx(result.values[0]["g"], 0.001) == -0.04174282
    elif on is ["nobs_total", "ps1_objid"]:
        assert pytest.approx(result.values[1]["g"], 0.001) == 1.2208577
        assert pytest.approx(result.values[1]["r"], 0.001) == -0.49639028


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

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = method
    arg_container.combine = combine
    arg_container.bin_count_target = sthresh

    res_sf2 = parquet_ensemble.sf2(argument_container=arg_container, use_map=use_map)
    res_batch = parquet_ensemble.batch(calc_sf2, use_map=use_map, argument_container=arg_container)

    if combine:
        assert not res_sf2.equals(res_batch)  # output should be different
    else:
        assert res_sf2.equals(res_batch)  # output should be identical


@pytest.mark.parametrize("sf_method", ["basic", "iqr", "macleod_2012"])
def test_sf2_methods(parquet_ensemble, sf_method, use_map=False):
    """
    Test calling sf2 from the ensemble
    """

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "loglength"
    arg_container.combine = False
    arg_container.bin_count_target = 50
    arg_container.sf_method = sf_method

    res_sf2 = parquet_ensemble.sf2(argument_container=arg_container, use_map=use_map)
    res_batch = parquet_ensemble.batch(calc_sf2, use_map=use_map, argument_container=arg_container)

    assert res_sf2.equals(res_batch)  # output should be identical
