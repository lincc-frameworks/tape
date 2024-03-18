"""Test ensemble manipulations"""

import copy
import os
import lsdb

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import tape

from tape import (
    Ensemble,
    EnsembleFrame,
    EnsembleSeries,
    ObjectFrame,
    SourceFrame,
    TapeFrame,
    TapeSeries,
    TapeObjectFrame,
    TapeSourceFrame,
    TimeSeries,
)
from tape.analysis.stetsonj import calc_stetson_J
from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structurefunction2 import calc_sf2
from tape.utils import ColumnMapper


# pylint: disable=protected-access
def test_with_client():
    """Test that we open and close a client on enter and exit."""
    with Ensemble() as ens:
        ens.from_parquet(
            "tests/tape_tests/data/source/test_source.parquet",
            id_col="ps1_objid",
            band_col="filterName",
            flux_col="psFlux",
            err_col="psFluxErr",
        )
        assert ens._data is not None


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
        "parquet_ensemble_from_source",
        "parquet_ensemble_with_column_mapper",
        "parquet_ensemble_partition_size",
        "read_parquet_ensemble",
        "read_parquet_ensemble_from_source",
        "read_parquet_ensemble_with_column_mapper",
    ],
)
def test_parquet_construction(data_fixture, request):
    """
    Test that ensemble loader functions successfully load parquet files
    """
    parquet_ensemble = request.getfixturevalue(data_fixture)

    # Check to make sure the source and object tables were created
    assert parquet_ensemble.source is not None
    assert parquet_ensemble.object is not None

    # Make sure divisions are set
    if data_fixture == "parquet_ensemble_with_divisions":
        assert parquet_ensemble.source.known_divisions
        assert parquet_ensemble.object.known_divisions

    # Check that the data is not empty.
    obj, source = parquet_ensemble.compute()
    assert len(source) == 2000
    assert len(obj) == 15

    # Check that source and object both have the same ids present
    assert sorted(np.unique(list(source.index))) == sorted(np.array(obj.index))

    # Check the we loaded the correct columns.
    for col in [
        parquet_ensemble._time_col,
        parquet_ensemble._flux_col,
        parquet_ensemble._err_col,
        parquet_ensemble._band_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert parquet_ensemble.source[col] is not None


@pytest.mark.parametrize(
    "data_fixture",
    [
        "dask_dataframe_ensemble",
        "dask_dataframe_ensemble_partition_size",
        "dask_dataframe_with_object_ensemble",
        "pandas_ensemble",
        "pandas_with_object_ensemble",
        "read_dask_dataframe_ensemble",
        "read_dask_dataframe_with_object_ensemble",
        "read_pandas_ensemble",
        "read_pandas_with_object_ensemble",
    ],
)
def test_dataframe_constructors(data_fixture, request):
    """
    Tests constructing an ensemble from pandas and dask dataframes.
    """
    ens = request.getfixturevalue(data_fixture)

    # Check to make sure the source and object tables were created
    assert ens.source is not None
    assert ens.object is not None

    # Check that the data is not empty.
    obj, source = ens.compute()
    assert len(source) == 1000
    assert len(obj) == 5

    # Check that source and object both have the same ids present
    np.testing.assert_array_equal(np.unique(source.index), np.sort(obj.index))

    # Check the we loaded the correct columns.
    for col in [
        ens._time_col,
        ens._flux_col,
        ens._err_col,
        ens._band_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert ens.source[col] is not None

    # Check that we can compute an analysis function on the ensemble.
    amplitude = ens.batch(calc_stetson_J)
    assert len(amplitude) == 5


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble_from_hipscat",
        "read_parquet_ensemble_from_hipscat",
        "read_parquet_ensemble_from_hipscat",
        "ensemble_from_lsdb",
        "read_ensemble_from_lsdb",
    ],
)
def test_hipscat_constructors(data_fixture, request):
    """
    Tests constructing an ensemble from LSDB and hipscat
    """
    parquet_ensemble = request.getfixturevalue(data_fixture)

    # Check to make sure the source and object tables were created
    assert parquet_ensemble.source is not None
    assert parquet_ensemble.object is not None

    # Make sure divisions are set
    assert parquet_ensemble.source.known_divisions
    assert parquet_ensemble.object.known_divisions

    # Check that the data is not empty.
    obj, source = parquet_ensemble.compute()
    assert len(source) == 16135  # full source is 17161, but we drop some in the join with object
    assert len(obj) == 131

    # Check that source and object both have the same ids present
    assert sorted(np.unique(list(source.index))) == sorted(np.array(obj.index))

    # Check the we loaded the correct columns.
    for col in [
        parquet_ensemble._time_col,
        parquet_ensemble._flux_col,
        parquet_ensemble._err_col,
        parquet_ensemble._band_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert parquet_ensemble.source[col] is not None


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_client",
    ],
)
def test_update_ensemble(data_fixture, request):
    """
    Tests that the ensemble can be updated with a result frame.
    """
    ens = request.getfixturevalue(data_fixture)

    # Filter the object table and have the ensemble track the updated table.
    updated_obj = ens.object.query("nobs_total > 50")
    assert updated_obj is not ens.object
    assert updated_obj.is_dirty()
    # Update the ensemble and validate that it marks the object table dirty
    assert ens.object.is_dirty() == False
    updated_obj.update_ensemble()
    assert ens.object.is_dirty() == True
    assert updated_obj is ens.object

    # Filter the source table and have the ensemble track the updated table.
    updated_src = ens.source.query("psFluxErr > 0.1")
    assert updated_src is not ens.source
    # Update the ensemble and validate that it marks the source table dirty
    assert ens.source.is_dirty() == False
    updated_src.update_ensemble()
    assert ens.source.is_dirty() == True
    assert updated_src is ens.source

    # Compute a result to trigger a table sync
    obj, src = ens.compute()
    assert len(obj) > 0
    assert len(src) > 0
    assert ens.object.is_dirty() == False
    assert ens.source.is_dirty() == False

    # Create an additional result table for the ensemble to track.
    cnts = ens.source.groupby([ens._id_col, ens._band_col])[ens._time_col].aggregate("count")
    res = (
        cnts.to_frame()
        .reset_index()
        .categorize(columns=[ens._band_col])
        .pivot_table(values=ens._time_col, index=ens._id_col, columns=ens._band_col, aggfunc="sum")
    )

    # Convert the resulting dataframe into an EnsembleFrame and update the Ensemble
    result_frame = EnsembleFrame.from_dask_dataframe(res, ensemble=ens, label="result")
    result_frame.update_ensemble()
    assert ens.select_frame("result") is result_frame

    # Test update_ensemble when a frame is unlinked to its parent ensemble.
    result_frame.ensemble = None
    assert result_frame.update_ensemble() is None


def test_available_datasets(dask_client):
    """
    Test that the ensemble is able to successfully read in the list of available TAPE datasets
    """
    ens = Ensemble(client=dask_client)

    datasets = ens.available_datasets()

    assert isinstance(datasets, dict)
    assert len(datasets) > 0  # Find at least one


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_files_and_ensemble_with_client",
    ],
)
def test_frame_tracking(data_fixture, request):
    """
    Tests a workflow of adding and removing the frames tracked by the Ensemble.
    """
    ens, source_file, object_file, colmap = request.getfixturevalue(data_fixture)

    # Since we load the ensemble from a parquet, we expect the Source and Object frames to be populated.
    assert len(ens.frames) == 2
    assert isinstance(ens.select_frame("source"), SourceFrame)
    assert isinstance(ens.select_frame("object"), ObjectFrame)

    # Check that we can select source and object frames
    assert len(ens.frames) == 2
    assert ens.select_frame("source") is ens.source
    assert isinstance(ens.select_frame("source"), SourceFrame)
    assert ens.select_frame("object") is ens.object
    assert isinstance(ens.select_frame("object"), ObjectFrame)

    # Construct some result frames for the Ensemble to track. Underlying data is irrelevant for
    # this test.
    num_points = 100
    data = TapeFrame(
        {
            "id": [8000 + 2 * i for i in range(num_points)],
            "time": [float(i) for i in range(num_points)],
            "flux": [0.5 * float(i % 4) for i in range(num_points)],
        }
    )
    # Labels to give the EnsembleFrames
    label1, label2, label3 = "frame1", "frame2", "frame3"
    ens_frame1 = EnsembleFrame.from_tapeframe(data, npartitions=1, ensemble=ens, label=label1)
    ens_frame2 = EnsembleFrame.from_tapeframe(data, npartitions=1, ensemble=ens, label=label2)
    ens_frame3 = EnsembleFrame.from_tapeframe(data, npartitions=1, ensemble=ens, label=label3)

    # Validate that new source and object frames can't be added or updated.
    with pytest.raises(ValueError):
        ens.add_frame(ens_frame1, "source")
    with pytest.raises(ValueError):
        ens.add_frame(ens_frame1, "object")

    # Test that we can add and select a new ensemble frame
    assert ens.add_frame(ens_frame1, label1).select_frame(label1) is ens_frame1
    assert len(ens.frames) == 3

    # Validate that we can't add a new frame that uses an exisiting label
    with pytest.raises(ValueError):
        ens.add_frame(ens_frame2, label1)

    # We add two more frames to track
    ens.add_frame(ens_frame2, label2).add_frame(ens_frame3, label3)
    assert ens.select_frame(label2) is ens_frame2
    assert ens.select_frame(label3) is ens_frame3
    assert len(ens.frames) == 5

    # Now we begin dropping frames. First verify that we can't drop object or source.
    with pytest.raises(ValueError):
        ens.drop_frame("source")
    with pytest.raises(ValueError):
        ens.drop_frame("object")

    # And verify that we can't call drop with an unknown label.
    with pytest.raises(KeyError):
        ens.drop_frame("nonsense")

    # Drop an existing frame and that it can no longer be selected.
    ens.drop_frame(label3)
    assert len(ens.frames) == 4
    with pytest.raises(KeyError):
        ens.select_frame(label3)

    # Update the ensemble with the dropped frame, and then select the frame
    assert ens.update_frame(ens_frame3).select_frame(label3) is ens_frame3
    assert len(ens.frames) == 5

    # Update the ensemble with an unlabeled frame, verifying a missing label generates an error.
    ens_frame4 = EnsembleFrame.from_tapeframe(data, npartitions=1, ensemble=ens, label=None)
    label4 = "frame4"
    with pytest.raises(ValueError):
        ens.update_frame(ens_frame4)
    ens_frame4.label = label4
    assert ens.update_frame(ens_frame4).select_frame(label4) is ens_frame4
    assert len(ens.frames) == 6

    # Change the label of the 4th ensemble frame to verify update overrides an existing frame
    ens_frame4.label = label3
    assert ens.update_frame(ens_frame4).select_frame(label3) is ens_frame4
    assert len(ens.frames) == 6


def test_from_rrl_dataset(dask_client):
    """
    Test a basic load and analyze workflow from the S82 RR Lyrae Dataset
    """

    ens = Ensemble(client=dask_client)
    ens.from_dataset("s82_rrlyrae")

    # larger dataset, let's just use a subset
    ens.prune(350)

    res = ens.batch(calc_stetson_J).compute()

    assert 377927 in res.index.values  # find a specific object

    # Check Stetson J results for a specific object
    assert res.loc[377927][0]["g"] == pytest.approx(9.676014, rel=0.001)
    assert res.loc[377927][0]["i"] == pytest.approx(14.22723, rel=0.001)
    assert res.loc[377927][0]["r"] == pytest.approx(6.958200, rel=0.001)
    assert res.loc[377927][0]["u"] == pytest.approx(9.499280, rel=0.001)
    assert res.loc[377927][0]["z"] == pytest.approx(14.03794, rel=0.001)


def test_from_qso_dataset(dask_client):
    """
    Test a basic load and analyze workflow from the S82 QSO Dataset
    """

    ens = Ensemble(client=dask_client)
    ens.from_dataset("s82_qso")

    # larger dataset, let's just use a subset
    ens.prune(650)

    res = ens.batch(calc_stetson_J).compute()

    assert 1257836 in res.index.values  # find a specific object

    # Check Stetson J results for a specific object
    assert res.loc[1257836][0]["g"] == pytest.approx(411.19885, rel=0.001)
    assert res.loc[1257836][0]["i"] == pytest.approx(86.371310, rel=0.001)
    assert res.loc[1257836][0]["r"] == pytest.approx(133.56796, rel=0.001)
    assert res.loc[1257836][0]["u"] == pytest.approx(231.93229, rel=0.001)
    assert res.loc[1257836][0]["z"] == pytest.approx(53.013018, rel=0.001)


def test_read_rrl_dataset(dask_client):
    """
    Test a basic load and analyze workflow from the S82 RR Lyrae Dataset
    """

    ens = tape.read_dataset("s82_rrlyrae", dask_client=dask_client)

    # larger dataset, let's just use a subset
    ens.prune(350)

    res = ens.batch(calc_stetson_J).compute()

    assert 377927 in res.index.values  # find a specific object

    # Check Stetson J results for a specific object
    assert res.loc[377927][0]["g"] == pytest.approx(9.676014, rel=0.001)
    assert res.loc[377927][0]["i"] == pytest.approx(14.22723, rel=0.001)
    assert res.loc[377927][0]["r"] == pytest.approx(6.958200, rel=0.001)
    assert res.loc[377927][0]["u"] == pytest.approx(9.499280, rel=0.001)
    assert res.loc[377927][0]["z"] == pytest.approx(14.03794, rel=0.001)


def test_read_qso_dataset(dask_client):
    """
    Test a basic load and analyze workflow from the S82 QSO Dataset
    """

    ens = tape.read_dataset("s82_qso", dask_client=dask_client)

    # larger dataset, let's just use a subset
    ens.prune(650)

    res = ens.batch(calc_stetson_J).compute()

    assert 1257836 in res.index.values  # find a specific object

    # Check Stetson J results for a specific object
    assert res.loc[1257836][0]["g"] == pytest.approx(411.19885, rel=0.001)
    assert res.loc[1257836][0]["i"] == pytest.approx(86.371310, rel=0.001)
    assert res.loc[1257836][0]["r"] == pytest.approx(133.56796, rel=0.001)
    assert res.loc[1257836][0]["u"] == pytest.approx(231.93229, rel=0.001)
    assert res.loc[1257836][0]["z"] == pytest.approx(53.013018, rel=0.001)


def test_from_source_dict(dask_client):
    """
    Test that ensemble.from_source_dict() successfully creates data from a dictionary.
    """
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps. Leave out the flux data initially.
    rows = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }

    # We get an error without all of the required rows.
    with pytest.raises(ValueError):
        ens.from_source_dict(rows)

    # Add the last row and build the ensemble.
    rows["flux"] = [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap)
    (obj_table, src_table) = ens.compute()

    # Check that the loaded source table is correct.
    assert src_table.shape[0] == 9
    for i in range(9):
        assert src_table.iloc[i][ens._flux_col] == rows[ens._flux_col][i]
        assert src_table.iloc[i][ens._time_col] == rows[ens._time_col][i]
        assert src_table.iloc[i][ens._band_col] == rows[ens._band_col][i]
        assert src_table.iloc[i][ens._err_col] == rows[ens._err_col][i]

    # Check that the derived object table is correct.
    assert 8001 in obj_table.index
    assert 8002 in obj_table.index


@pytest.mark.parametrize("bad_sort_kwargs", [True, False])
@pytest.mark.parametrize("use_object", [True, False])
@pytest.mark.parametrize("id_col", ["object_id", "_hipscat_index"])
def test_from_lsdb_warnings_errors(bad_sort_kwargs, use_object, id_col):
    """Test warnings in from_lsdb"""
    object_cat = lsdb.read_hipscat("tests/tape_tests/data/small_sky_hipscat/small_sky_object_catalog")
    source_cat = lsdb.read_hipscat("tests/tape_tests/data/small_sky_hipscat/small_sky_source_catalog")

    # Pain points: Suffixes here are a bit annoying, and I'd ideally want just the source columns (especially at scale)
    # We do this to get the source catalog indexed by the objects hipscat index
    joined_source_cat = object_cat.join(
        source_cat, left_on="id", right_on="object_id", suffixes=("_object", "")
    )

    colmap = ColumnMapper(
        id_col=id_col,
        time_col="mjd",
        flux_col="mag",
        err_col="Norder",  # no error column...
        band_col="band",
    )

    ens = Ensemble(client=False)

    # We just avoid needing to invoke the ._ddf property from the catalogs

    # When object and source are used with a id_col that is not _hipscat_index
    # Check to see if this gives the user the expected warning
    if id_col != "_hipscat_index" and use_object:
        # need to first rename
        object_cat._ddf = object_cat._ddf.rename(columns={"id": id_col})
        with pytest.warns(UserWarning):
            ens.from_lsdb(joined_source_cat, object_cat, column_mapper=colmap, sorted=False, sort=True)

    # When using just source and the _hipscat_index is chosen as the id_col
    # Check to see if this gives user the expected warning, do not test further
    # as this ensemble is incorrect (source _hipscat_index is unique per source)
    elif id_col == "_hipscat_index" and not use_object:
        with pytest.raises(ValueError):
            ens.from_lsdb(joined_source_cat, None, column_mapper=colmap, sorted=True, sort=False)

    # When using just source with bad sort kwargs, check that a warning is
    # raised, but this should still yield a valid result
    elif bad_sort_kwargs and not use_object:
        with pytest.warns(UserWarning):
            ens.from_lsdb(joined_source_cat, None, column_mapper=colmap, sorted=True, sort=False)

    else:
        return


@pytest.mark.parametrize("id_col", ["object_id", "_hipscat_index"])
def test_from_lsdb_no_object(id_col):
    """Ensemble from a hipscat directory, with just the source given"""
    source_cat = lsdb.read_hipscat("tests/tape_tests/data/small_sky_hipscat/small_sky_source_catalog")

    colmap = ColumnMapper(
        id_col=id_col,  # don't use _hipscat_index, it's per source
        time_col="mjd",
        flux_col="mag",
        err_col="Norder",  # no error column...
        band_col="band",
    )

    ens = Ensemble(client=False)

    # Just check to make sure users trying to use the _hipscat_index get an error
    # this ensemble is incorrect (one id per source)
    if id_col == "_hipscat_index":
        with pytest.raises(ValueError):
            ens.from_lsdb(source_cat, object_catalog=None, column_mapper=colmap, sorted=True, sort=False)
        return
    else:
        ens.from_lsdb(source_cat, object_catalog=None, column_mapper=colmap, sorted=False, sort=True)

    # Check to make sure the source and object tables were created
    assert ens.source is not None
    assert ens.object is not None

    # Make sure divisions are set
    assert ens.source.known_divisions
    assert ens.object.known_divisions

    # Check that the data is not empty.
    obj, source = ens.compute()
    assert len(source) == 17161
    assert len(obj) == 131

    # Check that source and object both have the same ids present
    assert sorted(np.unique(list(source.index))) == sorted(np.array(obj.index))

    # Check the we loaded the correct columns.
    for col in [
        ens._time_col,
        ens._flux_col,
        ens._err_col,
        ens._band_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert ens.source[col] is not None


def test_from_hipscat_no_object():
    """Ensemble from a hipscat directory, with just the source given"""
    ens = Ensemble(client=False)

    colmap = ColumnMapper(
        id_col="object_id",  # don't use _hipscat_index, it's per source
        time_col="mjd",
        flux_col="mag",
        err_col="Norder",  # no error column...
        band_col="band",
    )

    ens.from_hipscat(
        "tests/tape_tests/data/small_sky_hipscat/small_sky_source_catalog",
        object_path=None,
        column_mapper=colmap,
    )

    # Check to make sure the source and object tables were created
    assert ens.source is not None
    assert ens.object is not None

    # Make sure divisions are set
    assert ens.source.known_divisions
    assert ens.object.known_divisions

    # Check that the data is not empty.
    obj, source = ens.compute()
    assert len(source) == 17161
    assert len(obj) == 131

    # Check that source and object both have the same ids present
    assert sorted(np.unique(list(source.index))) == sorted(np.array(obj.index))

    # Check the we loaded the correct columns.
    for col in [
        ens._time_col,
        ens._flux_col,
        ens._err_col,
        ens._band_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert ens.source[col] is not None


def test_read_source_dict(dask_client):
    """
    Test that tape.read_source_dict() successfully creates data from a dictionary.
    """
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps. Leave out the flux data initially.
    rows = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }

    # We get an error without all of the required rows.
    with pytest.raises(ValueError):
        tape.read_source_dict(rows)

    # Add the last row and build the ensemble.
    rows["flux"] = [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")

    ens = tape.read_source_dict(rows, column_mapper=cmap, dask_client=dask_client)

    (obj_table, src_table) = ens.compute()

    # Check that the loaded source table is correct.
    assert src_table.shape[0] == 9
    for i in range(9):
        assert src_table.iloc[i][ens._flux_col] == rows[ens._flux_col][i]
        assert src_table.iloc[i][ens._time_col] == rows[ens._time_col][i]
        assert src_table.iloc[i][ens._band_col] == rows[ens._band_col][i]
        assert src_table.iloc[i][ens._err_col] == rows[ens._err_col][i]

    # Check that the derived object table is correct.
    assert 8001 in obj_table.index
    assert 8002 in obj_table.index


@pytest.mark.parametrize("add_frames", [True, False, ["max"], 42, ["max", "min"]])
@pytest.mark.parametrize("obj_nocols", [True, False])
@pytest.mark.parametrize("use_reader", [False, True])
def test_save_and_load_ensemble(dask_client, tmp_path, add_frames, obj_nocols, use_reader):
    """Test the save and load ensemble loop"""
    # Setup a temporary directory for files
    save_path = tmp_path / "."

    # Set a seed for reproducibility
    np.random.seed(1)

    # Create some toy data
    obj_ids = np.array([])
    mjds = np.array([])
    for i in range(10, 110):
        obj_ids = np.append(obj_ids, np.array([i] * 1250))
        mjds = np.append(mjds, np.arange(0.0, 1250.0, 1.0))
    obj_ids = np.array(obj_ids)
    flux = 10 * np.random.random(125000)
    err = flux / 10
    band = np.random.choice(["g", "r"], 125000)

    # Store it in a dictionary
    source_dict = {"id": obj_ids, "mjd": mjds, "flux": flux, "err": err, "band": band}

    # Create an Ensemble
    ens = Ensemble(client=dask_client)
    ens.from_source_dict(
        source_dict,
        column_mapper=ColumnMapper(
            id_col="id", time_col="mjd", flux_col="flux", err_col="err", band_col="band"
        ),
    )

    # object table as defined above has no columns, add a column to test both cases
    if not obj_nocols:
        # Make a column for the object table
        ens.calc_nobs(temporary=False)

    # Add a few result frames
    ens.batch(np.mean, "flux", label="mean")
    ens.batch(np.max, "flux", label="max")

    # Save the Ensemble
    if add_frames == 42 or add_frames == ["max", "min"]:
        with pytest.raises(ValueError):
            ens.save_ensemble(save_path, dirname="ensemble", additional_frames=add_frames)
        return
    else:
        ens.save_ensemble(save_path, dirname="ensemble", additional_frames=add_frames)
        # Inspect the save directory
        dircontents = os.listdir(os.path.join(save_path, "ensemble"))

        assert "source" in dircontents  # Source should always be there
        assert "ensemble_metadata.json" in dircontents  # should make a metadata file
        if obj_nocols:  # object shouldn't if it was empty
            assert "object" not in dircontents
        else:  # otherwise it should be present
            assert "object" in dircontents
        if add_frames is True:  # if additional_frames is true, mean and max should be there
            assert "max" in dircontents
            assert "mean" in dircontents
        elif add_frames is False:  # but they shouldn't be there if additional_frames is false
            assert "max" not in dircontents
            assert "mean" not in dircontents
        elif type(add_frames) == list:  # only max should be there if ["max"] is the input
            assert "max" in dircontents
            assert "mean" not in dircontents

    # Load a new Ensemble
    if not use_reader:
        loaded_ens = Ensemble(dask_client)
        loaded_ens.from_ensemble(os.path.join(save_path, "ensemble"), additional_frames=add_frames)
    else:
        loaded_ens = tape.ensemble_readers.read_ensemble(
            os.path.join(save_path, "ensemble"), additional_frames=add_frames, dask_client=dask_client
        )

    # compare object and source dataframes
    assert loaded_ens.source.compute().equals(ens.source.compute())
    assert loaded_ens.object.compute().equals(ens.object.compute())

    # Check the contents of the loaded ensemble
    if add_frames is True:  # if additional_frames is true, mean and max should be there
        assert "max" in loaded_ens.frames.keys()
        assert "mean" in loaded_ens.frames.keys()

        # Make sure the dataframes are identical
        assert loaded_ens.select_frame("max").compute().equals(ens.select_frame("max").compute())
        assert loaded_ens.select_frame("mean").compute().equals(ens.select_frame("mean").compute())

    elif add_frames is False:  # but they shouldn't be there if additional_frames is false
        assert "max" not in loaded_ens.frames.keys()
        assert "mean" not in loaded_ens.frames.keys()

    elif type(add_frames) == list:  # only max should be there if ["max"] is the input
        assert "max" in loaded_ens.frames.keys()
        assert "mean" not in loaded_ens.frames.keys()

        # Make sure the dataframes are identical
        assert loaded_ens.select_frame("max").compute().equals(ens.select_frame("max").compute())

    # Test a bad additional_frames call for the loader
    with pytest.raises(ValueError):
        bad_ens = Ensemble(dask_client)
        loaded_ens.from_ensemble(os.path.join(save_path, "ensemble"), additional_frames=3)


def test_save_overwrite(parquet_ensemble, tmp_path):
    """Test that successive saves produce the correct behavior"""
    # Setup a temporary directory for files
    save_path = tmp_path / "."

    ens = parquet_ensemble

    # Add a few result frames
    ens.batch(np.mean, "psFlux", label="mean")
    ens.batch(np.max, "psFlux", label="max")

    # Write first with all frames
    ens.save_ensemble(save_path, dirname="ensemble", additional_frames=True)

    # Inspect the save directory
    dircontents = os.listdir(os.path.join(save_path, "ensemble"))
    assert "max" in dircontents  # "max" should have been added

    # Then write again with "max" not included
    ens.save_ensemble(save_path, dirname="ensemble", additional_frames=["mean"])

    # Inspect the save directory
    dircontents = os.listdir(os.path.join(save_path, "ensemble"))
    assert "max" not in dircontents  # "max" should have been removed


def test_insert(parquet_ensemble):
    num_partitions = parquet_ensemble.source.npartitions
    (old_object, old_source) = parquet_ensemble.compute()
    old_size = old_source.shape[0]

    # Save the column names to shorter strings
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
    assert parquet_ensemble.source.npartitions == num_partitions

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
    assert parquet_ensemble.source.npartitions != num_partitions
    (new_obj, new_source) = parquet_ensemble.compute()
    assert new_source.shape[0] == old_size + 10


def test_insert_paritioned(dask_client):
    ens = Ensemble(client=dask_client)

    # Create all fake source data with known divisions.
    num_points = 1000
    all_bands = ["r", "g", "b", "i"]
    rows = {
        "id": [8000 + 2 * i for i in range(num_points)],
        "time": [float(i) for i in range(num_points)],
        "flux": [0.5 * float(i) for i in range(num_points)],
        "band": [all_bands[i % 4] for i in range(num_points)],
    }
    cmap = ColumnMapper(
        id_col="id",
        time_col="time",
        flux_col="flux",
        err_col="err",
        band_col="band",
    )
    ens.from_source_dict(rows, column_mapper=cmap, npartitions=4, sort=True)

    # Save the old data for comparison.
    old_data = ens.compute("source")
    old_div = copy.copy(ens.source.divisions)
    old_sizes = [len(ens.source.partitions[i]) for i in range(4)]
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
    assert ens.source.npartitions == 4
    assert ens.source.divisions == old_div
    assert len(ens.source.partitions[0]) == old_sizes[0] + 3
    assert len(ens.source.partitions[1]) == old_sizes[1]
    assert len(ens.source.partitions[2]) == old_sizes[2] + 2
    assert len(ens.source.partitions[3]) == old_sizes[3]

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
    assert ens.source.npartitions == 4
    assert ens.source.divisions == old_div
    assert len(ens.source.partitions[0]) == old_sizes[0] + 3
    assert len(ens.source.partitions[1]) == old_sizes[1] + 5
    assert len(ens.source.partitions[2]) == old_sizes[2] + 2
    assert len(ens.source.partitions[3]) == old_sizes[3]


def test_core_wrappers(parquet_ensemble):
    """
    Test that the core wrapper functions execute without errors
    """
    # Just test if these execute successfully
    parquet_ensemble.client_info()
    parquet_ensemble.info()
    parquet_ensemble.frame_info()
    with pytest.raises(KeyError):
        parquet_ensemble.frame_info(labels=["source", "invalid_label"])
    parquet_ensemble.columns()
    parquet_ensemble.head(n=5)
    parquet_ensemble.tail(n=5)
    parquet_ensemble.compute()


@pytest.mark.parametrize("data_sorted", [True, False])
@pytest.mark.parametrize("npartitions", [1, 2])
def test_check_sorted(dask_client, data_sorted, npartitions):
    # Create some fake data.

    if data_sorted:
        rows = {
            "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
            "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
            "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
            "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    else:
        rows = {
            "id": [8002, 8002, 8002, 8002, 8002, 8001, 8001, 8002, 8002],
            "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
            "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
            "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens = Ensemble(client=dask_client)
    ens.from_source_dict(rows, column_mapper=cmap, sort=False, npartitions=npartitions)

    assert ens.check_sorted("source") == data_sorted


@pytest.mark.parametrize("data_cohesion", [True, False])
def test_check_lightcurve_cohesion(dask_client, data_cohesion):
    # Create some fake data.

    if data_cohesion:
        rows = {
            "id": [8001, 8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002],
            "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
            "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
            "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    else:
        rows = {
            "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8001],
            "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
            "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
            "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens = Ensemble(client=dask_client)
    ens.from_source_dict(rows, column_mapper=cmap, sort=False, npartitions=2)

    assert ens.check_lightcurve_cohesion() == data_cohesion


def test_persist(dask_client):
    # Create some fake data.
    rows = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens = Ensemble(client=dask_client)
    ens.from_source_dict(rows, column_mapper=cmap)

    # Perform an operation that will build out the task graph.
    ens.query("flux <= 1.5", table="source")

    # Compute the task graph size before and after the persist.
    old_graph_size = len(ens.source.dask)
    ens.persist()
    new_graph_size = len(ens.source.dask)
    assert new_graph_size < old_graph_size


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble_with_divisions",
        "parquet_ensemble_with_client",
    ],
)
def test_sample(data_fixture, request):
    """
    Test Ensemble.sample
    """

    ens = request.getfixturevalue(data_fixture)
    ens.source.repartition(npartitions=10).update_ensemble()
    ens.object.repartition(npartitions=5).update_ensemble()

    prior_obj_len = len(ens.object)
    prior_src_len = len(ens.source)

    new_ens = ens.sample(frac=0.3)

    assert len(new_ens.object) < prior_obj_len  # frac is not exact
    assert len(new_ens.source) < prior_src_len  # should affect source

    # ens should not have been affected
    assert len(ens.object) == prior_obj_len
    assert len(ens.source) == prior_src_len


def test_update_column_map(dask_client):
    """
    Test that we can update the column maps in an Ensemble.
    """
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps. Leave out the flux data initially.
    rows = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [2.0, 1.0, 3.0, 5.0, 2.0, 1.0, 5.0, 4.0, 3.0],
        "p": [1, 1, 1, 2, 2, 2, 0, 0, 0],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap)

    # Check that we can retrieve the correct column map.
    cmap_1 = ens.make_column_map()
    assert cmap_1.map["id_col"] == "id"
    assert cmap_1.map["time_col"] == "time"
    assert cmap_1.map["flux_col"] == "flux"
    assert cmap_1.map["err_col"] == "err"
    assert cmap_1.map["band_col"] == "band"

    # Update the column map.
    ens.update_column_mapping(flux_col="f2")

    # Check that the flux column has been updated.
    assert ens._flux_col == "f2"

    # Check that we retrieve the updated column map.
    cmap_2 = ens.make_column_map()
    assert cmap_2.map["id_col"] == "id"
    assert cmap_2.map["time_col"] == "time"
    assert cmap_2.map["flux_col"] == "f2"
    assert cmap_2.map["err_col"] == "err"
    assert cmap_2.map["band_col"] == "band"


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
    ],
)
@pytest.mark.parametrize("legacy", [True, False])
def test_sync_tables(data_fixture, request, legacy):
    """
    Test that _sync_tables works as expected, using Ensemble-level APIs
    when `legacy` is `True`, and EsnembleFrame APIs when `legacy` is `False`.
    """
    parquet_ensemble = request.getfixturevalue(data_fixture)

    if legacy:
        assert len(parquet_ensemble.compute("object")) == 15
        assert len(parquet_ensemble.compute("source")) == 2000
    else:
        assert len(parquet_ensemble.object.compute()) == 15
        assert len(parquet_ensemble.source.compute()) == 2000

    parquet_ensemble.prune(50, col_name="nobs_r").prune(50, col_name="nobs_g")
    assert parquet_ensemble.object.is_dirty()  # Prune should set the object dirty flag

    if legacy:
        assert len(parquet_ensemble.compute("object")) == 5
    else:
        assert len(parquet_ensemble.object.compute()) == 5

    if legacy:
        parquet_ensemble.dropna(table="source")
    else:
        parquet_ensemble.source.dropna().update_ensemble()
    assert parquet_ensemble.source.is_dirty()  # Dropna should set the source dirty flag

    # Drop a whole object from Source to test that the object is dropped in the object table
    dropped_obj_id = 88472935274829959
    if legacy:
        parquet_ensemble.query(f"{parquet_ensemble._id_col} != {dropped_obj_id}", table="source")
    else:
        filtered_src = parquet_ensemble.source.query(f"{parquet_ensemble._id_col} != 88472935274829959")

        # Since we have not yet called update_ensemble, the compute call should not trigger
        # a sync and the source table should remain dirty.
        assert parquet_ensemble.source.is_dirty()
        filtered_src.compute()
        assert parquet_ensemble.source.is_dirty()

        # Update the ensemble to use the filtered source.
        filtered_src.update_ensemble()

    # Verify that the object ID we removed from the source table is present in the object table
    assert dropped_obj_id in parquet_ensemble.object.index.compute().values

    # Perform an operation which should trigger syncing both tables.
    parquet_ensemble.compute()

    # Both tables should have the expected number of rows after a sync
    if legacy:
        assert len(parquet_ensemble.compute("object")) == 4
        assert len(parquet_ensemble.compute("source")) == 1063
    else:
        assert len(parquet_ensemble.object.compute()) == 4
        assert len(parquet_ensemble.source.compute()) == 1063

    # Validate that the filtered object has been removed from both tables.
    assert dropped_obj_id not in parquet_ensemble.source.index.compute().values
    assert dropped_obj_id not in parquet_ensemble.object.index.compute().values

    # Dirty flags should be unset after sync
    assert not parquet_ensemble.object.is_dirty()
    assert not parquet_ensemble.source.is_dirty()

    # Make sure that divisions are preserved
    if data_fixture == "parquet_ensemble_with_divisions":
        assert parquet_ensemble.source.known_divisions
        assert parquet_ensemble.object.known_divisions


@pytest.mark.parametrize("legacy", [True, False])
def test_lazy_sync_tables(parquet_ensemble, legacy):
    """
    Test that _lazy_sync_tables works as expected, using Ensemble-level APIs
    when `legacy` is `True`, and EsnembleFrame APIs when `legacy` is `False`.
    """
    if legacy:
        assert len(parquet_ensemble.compute("object")) == 15
        assert len(parquet_ensemble.compute("source")) == 2000
    else:
        assert len(parquet_ensemble.object.compute()) == 15
        assert len(parquet_ensemble.source.compute()) == 2000

    # Modify only the object table.
    parquet_ensemble.prune(50, col_name="nobs_r").prune(50, col_name="nobs_g")
    assert parquet_ensemble.object.is_dirty()
    assert not parquet_ensemble.source.is_dirty()

    # For a lazy sync on the object table, nothing should change, because
    # it is already dirty.
    if legacy:
        parquet_ensemble.compute("object")
    else:
        parquet_ensemble.object.compute()
    assert parquet_ensemble.object.is_dirty()
    assert not parquet_ensemble.source.is_dirty()

    # For a lazy sync on the source table, the source table should be updated.
    if legacy:
        parquet_ensemble.compute("source")
    else:
        parquet_ensemble.source.compute()
    assert not parquet_ensemble.object.is_dirty()
    assert not parquet_ensemble.source.is_dirty()

    # Modify only the source table.
    # Replace the maximum flux value with a NaN so that we will have a row to drop.
    max_flux = max(parquet_ensemble.source[parquet_ensemble._flux_col])
    parquet_ensemble.source[parquet_ensemble._flux_col] = parquet_ensemble.source[
        parquet_ensemble._flux_col
    ].apply(lambda x: np.nan if x == max_flux else x, meta=pd.Series(dtype=float))

    assert not parquet_ensemble.object.is_dirty()
    assert not parquet_ensemble.source.is_dirty()

    if legacy:
        parquet_ensemble.dropna(table="source")
    else:
        parquet_ensemble.source.dropna().update_ensemble()
    assert not parquet_ensemble.object.is_dirty()
    assert parquet_ensemble.source.is_dirty()

    # For a lazy sync on the source table, nothing should change, because
    # it is already dirty.
    if legacy:
        parquet_ensemble.compute("source")
    else:
        parquet_ensemble.source.compute()
    assert not parquet_ensemble.object.is_dirty()
    assert parquet_ensemble.source.is_dirty()

    # For a lazy sync on the source, the object table should be updated.
    if legacy:
        parquet_ensemble.compute("object")
    else:
        parquet_ensemble.object.compute()
    assert not parquet_ensemble.object.is_dirty()
    assert not parquet_ensemble.source.is_dirty()


def test_compute_triggers_syncing(parquet_ensemble):
    """
    Tests that tape.EnsembleFrame.compute() only triggers an Ensemble sync if the
    frame is the actively tracked source or object table of the Ensemble.
    """
    # Test that an object table can trigger a sync that will clean a dirty
    # source table.
    parquet_ensemble.source.set_dirty(True)
    updated_obj = parquet_ensemble.object.dropna()

    # Because we have not yet called update_ensemble(), a sync is not triggered
    # and the source table remains dirty.
    updated_obj.compute()
    assert parquet_ensemble.source.is_dirty()

    # Update the Ensemble so that computing the object table will trigger
    # a sync
    updated_obj.update_ensemble()
    updated_obj.compute()  # Now equivalent to Ensemble.object.compute()
    assert not parquet_ensemble.source.is_dirty()

    # Test that an source table can trigger a sync that will clean a dirty
    # object table.
    parquet_ensemble.object.set_dirty(True)
    updated_src = parquet_ensemble.source.dropna()

    # Because we have not yet called update_ensemble(), a sync is not triggered
    # and the object table remains dirty.
    updated_src.compute()
    assert parquet_ensemble.object.is_dirty()

    # Update the Ensemble so that computing the object table will trigger
    # a sync
    updated_src.update_ensemble()
    updated_src.compute()  # Now equivalent to Ensemble.source.compute()
    assert not parquet_ensemble.object.is_dirty()

    # Generate a new Object frame and set the Ensemble to None to
    # validate that we return a valid result even for untracked frames
    # which cannot be synced.
    new_obj_frame = parquet_ensemble.object.dropna()
    new_obj_frame.ensemble = None
    assert len(new_obj_frame.compute()) > 0


def test_temporary_cols(parquet_ensemble):
    """
    Test that temporary columns are tracked and dropped as expected.
    """

    ens = parquet_ensemble
    ens.update_frame(ens.object.drop(columns=["nobs_r", "nobs_g", "nobs_total"]))

    # Make sure temp lists are available but empty
    assert not len(ens._source_temp)
    assert not len(ens._object_temp)

    ens.calc_nobs(temporary=True)  # Generates "nobs_total"

    # nobs_total should be a temporary column
    assert "nobs_total" in ens._object_temp
    assert "nobs_total" in ens.object.columns

    ens.assign(nobs2=lambda x: x["nobs_total"] * 2, table="object", temporary=True)

    # nobs2 should be a temporary column
    assert "nobs2" in ens._object_temp
    assert "nobs2" in ens.object.columns

    # drop NaNs from source, source should be dirty now
    ens.dropna(how="any", table="source")

    assert ens.source.is_dirty()

    # try a sync
    ens._sync_tables()

    # nobs_total should be removed from object
    assert "nobs_total" not in ens._object_temp
    assert "nobs_total" not in ens.object.columns

    # nobs2 should be removed from object
    assert "nobs2" not in ens._object_temp
    assert "nobs2" not in ens.object.columns

    # add a source column that we manually set as dirty, don't have a function
    # that adds temporary source columns at the moment
    ens.assign(f2=lambda x: x[ens._flux_col] ** 2, table="source", temporary=True)

    # prune object, object should be dirty
    ens.prune(threshold=10)

    assert ens.object.is_dirty()

    # try a sync
    ens._sync_tables()

    # f2 should be removed from source
    assert "f2" not in ens._source_temp
    assert "f2" not in ens.source.columns


def test_temporary_cols(parquet_ensemble):
    """
    Test that temporary columns are tracked and dropped as expected.
    """

    ens = parquet_ensemble
    ens.object = ens.object.drop(columns=["nobs_r", "nobs_g", "nobs_total"])

    # Make sure temp lists are available but empty
    assert not len(ens._source_temp)
    assert not len(ens._object_temp)

    ens.calc_nobs(temporary=True)  # Generates "nobs_total"

    # nobs_total should be a temporary column
    assert "nobs_total" in ens._object_temp
    assert "nobs_total" in ens.object.columns

    ens.assign(nobs2=lambda x: x["nobs_total"] * 2, table="object", temporary=True)

    # nobs2 should be a temporary column
    assert "nobs2" in ens._object_temp
    assert "nobs2" in ens.object.columns

    # Replace the maximum flux value with a NaN so that we will have a row to drop.
    max_flux = max(parquet_ensemble.source[parquet_ensemble._flux_col])
    parquet_ensemble.source[parquet_ensemble._flux_col] = parquet_ensemble.source[
        parquet_ensemble._flux_col
    ].apply(lambda x: np.nan if x == max_flux else x, meta=pd.Series(dtype=float))

    # drop NaNs from source, source should be dirty now
    ens.dropna(how="any", table="source")

    assert ens.source.is_dirty()

    # try a sync
    ens._sync_tables()

    # nobs_total should be removed from object
    assert "nobs_total" not in ens._object_temp
    assert "nobs_total" not in ens.object.columns

    # nobs2 should be removed from object
    assert "nobs2" not in ens._object_temp
    assert "nobs2" not in ens.object.columns

    # add a source column that we manually set as dirty, don't have a function
    # that adds temporary source columns at the moment
    ens.assign(f2=lambda x: x[ens._flux_col] ** 2, table="source", temporary=True)

    # prune object, object should be dirty
    ens.prune(threshold=10)

    assert ens.object.is_dirty()

    # try a sync
    ens._sync_tables()

    # f2 should be removed from source
    assert "f2" not in ens._source_temp
    assert "f2" not in ens.source.columns


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
    ],
)
@pytest.mark.parametrize("legacy", [True, False])
def test_dropna(data_fixture, request, legacy):
    """Tests dropna, using Ensemble.dropna when `legacy` is `True`, and
    EnsembleFrame.dropna when `legacy` is `False`."""
    parquet_ensemble = request.getfixturevalue(data_fixture)

    # Try passing in an unrecognized 'table' parameter and verify an exception is thrown
    with pytest.raises(ValueError):
        parquet_ensemble.dropna(table="banana")

    # First test dropping na from the 'source' table
    source_pdf = parquet_ensemble.source.compute()
    source_length = len(source_pdf.index)

    # Try dropping NaNs from source and confirm nothing is dropped (there are no NaNs).
    if legacy:
        parquet_ensemble.dropna(table="source")
    else:
        parquet_ensemble.source.dropna().update_ensemble()
    assert len(parquet_ensemble.source) == source_length

    # Get a valid ID to use and count its occurrences.
    valid_source_id = source_pdf.index.values[1]
    occurrences_source = len(source_pdf.loc[valid_source_id].values)

    # Set the psFlux values for one source to NaN so we can drop it.
    # We do this on the instantiated source (pdf) and convert it back into a
    # SourceFrame.
    source_pdf.loc[valid_source_id, parquet_ensemble._flux_col] = pd.NA
    parquet_ensemble.update_frame(
        SourceFrame.from_tapeframe(TapeSourceFrame(source_pdf), label="source", npartitions=1)
    )

    # Try dropping NaNs from source and confirm that we did.
    if legacy:
        parquet_ensemble.dropna(table="source")
    else:
        parquet_ensemble.source.dropna().update_ensemble()
    assert len(parquet_ensemble.source.compute().index) == source_length - occurrences_source

    if data_fixture == "parquet_ensemble_with_divisions":
        # divisions should be preserved
        assert parquet_ensemble.source.known_divisions

    # Now test dropping na from 'object' table
    # Sync the tables
    parquet_ensemble._sync_tables()

    # Sync (triggered by the compute) the table and check that the number of objects decreased.
    object_pdf = parquet_ensemble.object.compute()
    object_length = len(object_pdf.index)

    # Try dropping NaNs from object and confirm nothing is dropped (there are no NaNs).
    if legacy:
        parquet_ensemble.dropna(table="object")
    else:
        parquet_ensemble.object.dropna().update_ensemble()
    assert len(parquet_ensemble.object.compute().index) == object_length

    # select an id from the object table
    valid_object_id = object_pdf.index.values[1]

    # Set the nobs_g values for one object to NaN so we can drop it.
    # We do this on the instantiated object (pdf) and convert it back into a
    # ObjectFrame.
    object_pdf.loc[valid_object_id, parquet_ensemble.object.columns[0]] = pd.NA
    parquet_ensemble.update_frame(
        ObjectFrame.from_tapeframe(TapeObjectFrame(object_pdf), label="object", npartitions=1)
    )

    # Try dropping NaNs from object and confirm that we dropped a row
    if legacy:
        parquet_ensemble.dropna(table="object")
    else:
        parquet_ensemble.object.dropna().update_ensemble()
    assert len(parquet_ensemble.object.compute().index) == object_length - 1

    if data_fixture == "parquet_ensemble_with_divisions":
        # divisions should be preserved
        assert parquet_ensemble.object.known_divisions

    new_objects_pdf = parquet_ensemble.object.compute()
    assert len(new_objects_pdf.index) == len(object_pdf.index) - 1

    # Assert the filtered ID is no longer in the objects.
    assert valid_source_id not in new_objects_pdf.index.values

    # Check that none of the other counts have changed.
    for i in new_objects_pdf.index.values:
        for c in new_objects_pdf.columns.values:
            assert new_objects_pdf.loc[i, c] == object_pdf.loc[i, c]


@pytest.mark.parametrize("legacy", [True, False])
def test_keep_zeros(parquet_ensemble, legacy):
    """Test that we can sync the tables and keep objects with zero sources, using
    Ensemble.dropna when `legacy` is `True`, and EnsembleFrame.dropna when `legacy` is `False`."""
    parquet_ensemble.keep_empty_objects = True

    prev_npartitions = parquet_ensemble.object.npartitions
    old_objects_pdf = parquet_ensemble.object.compute()
    pdf = parquet_ensemble.source.compute()

    # Set the psFlux values for one object to NaN so we can drop it.
    # We do this on the instantiated object (pdf) and convert it back into a
    # Dask DataFrame.
    valid_id = pdf.index.values[1]
    pdf.loc[valid_id, parquet_ensemble._flux_col] = pd.NA
    parquet_ensemble.source = dd.from_pandas(pdf, npartitions=1)
    parquet_ensemble.update_frame(
        SourceFrame.from_tapeframe(TapeSourceFrame(pdf), npartitions=1, label="source")
    )

    # Sync the table and check that the number of objects decreased.
    if legacy:
        parquet_ensemble.dropna("source")
    else:
        parquet_ensemble.source.dropna().update_ensemble()
    parquet_ensemble._sync_tables()

    # Check that objects are preserved after sync
    new_objects_pdf = parquet_ensemble.object.compute()
    assert len(new_objects_pdf.index) == len(old_objects_pdf.index)
    assert parquet_ensemble.object.npartitions == prev_npartitions


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
    ],
)
@pytest.mark.parametrize("by_band", [True, False])
@pytest.mark.parametrize("multi_partition", [True, False])
def test_calc_nobs(data_fixture, request, by_band, multi_partition):
    # Get the Ensemble from a fixture
    ens = request.getfixturevalue(data_fixture)

    if multi_partition:
        ens.source = ens.source.repartition(3)

    # Drop the existing nobs columns
    ens.object = ens.object.drop(["nobs_g", "nobs_r", "nobs_total"], axis=1)

    # Make sure syncs work, do novel query to remove some sources
    ens._lazy_sync_tables("all")  # do an initial sync to clear state
    ens.source.query("index != 88472468910699998").update_ensemble()
    assert ens.source.is_dirty()

    # Calculate nobs
    ens.calc_nobs(by_band)

    # Check to make sure a sync was performed
    assert not ens.source.is_dirty()

    # Check that things turned out as we expect
    lc = ens.object.loc[88472935274829959].compute()

    if by_band:
        assert np.all([col in ens.object.columns for col in ["nobs_g", "nobs_r"]])
        assert lc["nobs_g"].values[0] == 98
        assert lc["nobs_r"].values[0] == 401

    assert "nobs_total" in ens.object.columns
    assert lc["nobs_total"].values[0] == 499

    # Make sure that if divisions were set previously, they are preserved
    if data_fixture == "parquet_ensemble_with_divisions":
        assert ens.object.known_divisions
        assert ens.source.known_divisions


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
    ],
)
@pytest.mark.parametrize("generate_nobs", [False, True])
def test_prune(data_fixture, request, generate_nobs):
    """
    Test that ensemble.prune() appropriately filters the dataframe
    """

    # Get the Ensemble from a fixture
    parquet_ensemble = request.getfixturevalue(data_fixture)

    threshold = 10
    # Generate the nobs cols from within prune
    if generate_nobs:
        # Drop the existing nobs columns
        parquet_ensemble.object = parquet_ensemble.object.drop(["nobs_g", "nobs_r", "nobs_total"], axis=1)
        parquet_ensemble.prune(threshold)

    # Use an existing column
    else:
        parquet_ensemble.prune(threshold, col_name="nobs_total")

    assert not np.any(parquet_ensemble.object["nobs_total"].values < threshold)

    # Make sure that if divisions were set previously, they are preserved
    if data_fixture == "parquet_ensemble_with_divisions":
        assert parquet_ensemble.source.known_divisions
        assert parquet_ensemble.object.known_divisions


def test_query(dask_client):
    ens = Ensemble(client=dask_client)

    num_points = 1000
    all_bands = ["r", "g", "b", "i"]
    rows = {
        "id": [8000 + 2 * i for i in range(num_points)],
        "time": [float(i) for i in range(num_points)],
        "flux": [float(i % 4) for i in range(num_points)],
        "band": [all_bands[i % 4] for i in range(num_points)],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap, npartitions=2)

    # Filter the data set to low flux sources only.
    ens.query("flux <= 1.5", table="source")

    # Check that all of the filtered rows are value.
    (new_obj, new_source) = ens.compute()
    assert new_source.shape[0] == 500
    for i in range(500):
        assert new_source.iloc[i][ens._flux_col] <= 1.5


def test_filter_from_series(dask_client):
    ens = Ensemble(client=dask_client)

    num_points = 1000
    all_bands = ["r", "g", "b", "i"]
    rows = {
        "id": [8000 + 2 * i for i in range(num_points)],
        "time": [float(i) for i in range(num_points)],
        "flux": [0.5 * float(i % 4) for i in range(num_points)],
        "band": [all_bands[i % 4] for i in range(num_points)],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap, npartitions=2)

    # Filter the data set to low flux sources only.
    keep_series = ens.source[ens._time_col] >= 250.0
    ens.filter_from_series(keep_series, table="source")

    # Check that all of the filtered rows are value.
    (new_obj, new_source) = ens.compute()
    assert new_source.shape[0] == 750
    for i in range(750):
        assert new_source.iloc[i][ens._time_col] >= 250.0


def test_select(dask_client):
    ens = Ensemble(client=dask_client)

    num_points = 1000
    all_bands = ["r", "g", "b", "i"]
    rows = {
        "id": [8000 + (i % 5) for i in range(num_points)],
        "time": [float(i) for i in range(num_points)],
        "flux": [float(i % 4) for i in range(num_points)],
        "band": [all_bands[i % 4] for i in range(num_points)],
        "count": [i for i in range(num_points)],
        "something_else": [None for _ in range(num_points)],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap, npartitions=2)
    assert len(ens.source.columns) == 5
    assert "time" in ens.source.columns
    assert "flux" in ens.source.columns
    assert "band" in ens.source.columns
    assert "count" in ens.source.columns
    assert "something_else" in ens.source.columns

    # Select on just time and flux
    ens.select(["time", "flux"], table="source")

    assert len(ens.source.columns) == 2
    assert "time" in ens.source.columns
    assert "flux" in ens.source.columns
    assert "band" not in ens.source.columns
    assert "count" not in ens.source.columns
    assert "something_else" not in ens.source.columns


@pytest.mark.parametrize("legacy", [True, False])
def test_assign(dask_client, legacy):
    """Tests assign for column-manipulation, using Ensemble.assign when `legacy` is `True`,
    and EnsembleFrame.assign when `legacy` is `False`."""
    ens = Ensemble(client=dask_client)

    num_points = 1000
    all_bands = ["r", "g", "b", "i"]
    rows = {
        "id": [8000 + (i % 10) for i in range(num_points)],
        "time": [float(i) for i in range(num_points)],
        "flux": [float(i % 4) for i in range(num_points)],
        "band": [all_bands[i % 4] for i in range(num_points)],
        "err": [0.1 * float((i + 2) % 5) for i in range(num_points)],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap, npartitions=1)
    assert len(ens.source.columns) == 4
    assert "lower_bnd" not in ens.source.columns

    # Insert a new column for the "lower bound" computation.
    if legacy:
        ens.assign(table="source", lower_bnd=lambda x: x["flux"] - 2.0 * x["err"])
    else:
        ens.source.assign(lower_bnd=lambda x: x["flux"] - 2.0 * x["err"]).update_ensemble()
    assert len(ens.source.columns) == 5
    assert "lower_bnd" in ens.source.columns

    # Check the values in the new column.
    new_source = ens.source.compute() if not legacy else ens.compute(table="source")
    assert new_source.shape[0] == 1000
    for i in range(1000):
        expected = new_source.iloc[i]["flux"] - 2.0 * new_source.iloc[i]["err"]
        assert new_source.iloc[i]["lower_bnd"] == expected

    # Create a series directly from the table.
    res_col = ens.source["band"] + "2"
    if legacy:
        ens.assign(table="source", band2=res_col)
    else:
        ens.source.assign(band2=res_col).update_ensemble()
    assert len(ens.source.columns) == 6
    assert "band2" in ens.source.columns

    # Check the values in the new column.
    new_source = ens.source.compute() if not legacy else ens.compute(table="source")
    for i in range(1000):
        assert new_source.iloc[i]["band2"] == new_source.iloc[i]["band"] + "2"


@pytest.mark.parametrize("zero_point", [("zp_mag", "zp_flux"), (25.0, 10**10)])
@pytest.mark.parametrize("zp_form", ["flux", "mag", "magnitude", "lincc"])
@pytest.mark.parametrize("out_col_name", [None, "mag"])
def test_convert_flux_to_mag(dask_client, zero_point, zp_form, out_col_name):
    ens = Ensemble(client=dask_client)

    source_dict = {
        "id": [0, 0, 0, 0, 0],
        "time": [1, 2, 3, 4, 5],
        "flux": [30.5, 70, 80.6, 30.2, 60.3],
        "zp_mag": [25.0, 25.0, 25.0, 25.0, 25.0],
        "zp_flux": [10**10, 10**10, 10**10, 10**10, 10**10],
        "error": [10, 10, 10, 10, 10],
        "band": ["g", "g", "g", "g", "g"],
    }

    if out_col_name is None:
        output_column = "flux_mag"
    else:
        output_column = out_col_name

    # map flux_col to one of the flux columns at the start
    col_map = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="error", band_col="band")
    ens.from_source_dict(source_dict, column_mapper=col_map)

    if zp_form == "flux":
        ens.convert_flux_to_mag(zero_point[1], zp_form, out_col_name)

        res_mag = ens.source.compute()[output_column].to_list()[0]
        assert pytest.approx(res_mag, 0.001) == 21.28925

        res_err = ens.source.compute()[output_column + "_err"].to_list()[0]
        assert pytest.approx(res_err, 0.001) == 0.355979

    elif zp_form == "mag" or zp_form == "magnitude":
        ens.convert_flux_to_mag(zero_point[0], zp_form, out_col_name)

        res_mag = ens.source.compute()[output_column].to_list()[0]
        assert pytest.approx(res_mag, 0.001) == 21.28925

        res_err = ens.source.compute()[output_column + "_err"].to_list()[0]
        assert pytest.approx(res_err, 0.001) == 0.355979

    else:
        with pytest.raises(ValueError):
            ens.convert_flux_to_mag(zero_point[0], zp_form, "mag")


def test_find_day_gap_offset(dask_client):
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps.
    rows = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 10.9, 11.1, 15.0, 15.1],
        "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }

    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap)
    gap_time = ens.find_day_gap_offset()
    assert abs(gap_time - 13.0 / 24.0) < 1e-6

    # Create fake observations covering all times
    rows = {
        "id": [8001] * 100,
        "time": [24.0 * (float(i) / 100.0) for i in range(100)],
        "flux": [1.0] * 100,
        "band": ["g"] * 100,
    }

    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap)
    assert ens.find_day_gap_offset() == -1


def test_bin_sources_day(dask_client):
    ens = Ensemble(client=dask_client)

    # Create some fake data with two IDs (8001, 8002), two bands ["g", "b"]
    # and a few time steps.
    rows = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 10.9, 11.1, 15.0, 15.1],
        "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }

    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap)

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
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 10.9, 11.1, 15.0, 15.1, 14.0],
        "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0],
    }

    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap)

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


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
        "parquet_ensemble_with_client",
    ],
)
@pytest.mark.parametrize("use_map", [True, False])
@pytest.mark.parametrize("on", [None, ["ps1_objid", "filterName"], ["nobs_total", "ps1_objid"]])
def test_batch(data_fixture, request, use_map, on):
    """
    Test that ensemble.batch() returns the correct values of the first result
    """
    parquet_ensemble = request.getfixturevalue(data_fixture)
    frame_cnt = len(parquet_ensemble.frames)

    result = (
        parquet_ensemble.prune(10)
        .dropna(table="source")
        .batch(calc_stetson_J, use_map=use_map, on=on, band_to_calc=None, label="stetson_j")
    )

    # Validate that the ensemble is now tracking a new result frame.
    assert len(parquet_ensemble.frames) == frame_cnt + 1
    tracked_result = parquet_ensemble.select_frame("stetson_j")

    print(tracked_result)
    assert isinstance(tracked_result, EnsembleFrame)
    assert result is tracked_result

    # Make sure that divisions information is propagated if known
    if parquet_ensemble.source.known_divisions and parquet_ensemble.object.known_divisions:
        assert result.known_divisions

    result = result.compute()

    if on is None:
        print(result.values[0])
        assert pytest.approx(result.values[0][0]["g"], 0.001) == -0.04174282
        assert pytest.approx(result.values[0][0]["r"], 0.001) == 0.6075282
    elif on is ["ps1_objid", "filterName"]:  # In case where we group on id and band, the structure changes
        assert pytest.approx(result.values[1]["r"], 0.001) == 0.6075282
        assert pytest.approx(result.values[0]["g"], 0.001) == -0.04174282
    elif on is ["nobs_total", "ps1_objid"]:
        assert pytest.approx(result.values[1]["g"], 0.001) == 1.2208577
        assert pytest.approx(result.values[1]["r"], 0.001) == -0.49639028


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
    ],
)
@pytest.mark.parametrize("sort_by_band", [True, False])
def test_sort_lightcurves(data_fixture, request, sort_by_band):
    """
    Test that we can have the ensemble sort its lightcurves by timestamp.
    """
    parquet_ensemble = request.getfixturevalue(data_fixture)

    # filter NaNs from the source table
    parquet_ensemble = parquet_ensemble.prune(10).dropna(table="source")

    # To check that all columns are rearranged when sorting the time column,
    # we create a duplicate time column which should be sorted as well.
    parquet_ensemble.source.assign(
        dup_time=parquet_ensemble.source[parquet_ensemble._time_col]
    ).update_ensemble()

    # Validate the Ensemble is sorted by ID
    assert parquet_ensemble.check_sorted("source")

    bands = parquet_ensemble.source[parquet_ensemble._band_col].unique().compute()

    # A trivial function that raises an Exception if the data is not temporally sorted
    def my_mean(flux, time, dup_time, band):
        if not sort_by_band:
            # Check that the time column is sorted
            if not np.all(time[:-1] <= time[1:]):
                raise ValueError("The time column was not sorted in ascending order")
        else:
            # Check that the band column is sorted
            if not np.all(band[:-1] <= band[1:]):
                raise ValueError("The bands column was not sorted in ascending order")
            # Check that the time column is sorted for each band
            for curr_band in bands:
                # Get a mask for the current band
                mask = band == curr_band
                if not np.all(time[mask][:-1] <= time[mask][1:]):
                    raise ValueError(f"The time column was not sorted in ascending order for band {band}")
        # Check that the other columns were rearranged to preserve the dataframe's rows
        # We can use the duplicate time column as an easy check.
        if not np.array_equal(time, dup_time):
            raise ValueError("The dataframe's time column was sorted but isn't aligned with other columns")
        return np.mean(flux)

    band = parquet_ensemble._band_col if sort_by_band else None

    # Validate that our custom function throws an Exception on the unsorted data to
    # ensure that we actually sort when requested.
    with pytest.raises(ValueError):
        parquet_ensemble.batch(
            my_mean,
            parquet_ensemble._flux_col,
            parquet_ensemble._time_col,
            "dup_time",
            parquet_ensemble._band_col,
            by_band=False,
        ).compute()

    parquet_ensemble.sort_lightcurves(by_band=sort_by_band)

    result = parquet_ensemble.batch(
        my_mean,
        parquet_ensemble._flux_col,
        parquet_ensemble._time_col,
        "dup_time",
        parquet_ensemble._band_col,
        by_band=False,
    )

    # Validate that the result is non-empty
    assert len(result.compute()) > 0

    # Make sure that divisions information was propagated if known
    if parquet_ensemble.source.known_divisions and parquet_ensemble.object.known_divisions:
        assert result.known_divisions

    # Check that the dataframe is still sorted by the ID column
    assert parquet_ensemble.check_sorted("source")

    # Verify that we preserved lightcurve cohesion
    assert parquet_ensemble.check_lightcurve_cohesion()


@pytest.mark.parametrize("on", [None, ["ps1_objid", "filterName"], ["filterName", "ps1_objid"]])
@pytest.mark.parametrize("func_label", ["mean", "bounds"])
def test_batch_by_band(parquet_ensemble, func_label, on):
    """
    Test that ensemble.batch(by_band=True) works as intended.
    """

    if func_label == "mean":

        def my_mean(flux):
            """returns a single value"""
            return np.mean(flux)

        res = parquet_ensemble.batch(my_mean, parquet_ensemble._flux_col, on=on, by_band=True)

        parquet_ensemble.source.query(f"{parquet_ensemble._band_col}=='g'").update_ensemble()
        filter_res = parquet_ensemble.batch(my_mean, parquet_ensemble._flux_col, on=on, by_band=False)

        # An EnsembleFrame should be returned
        assert isinstance(res, EnsembleFrame)

        # Make sure we get all the expected columns
        assert all([col in res.columns for col in ["result_g", "result_r"]])

        # These should be equivalent
        assert (
            res.loc[88472935274829959]["result_g"]
            .compute()
            .equals(filter_res.loc[88472935274829959]["result"].compute())
        )

    elif func_label == "bounds":

        def my_bounds(flux):
            """returns a series"""
            return pd.Series({"min": np.min(flux), "max": np.max(flux)})

        res = parquet_ensemble.batch(
            my_bounds, "psFlux", on=on, by_band=True, meta={"min": float, "max": float}
        )

        parquet_ensemble.source.query(f"{parquet_ensemble._band_col}=='g'").update_ensemble()
        filter_res = parquet_ensemble.batch(
            my_bounds, "psFlux", on=on, by_band=False, meta={"min": float, "max": float}
        )

        # An EnsembleFrame should be returned
        assert isinstance(res, EnsembleFrame)

        # Make sure we get all the expected columns
        assert all([col in res.columns for col in ["max_g", "max_r", "min_g", "min_r"]])

        # These should be equivalent
        assert (
            res.loc[88472935274829959]["max_g"]
            .compute()
            .equals(filter_res.loc[88472935274829959]["max"].compute())
        )
        assert (
            res.loc[88472935274829959]["min_g"]
            .compute()
            .equals(filter_res.loc[88472935274829959]["min"].compute())
        )

    # Meta should reflect the actual columns, this can get out of sync
    # whenever multi-indexes are involved, which batch tries to handle
    assert all([col in res.columns for col in res.compute().columns])


def test_batch_labels(parquet_ensemble):
    """
    Test that ensemble.batch() generates unique labels for result frames when none are provided.
    """
    # Since no label was provided we generate a label of "result_1"
    parquet_ensemble.prune(10).batch(np.mean, parquet_ensemble._flux_col)
    assert "result_1" in parquet_ensemble.frames
    assert len(parquet_ensemble.select_frame("result_1")) > 0

    # Now give a user-provided custom label.
    parquet_ensemble.batch(np.mean, parquet_ensemble._flux_col, label="flux_mean")
    assert "flux_mean" in parquet_ensemble.frames
    assert len(parquet_ensemble.select_frame("flux_mean")) > 0

    # Since this is the second batch call where a label is *not* provided, we generate label "result_2"
    parquet_ensemble.batch(np.mean, parquet_ensemble._flux_col)
    assert "result_2" in parquet_ensemble.frames
    assert len(parquet_ensemble.select_frame("result_2")) > 0

    # Explicitly provide label "result_3"
    parquet_ensemble.batch(np.mean, parquet_ensemble._flux_col, label="result_3")
    assert "result_3" in parquet_ensemble.frames
    assert len(parquet_ensemble.select_frame("result_3")) > 0

    # Validate that the next generated label is "result_4" since "result_3" is taken.
    parquet_ensemble.batch(np.mean, parquet_ensemble._flux_col)
    assert "result_4" in parquet_ensemble.frames
    assert len(parquet_ensemble.select_frame("result_4")) > 0

    frame_cnt = len(parquet_ensemble.frames)

    # Validate that when the label is None, the result frame isn't tracked by the Ensemble.s
    result = parquet_ensemble.batch(np.mean, parquet_ensemble._flux_col, label=None)
    assert frame_cnt == len(parquet_ensemble.frames)
    assert len(result) > 0


def test_batch_with_custom_func(parquet_ensemble):
    """
    Test Ensemble.batch with a custom analysis function
    """

    result = parquet_ensemble.prune(10).batch(np.mean, parquet_ensemble._flux_col)
    assert len(result) > 0


@pytest.mark.parametrize(
    "custom_meta",
    [
        ("flux_mean", float),  # A tuple representing a series
        pd.Series(name="flux_mean_pandas", dtype="float64"),
        TapeSeries(name="flux_mean_tape", dtype="float64"),
    ],
)
def test_batch_with_custom_series_meta(parquet_ensemble, custom_meta):
    """
    Test Ensemble.batch with various styles of output meta for a Series-style result.
    """
    num_frames = len(parquet_ensemble.frames)

    parquet_ensemble.prune(10).batch(np.mean, parquet_ensemble._flux_col, meta=custom_meta, label="flux_mean")

    assert len(parquet_ensemble.frames) == num_frames + 1
    assert len(parquet_ensemble.select_frame("flux_mean")) > 0
    assert isinstance(parquet_ensemble.select_frame("flux_mean"), EnsembleFrame)


@pytest.mark.parametrize(
    "custom_meta",
    [
        {"lc_id": int, "band": str, "dt": float, "sf2": float, "1_sigma": float},
        [("lc_id", int), ("band", str), ("dt", float), ("sf2", float), ("1_sigma", float)],
        pd.DataFrame(
            {
                "lc_id": pd.Series([], dtype=int),
                "band": pd.Series([], dtype=str),
                "dt": pd.Series([], dtype=float),
                "sf2": pd.Series([], dtype=float),
                "1_sigma": pd.Series([], dtype=float),
            }
        ),
        TapeFrame(
            {
                "lc_id": pd.Series([], dtype=int),
                "band": pd.Series([], dtype=str),
                "dt": pd.Series([], dtype=float),
                "sf2": pd.Series([], dtype=float),
                "1_sigma": pd.Series([], dtype=float),
            }
        ),
    ],
)
def test_batch_with_custom_frame_meta(parquet_ensemble, custom_meta):
    """
    Test Ensemble.batch with various sytles of output meta for a DataFrame-style result.
    """
    num_frames = len(parquet_ensemble.frames)

    parquet_ensemble.prune(10).batch(
        calc_sf2, parquet_ensemble._flux_col, meta=custom_meta, label="sf2_result"
    )

    assert len(parquet_ensemble.frames) == num_frames + 1
    assert len(parquet_ensemble.select_frame("sf2_result")) > 0
    assert isinstance(parquet_ensemble.select_frame("sf2_result"), EnsembleFrame)


@pytest.mark.parametrize("repartition", [False, True])
@pytest.mark.parametrize("seed", [None, 42])
def test_select_random_timeseries(parquet_ensemble, repartition, seed):
    """Test the behavior of ensemble.select_random_timeseries"""

    ens = parquet_ensemble

    if repartition:
        ens.object = ens.object.repartition(3)

    ts = ens.select_random_timeseries(seed=seed)

    assert isinstance(ts, TimeSeries)

    if seed == 42 and not repartition:
        assert ts.meta["id"] == 88472935274829959
    elif seed == 42 and repartition:
        assert ts.meta["id"] == 88480001333818899


@pytest.mark.parametrize("all_empty", [False, True])
def test_select_random_timeseries_empty_partitions(dask_client, all_empty):
    "Test the edge case where object has empty partitions"

    data_dict = {
        "id": [42],
        "flux": [1],
        "time": [1],
        "err": [1],
        "band": [1],
    }

    colmap = ColumnMapper().assign(
        id_col="id",
        time_col="time",
        flux_col="flux",
        err_col="err",
        band_col="band",
    )

    ens = Ensemble(client=dask_client)
    ens.from_source_dict(data_dict, column_mapper=colmap)

    # The single id will be in the last partition
    ens.object = ens.object.repartition(5)

    # Remove the last partition, make sure we get the expected error when the
    # Object table has no IDs in any partition
    if all_empty:
        ens.object = ens.object.partitions[0:-1]
        with pytest.raises(IndexError):
            ens.select_random_timeseries()
    else:
        ts = ens.select_random_timeseries()
        assert ts.meta["id"] == 42  # Should always find the only object


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


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_ensemble",
        "parquet_ensemble_with_divisions",
    ],
)
@pytest.mark.parametrize("method", ["size", "length", "loglength"])
@pytest.mark.parametrize("combine", [True, False])
@pytest.mark.parametrize("sthresh", [50, 100])
def test_sf2(data_fixture, request, method, combine, sthresh, use_map=False):
    """
    Test calling sf2 from the ensemble
    """
    parquet_ensemble = request.getfixturevalue(data_fixture)

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = method
    arg_container.combine = combine
    arg_container.bin_count_target = sthresh

    if not combine:
        res_sf2 = parquet_ensemble.sf2(argument_container=arg_container, use_map=use_map)
    else:
        res_sf2 = parquet_ensemble.sf2(argument_container=arg_container, use_map=use_map)
    res_batch = parquet_ensemble.batch(calc_sf2, use_map=use_map, argument_container=arg_container)

    if parquet_ensemble.source.known_divisions and parquet_ensemble.object.known_divisions:
        if not combine:
            assert res_sf2.known_divisions

    if combine:
        assert not res_sf2.equals(res_batch.compute())  # output should be different
    else:
        assert res_sf2.compute().equals(res_batch.compute())  # output should be identical


@pytest.mark.parametrize("sf_method", ["basic", "macleod_2012", "bauer_2009a", "bauer_2009b", "schmidt_2010"])
def test_sf2_methods(parquet_ensemble, sf_method, use_map=False):
    """
    Test calling sf2 from the ensemble
    """

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "loglength"
    arg_container.combine = False
    arg_container.bin_count_target = 50
    arg_container.sf_method = sf_method

    res_sf2 = parquet_ensemble.sf2(argument_container=arg_container, use_map=use_map).compute()
    res_batch = parquet_ensemble.batch(calc_sf2, use_map=use_map, argument_container=arg_container).compute()

    assert res_sf2.equals(res_batch)  # output should be identical
