"""Test ensemble manipulations"""
import copy

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from tape import Ensemble, ObjectFrame, SourceFrame, TapeObjectFrame, TapeSourceFrame
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
        "parquet_ensemble_without_client",
        "parquet_ensemble_from_source",
        "parquet_ensemble_from_hipscat",
        "parquet_ensemble_with_column_mapper",
        "parquet_ensemble_with_known_column_mapper",
    ],
)
def test_from_parquet(data_fixture, request):
    """
    Test that ensemble loader functions successfully load parquet files
    """
    parquet_ensemble = request.getfixturevalue(data_fixture)

    # Check to make sure the source and object tables were created
    assert parquet_ensemble._source is not None
    assert parquet_ensemble._object is not None

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
        parquet_ensemble._provenance_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert parquet_ensemble._source[col] is not None

 
@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_files_and_ensemble_without_client",
    ],
)
def test_objsor_from_parquet(data_fixture, request):
    """
    Test that the ensemble successfully loads a SourceFrame and ObjectFrame form parquet files.
    """
    _, source_file, object_file, colmap = request.getfixturevalue(data_fixture)

    ens = Ensemble(client=False)
    ens = ens.objsor_from_parquet(source_file, object_file, column_mapper=colmap)

    assert ens is not None

    # Check to make sure the source and object tables were created
    assert ens.source is not None
    assert ens.object is not None
    assert isinstance(ens.source, SourceFrame)
    assert isinstance(ens.object, ObjectFrame)

    # Check that the data is not empty.
    obj, source = ens.compute()
    assert len(source) == 2000
    assert len(obj) == 15

    # Check that source and object both have the same ids present
    assert sorted(np.unique(list(source.index))) == sorted(np.array(obj.index))

    # Check the we loaded the correct columns.
    for col in [
        ens._time_col,
        ens._flux_col,
        ens._err_col,
        ens._band_col,
        ens._provenance_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert ens.source[col] is not None


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
        "parquet_files_and_ensemble_without_client",
    ],
)
def test_frame_tracking(data_fixture, request):
    """
    Tests a workflow of adding and removing the frames tracked by the Ensemble.
    """
    ens, source_file, object_file, colmap = request.getfixturevalue(data_fixture)

    ens = ens.objsor_from_parquet(source_file, object_file, column_mapper=colmap)

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
    ens_frame1 = ens.select_frame("source").copy()
    ens_frame2 = ens.select_frame("source").copy()
    ens_frame3 = ens.select_frame("source").copy()
    ens_frame4 = ens.select_frame("source").copy()

    # Labels to give the EnsembleFrames
    label1, label2, label3, label4 = "frame1", "frame2", "frame3", "frame4"

    # Validate that new source and object frames can't be added or updated.
    with pytest.raises(ValueError):
        ens.add_frame(ens_frame1, "source")
    with pytest.raises(ValueError):
        ens.add_frame(ens_frame1, "object")
    with pytest.raises(ValueError):
        ens.update_frame(ens.source)
    with pytest.raises(ValueError):
        ens.update_frame(ens.object)

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

    # Update the ensemble with a new frame, verifying a missing label generates an error.
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

    # larger dataset, let's just use a subset of ~100
    ens.prune(350)

    res = ens.batch(calc_stetson_J)

    assert 377927 in res.index  # find a specific object

    # Check Stetson J results for a specific object
    assert res[377927]["g"] == pytest.approx(9.676014, rel=0.001)
    assert res[377927]["i"] == pytest.approx(14.22723, rel=0.001)
    assert res[377927]["r"] == pytest.approx(6.958200, rel=0.001)
    assert res[377927]["u"] == pytest.approx(9.499280, rel=0.001)
    assert res[377927]["z"] == pytest.approx(14.03794, rel=0.001)


def test_from_qso_dataset(dask_client):
    """
    Test a basic load and analyze workflow from the S82 QSO Dataset
    """

    ens = Ensemble(client=dask_client)
    ens.from_dataset("s82_qso")

    # larger dataset, let's just use a subset of ~100
    ens.prune(650)

    res = ens.batch(calc_stetson_J)

    assert 1257836 in res  # find a specific object

    # Check Stetson J results for a specific object
    assert res.loc[1257836]["g"] == pytest.approx(411.19885, rel=0.001)
    assert res.loc[1257836]["i"] == pytest.approx(86.371310, rel=0.001)
    assert res.loc[1257836]["r"] == pytest.approx(133.56796, rel=0.001)
    assert res.loc[1257836]["u"] == pytest.approx(231.93229, rel=0.001)
    assert res.loc[1257836]["z"] == pytest.approx(53.013018, rel=0.001)


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
    assert obj_table.shape[0] == 2
    assert obj_table.iloc[0][ens._nobs_tot_col] == 4
    assert obj_table.iloc[1][ens._nobs_tot_col] == 5


def test_insert(parquet_ensemble):
    num_partitions = parquet_ensemble._source.npartitions
    (old_object, old_source) = parquet_ensemble.compute()
    old_size = old_source.shape[0]

    # Save the column names to shorter strings
    time_col = parquet_ensemble._time_col
    flux_col = parquet_ensemble._flux_col
    err_col = parquet_ensemble._err_col
    band_col = parquet_ensemble._band_col
    prov_col = parquet_ensemble._provenance_col

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
        assert new_source.loc[new_inds[i]][prov_col] == "custom"

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
        "id": [8000 + 2 * i for i in range(num_points)],
        "time": [float(i) for i in range(num_points)],
        "flux": [0.5 * float(i) for i in range(num_points)],
        "band": [all_bands[i % 4] for i in range(num_points)],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens.from_source_dict(rows, column_mapper=cmap, npartitions=4)

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
    parquet_ensemble.frame_info()
    with pytest.raises(KeyError):
        parquet_ensemble.frame_info(labels=["source", "invalid_label"])
    parquet_ensemble.columns()
    parquet_ensemble.head(n=5)
    parquet_ensemble.tail(n=5)
    parquet_ensemble.compute()


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
    old_graph_size = len(ens._source.dask)
    ens.persist()
    new_graph_size = len(ens._source.dask)
    assert new_graph_size < old_graph_size


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
    assert cmap_1.map["provenance_col"] is None

    # Update the column map.
    ens.update_column_mapping(flux_col="f2", provenance_col="p")

    # Check that the flux and provenance columns have been updates.
    assert ens._flux_col == "f2"
    assert ens._provenance_col == "p"

    # Check that we retrieve the updated column map.
    cmap_2 = ens.make_column_map()
    assert cmap_2.map["id_col"] == "id"
    assert cmap_2.map["time_col"] == "time"
    assert cmap_2.map["flux_col"] == "f2"
    assert cmap_2.map["err_col"] == "err"
    assert cmap_2.map["band_col"] == "band"
    assert cmap_2.map["provenance_col"] == "p"


def test_sync_tables(parquet_ensemble):
    """
    Test that _sync_tables works as expected
    """

    assert len(parquet_ensemble.compute("object")) == 15
    assert len(parquet_ensemble.compute("source")) == 2000

    parquet_ensemble.prune(50, col_name="nobs_r").prune(50, col_name="nobs_g")
    assert parquet_ensemble._object.is_dirty()  # Prune should set the object dirty flag

    parquet_ensemble.dropna(table="source")
    assert parquet_ensemble._source.is_dirty()  # Dropna should set the source dirty flag

    parquet_ensemble._sync_tables()

    # both tables should have the expected number of rows after a sync
    assert len(parquet_ensemble.compute("object")) == 5
    assert len(parquet_ensemble.compute("source")) == 1562

    # dirty flags should be unset after sync
    assert not parquet_ensemble._object.is_dirty()
    assert not parquet_ensemble._source.is_dirty()


def test_lazy_sync_tables(parquet_ensemble):
    """
    Test that _lazy_sync_tables works as expected
    """
    assert len(parquet_ensemble.compute("object")) == 15
    assert len(parquet_ensemble.compute("source")) == 2000

    # Modify only the object table.
    parquet_ensemble.prune(50, col_name="nobs_r").prune(50, col_name="nobs_g")
    assert parquet_ensemble._object.is_dirty()
    assert not parquet_ensemble._source.is_dirty()

    # For a lazy sync on the object table, nothing should change, because
    # it is already dirty.
    parquet_ensemble._lazy_sync_tables(table="object")
    assert parquet_ensemble._object.is_dirty()
    assert not parquet_ensemble._source.is_dirty()

    # For a lazy sync on the source table, the source table should be updated.
    parquet_ensemble._lazy_sync_tables(table="source")
    assert not parquet_ensemble._object.is_dirty()
    assert not parquet_ensemble._source.is_dirty()

    # Modify only the source table.
    parquet_ensemble.dropna(table="source")
    assert not parquet_ensemble._object.is_dirty()
    assert parquet_ensemble._source.is_dirty()

    # For a lazy sync on the source table, nothing should change, because
    # it is already dirty.
    parquet_ensemble._lazy_sync_tables(table="source")
    assert not parquet_ensemble._object.is_dirty()
    assert parquet_ensemble._source.is_dirty()

    # For a lazy sync on the source, the object table should be updated.
    parquet_ensemble._lazy_sync_tables(table="object")
    assert not parquet_ensemble._object.is_dirty()
    assert not parquet_ensemble._source.is_dirty()


def test_dropna(parquet_ensemble):
    # Try passing in an unrecognized 'table' parameter and verify an exception is thrown
    with pytest.raises(ValueError):
        parquet_ensemble.dropna(table="banana")

    # First test dropping na from the 'source' table
    #
    source_pdf = parquet_ensemble._source.compute()
    source_length = len(source_pdf.index)

    # Try dropping NaNs from source and confirm nothing is dropped (there are no NaNs).
    parquet_ensemble.dropna(table="source")
    assert len(parquet_ensemble._source.compute().index) == source_length

    # Get a valid ID to use and count its occurrences.
    valid_source_id = source_pdf.index.values[1]
    occurrences_source = len(source_pdf.loc[valid_source_id].values)

    # Set the psFlux values for one source to NaN so we can drop it.
    # We do this on the instantiated source (pdf) and convert it back into a
    # SourceFrame.
    source_pdf.loc[valid_source_id, parquet_ensemble._flux_col] = pd.NA
    parquet_ensemble.update_frame(SourceFrame.from_tapeframe(TapeSourceFrame(source_pdf), label="source", npartitions=1))

    # Try dropping NaNs from source and confirm that we did.
    parquet_ensemble.dropna(table="source")
    assert len(parquet_ensemble._source.compute().index) == source_length - occurrences_source

    # Sync the table and check that the number of objects decreased.
    # parquet_ensemble._sync_tables()

    # Now test dropping na from 'object' table
    #
    object_pdf = parquet_ensemble._object.compute()
    object_length = len(object_pdf.index)

    # Try dropping NaNs from object and confirm nothing is dropped (there are no NaNs).
    parquet_ensemble.dropna(table="object")
    assert len(parquet_ensemble._object.compute().index) == object_length

    # get a valid object id and set at least two occurences of that id in the object table
    valid_object_id = object_pdf.index.values[1]
    object_pdf.index.values[0] = valid_object_id
    occurrences_object = len(object_pdf.loc[valid_object_id].values)

    # Set the nobs_g values for one object to NaN so we can drop it.
    # We do this on the instantiated object (pdf) and convert it back into a
    # ObjectFrame.
    object_pdf.loc[valid_object_id, parquet_ensemble._object.columns[0]] = pd.NA
    parquet_ensemble.udpate_frame(ObjectFrame.from_tapeframe(TapeObjectFrame(object_pdf), label="object", npartitions=1))

    # Try dropping NaNs from object and confirm that we did.
    parquet_ensemble.dropna(table="object")
    assert len(parquet_ensemble._object.compute().index) == object_length - occurrences_object

    new_objects_pdf = parquet_ensemble._object.compute()
    assert len(new_objects_pdf.index) == len(object_pdf.index) - occurrences_object

    # Assert the filtered ID is no longer in the objects.
    assert valid_source_id not in new_objects_pdf.index.values

    # Check that none of the other counts have changed.
    for i in new_objects_pdf.index.values:
        for c in new_objects_pdf.columns.values:
            assert new_objects_pdf.loc[i, c] == object_pdf.loc[i, c]


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
    parquet_ensemble.update_frame(SourceFrame.from_tapeframe(TapeSourceFrame(pdf), npartitions=1, label="source"))

    # Sync the table and check that the number of objects decreased.
    parquet_ensemble.dropna(table="source")
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
    keep_series = ens._source[ens._time_col] >= 250.0
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
    assert len(ens._source.columns) == 5
    assert "time" in ens._source.columns
    assert "flux" in ens._source.columns
    assert "band" in ens._source.columns
    assert "count" in ens._source.columns
    assert "something_else" in ens._source.columns

    # Select on just time and flux
    ens.select(["time", "flux"], table="source")

    assert len(ens._source.columns) == 2
    assert "time" in ens._source.columns
    assert "flux" in ens._source.columns
    assert "band" not in ens._source.columns
    assert "count" not in ens._source.columns
    assert "something_else" not in ens._source.columns


def test_assign(dask_client):
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
    assert len(ens._source.columns) == 4
    assert "lower_bnd" not in ens._source.columns

    # Insert a new column for the "lower bound" computation.
    ens.assign(table="source", lower_bnd=lambda x: x["flux"] - 2.0 * x["err"])
    assert len(ens._source.columns) == 5
    assert "lower_bnd" in ens._source.columns

    # Check the values in the new column.
    new_source = ens.compute(table="source")
    assert new_source.shape[0] == 1000
    for i in range(1000):
        expected = new_source.iloc[i]["flux"] - 2.0 * new_source.iloc[i]["err"]
        assert new_source.iloc[i]["lower_bnd"] == expected

    # Create a series directly from the table.
    res_col = ens._source["band"] + "2"
    ens.assign(table="source", band2=res_col)
    assert len(ens._source.columns) == 6
    assert "band2" in ens._source.columns

    # Check the values in the new column.
    new_source = ens.compute(table="source")
    for i in range(1000):
        assert new_source.iloc[i]["band2"] == new_source.iloc[i]["band"] + "2"


@pytest.mark.parametrize("drop_inputs", [True, False])
def test_coalesce(dask_client, drop_inputs):
    ens = Ensemble(client=dask_client)

    # Generate some data that needs to be coalesced

    source_dict = {
        "id": [0, 0, 0, 0, 0],
        "time": [1, 2, 3, 4, 5],
        "flux1": [5, np.nan, np.nan, 10, np.nan],
        "flux2": [np.nan, 3, np.nan, np.nan, 7],
        "flux3": [np.nan, np.nan, 4, np.nan, np.nan],
        "error": [1, 1, 1, 1, 1],
        "band": ["g", "g", "g", "g", "g"],
    }

    # map flux_col to one of the flux columns at the start
    col_map = ColumnMapper(id_col="id", time_col="time", flux_col="flux1", err_col="error", band_col="band")
    ens.from_source_dict(source_dict, column_mapper=col_map)

    ens.coalesce(["flux1", "flux2", "flux3"], "flux", table="source", drop_inputs=drop_inputs)

    # Coalesce should return this exact flux array
    assert list(ens._source["flux"].values.compute()) == [5.0, 3.0, 4.0, 10.0, 7.0]

    if drop_inputs:
        # The column mapping should be updated
        assert ens.make_column_map().map["flux_col"] == "flux"

        # The columns to drop should be dropped
        for col in ["flux1", "flux2", "flux3"]:
            assert col not in ens._source.columns

        # Test for the drop warning
        with pytest.warns(UserWarning):
            ens.coalesce(["time", "flux"], "bad_col", table="source", drop_inputs=drop_inputs)

    else:
        # The input columns should still be present
        for col in ["flux1", "flux2", "flux3"]:
            assert col in ens._source.columns

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
        "parquet_ensemble_without_client",
    ],
)
@pytest.mark.parametrize("use_map", [True, False])
@pytest.mark.parametrize("on", [None, ["ps1_objid", "filterName"], ["nobs_total", "ps1_objid"]])
def test_batch(data_fixture, request, use_map, on):
    """
    Test that ensemble.batch() returns the correct values of the first result
    """

    parquet_ensemble = request.getfixturevalue(data_fixture)

    result = (
        parquet_ensemble.prune(10)
        .dropna(table="source")
        .batch(calc_stetson_J, use_map=use_map, on=on, band_to_calc=None)
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


def test_batch_with_custom_func(parquet_ensemble):
    """
    Test Ensemble.batch with a custom analysis function
    """

    result = parquet_ensemble.prune(10).batch(np.mean, parquet_ensemble._flux_col)
    assert len(result) > 0


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

    res_sf2 = parquet_ensemble.sf2(argument_container=arg_container, use_map=use_map)
    res_batch = parquet_ensemble.batch(calc_sf2, use_map=use_map, argument_container=arg_container)

    assert res_sf2.equals(res_batch)  # output should be identical
