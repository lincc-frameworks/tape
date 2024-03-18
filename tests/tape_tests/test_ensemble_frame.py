""" Test EnsembleFrame (inherited from Dask.DataFrame) creation and manipulations. """

import numpy as np
import pandas as pd
from tape import (
    Ensemble,
    ColumnMapper,
    EnsembleFrame,
    ObjectFrame,
    SourceFrame,
    TapeObjectFrame,
    TapeSourceFrame,
    TapeFrame,
)

import pytest

TEST_LABEL = "test_frame"
SOURCE_LABEL = "source"
OBJECT_LABEL = "object"


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "data_fixture",
    [
        "ensemble_from_source_dict",
    ],
)
def test_from_dict(data_fixture, request):
    """
    Test creating an EnsembleFrame from a dictionary and verify that dask lazy evaluation was appropriately inherited.
    """
    _, data = request.getfixturevalue(data_fixture)
    ens_frame = EnsembleFrame.from_dict(data, npartitions=1)

    assert isinstance(ens_frame, EnsembleFrame)
    assert isinstance(ens_frame._meta, TapeFrame)

    # The calculation for finding the max flux from the data. Note that the
    # inherited dask compute method must be called to obtain the result.
    assert ens_frame.flux.max().compute() == 80.6


@pytest.mark.parametrize(
    "data_fixture",
    [
        "ensemble_from_source_dict",
    ],
)
def test_from_pandas(data_fixture, request):
    """
    Test creating an EnsembleFrame from a Pandas dataframe and verify that dask lazy evaluation was appropriately inherited.
    """
    ens, data = request.getfixturevalue(data_fixture)
    frame = TapeFrame(data)
    ens_frame = EnsembleFrame.from_tapeframe(frame, label=TEST_LABEL, ensemble=ens, npartitions=1)

    assert isinstance(ens_frame, EnsembleFrame)
    assert isinstance(ens_frame._meta, TapeFrame)
    assert ens_frame.label == TEST_LABEL
    assert ens_frame.ensemble is ens

    # The calculation for finding the max flux from the data. Note that the
    # inherited dask compute method must be called to obtain the result.
    assert ens_frame.flux.max().compute() == 80.6


def test_from_parquet():
    """
    Test creating an EnsembleFrame from a parquet file.
    """
    frame = EnsembleFrame.from_parquet(
        "tests/tape_tests/data/source/test_source.parquet", label=TEST_LABEL, ensemble=None
    )
    assert isinstance(frame, EnsembleFrame)
    assert isinstance(frame._meta, TapeFrame)
    assert frame.label == TEST_LABEL
    assert frame.ensemble is None

    # Validate that we loaded a non-empty frame.
    assert len(frame) > 0


@pytest.mark.parametrize(
    "data_fixture",
    [
        "ensemble_from_source_dict",
    ],
)
def test_ensemble_frame_propagation(data_fixture, request):
    """
    Test ensuring that slices and copies of an EnsembleFrame or still the same class.
    """
    ens, data = request.getfixturevalue(data_fixture)
    ens_frame = EnsembleFrame.from_dict(data, npartitions=1)
    # Set a label and ensemble for the frame and copies/transformations retain them.
    ens_frame.label = TEST_LABEL
    ens_frame.ensemble = ens
    assert not ens_frame.is_dirty()
    ens_frame.set_dirty(True)

    # Create a copy of an EnsembleFrame and verify that it's still a proper
    # EnsembleFrame with appropriate metadata propagated.
    copied_frame = ens_frame.copy()
    assert isinstance(copied_frame, EnsembleFrame)
    assert isinstance(copied_frame._meta, TapeFrame)
    assert copied_frame.label == TEST_LABEL
    assert copied_frame.ensemble == ens
    assert copied_frame.is_dirty()

    # Verify that the above is also true by calling copy via map_partitions
    mapped_frame = ens_frame.copy().map_partitions(lambda x: x.copy())
    assert isinstance(mapped_frame, EnsembleFrame)
    assert isinstance(mapped_frame._meta, TapeFrame)
    assert mapped_frame.label == TEST_LABEL
    assert mapped_frame.ensemble == ens
    assert mapped_frame.is_dirty()

    # Test that a filtered EnsembleFrame is still an EnsembleFrame.
    filtered_frame = ens_frame[["id", "time"]]
    assert isinstance(filtered_frame, EnsembleFrame)
    assert isinstance(filtered_frame._meta, TapeFrame)
    assert filtered_frame.label == TEST_LABEL
    assert filtered_frame.ensemble == ens
    assert filtered_frame.is_dirty()

    # Test that the output of an EnsembleFrame query is still an EnsembleFrame
    queried_rows = ens_frame.query("flux > 3.0")
    assert isinstance(queried_rows, EnsembleFrame)
    assert isinstance(queried_rows._meta, TapeFrame)
    assert queried_rows.label == TEST_LABEL
    assert queried_rows.ensemble == ens
    assert queried_rows.is_dirty()

    # Test merging two subsets of the dataframe, dropping some columns, and persisting the result.
    merged_frame = ens_frame.copy()[["id", "time", "error"]].merge(
        ens_frame.copy()[["id", "time", "flux"]], on=["id"], suffixes=(None, "_drop_me")
    )
    cols_to_drop = [col for col in merged_frame.columns if "_drop_me" in col]
    merged_frame = merged_frame.drop(cols_to_drop, axis=1).persist()
    assert isinstance(merged_frame, EnsembleFrame)
    assert merged_frame.label == TEST_LABEL
    assert merged_frame.ensemble == ens
    assert merged_frame.is_dirty()

    # Test that frame metadata is preserved after repartitioning
    repartitioned_frame = ens_frame.copy().repartition(npartitions=10)
    assert isinstance(repartitioned_frame, EnsembleFrame)
    assert repartitioned_frame.label == TEST_LABEL
    assert repartitioned_frame.ensemble == ens
    assert repartitioned_frame.is_dirty()

    # Test that head returns a subset of the underlying TapeFrame.
    h = ens_frame.head(5)
    assert isinstance(h, TapeFrame)
    assert len(h) == 5

    # Test that the inherited dask.DataFrame.compute method returns
    # the underlying TapeFrame.
    assert isinstance(ens_frame.compute(), TapeFrame)
    assert len(ens_frame) == len(ens_frame.compute())

    # Set an index and then group by that index.
    ens_frame = ens_frame.set_index("id", drop=True)
    assert ens_frame.label == TEST_LABEL
    assert ens_frame.ensemble == ens
    group_result = ens_frame.groupby(["id"]).count()
    assert len(group_result) > 0
    assert isinstance(group_result, EnsembleFrame)
    assert isinstance(group_result._meta, TapeFrame)


@pytest.mark.parametrize(
    "data_fixture",
    [
        "ensemble_from_source_dict",
    ],
)
@pytest.mark.parametrize("err_col", [None, "error"])
@pytest.mark.parametrize("zp_form", ["flux", "mag", "magnitude", "lincc"])
@pytest.mark.parametrize("out_col_name", [None, "mag"])
def test_convert_flux_to_mag(data_fixture, request, err_col, zp_form, out_col_name):
    ens, data = request.getfixturevalue(data_fixture)

    if out_col_name is None:
        output_column = "flux_mag"
    else:
        output_column = out_col_name

    ens_frame = EnsembleFrame.from_dict(data, npartitions=1)
    ens_frame.label = TEST_LABEL
    ens_frame.ensemble = ens

    if zp_form == "flux":
        ens_frame = ens_frame.convert_flux_to_mag("flux", "zp_flux", err_col, zp_form, out_col_name)

        res_mag = ens_frame.compute()[output_column].to_list()[0]
        assert pytest.approx(res_mag, 0.001) == 21.28925

        if err_col is not None:
            res_err = ens_frame.compute()[output_column + "_err"].to_list()[0]
            assert pytest.approx(res_err, 0.001) == 0.355979
        else:
            assert output_column + "_err" not in ens_frame.columns

    elif zp_form == "mag" or zp_form == "magnitude":
        ens_frame = ens_frame.convert_flux_to_mag("flux", "zp_mag", err_col, zp_form, out_col_name)

        res_mag = ens_frame.compute()[output_column].to_list()[0]
        assert pytest.approx(res_mag, 0.001) == 21.28925

        if err_col is not None:
            res_err = ens_frame.compute()[output_column + "_err"].to_list()[0]
            assert pytest.approx(res_err, 0.001) == 0.355979
        else:
            assert output_column + "_err" not in ens_frame.columns

    else:
        with pytest.raises(ValueError):
            ens_frame = ens_frame.convert_flux_to_mag("flux", "zp_mag", err_col, zp_form, "mag")

    # Verify that if we converted to a new frame, it's still an EnsembleFrame.
    assert isinstance(ens_frame, EnsembleFrame)
    assert ens_frame.label == TEST_LABEL
    assert ens_frame.ensemble is ens


@pytest.mark.parametrize(
    "data_fixture",
    [
        "parquet_files_and_ensemble_with_client",
    ],
)
def test_object_and_source_frame_propagation(data_fixture, request):
    """
    Test that SourceFrame and ObjectFrame metadata and class type is correctly preserved across
    typical Pandas operations.
    """
    ens, source_file, object_file, _ = request.getfixturevalue(data_fixture)

    assert ens is not None

    # Create a SourceFrame from a parquet file
    source_frame = SourceFrame.from_parquet(source_file, ensemble=ens)

    assert isinstance(source_frame, EnsembleFrame)
    assert isinstance(source_frame, SourceFrame)
    assert isinstance(source_frame._meta, TapeSourceFrame)

    assert source_frame.ensemble is not None
    assert source_frame.ensemble == ens
    assert source_frame.ensemble is ens

    assert not source_frame.is_dirty()
    source_frame.set_dirty(True)

    # Perform a series of operations on the SourceFrame and then verify the result is still a
    # proper SourceFrame with appropriate metadata propagated.
    source_frame["psFlux"].mean().compute()
    result_source_frame = source_frame.copy()[["psFlux", "psFluxErr"]]
    result_source_frame = result_source_frame.map_partitions(lambda x: x.copy())
    assert isinstance(result_source_frame, SourceFrame)
    assert isinstance(result_source_frame._meta, TapeSourceFrame)
    assert len(result_source_frame) > 0
    assert result_source_frame.label == SOURCE_LABEL
    assert result_source_frame.ensemble is not None
    assert result_source_frame.ensemble is ens
    assert result_source_frame.is_dirty()

    # Mark the frame clean to verify that we propagate that state as well
    result_source_frame.set_dirty(False)

    # Set an index and then group by that index.
    result_source_frame = result_source_frame.set_index("psFlux", drop=True)
    assert result_source_frame.label == SOURCE_LABEL
    assert result_source_frame.ensemble == ens
    assert not result_source_frame.is_dirty()  # frame is still clean.
    group_result = result_source_frame.groupby(["psFlux"]).count()
    assert len(group_result) > 0
    assert isinstance(group_result, SourceFrame)
    assert isinstance(group_result._meta, TapeSourceFrame)

    # Create an ObjectFrame from a parquet file
    object_frame = ObjectFrame.from_parquet(
        object_file,
        ensemble=ens,
        index="ps1_objid",
    )

    assert isinstance(object_frame, EnsembleFrame)
    assert isinstance(object_frame, ObjectFrame)
    assert isinstance(object_frame._meta, TapeObjectFrame)

    assert not object_frame.is_dirty()
    object_frame.set_dirty(True)
    # Verify that the source frame stays clean when object frame is marked dirty.
    assert not result_source_frame.is_dirty()

    # Perform a series of operations on the ObjectFrame and then verify the result is still a
    # proper ObjectFrame with appropriate metadata propagated.
    result_object_frame = object_frame.copy()[["nobs_g", "nobs_total"]]
    result_object_frame = result_object_frame.map_partitions(lambda x: x.copy())
    assert isinstance(result_object_frame, ObjectFrame)
    assert isinstance(result_object_frame._meta, TapeObjectFrame)
    assert result_object_frame.label == OBJECT_LABEL
    assert result_object_frame.ensemble is ens
    assert result_object_frame.is_dirty()

    # Mark the frame clean to verify that we propagate that state as well
    result_object_frame.set_dirty(False)

    # Set an index and then group by that index.
    result_object_frame = result_object_frame.set_index("nobs_g", drop=True)
    assert result_object_frame.label == OBJECT_LABEL
    assert result_object_frame.ensemble == ens
    assert not result_object_frame.is_dirty()  # frame is still clean
    group_result = result_object_frame.groupby(["nobs_g"]).count()
    assert len(group_result) > 0
    assert isinstance(group_result, ObjectFrame)
    assert isinstance(group_result._meta, TapeObjectFrame)

    # Test merging source and object frames, dropping some columns, and persisting the result.
    merged_frame = source_frame.copy().merge(
        object_frame.copy(), on=[ens._id_col], suffixes=(None, "_drop_me")
    )
    cols_to_drop = [col for col in merged_frame.columns if "_drop_me" in col]
    merged_frame = merged_frame.drop(cols_to_drop, axis=1).persist()
    assert isinstance(merged_frame, SourceFrame)
    assert merged_frame.label == SOURCE_LABEL
    assert merged_frame.ensemble == ens
    assert merged_frame.is_dirty()


def test_object_and_source_joins(parquet_ensemble):
    """
    Test that SourceFrame and ObjectFrame metadata and class type are correctly propagated across
    joins.
    """
    # Get Source and object frames to test joins on.
    source_frame, object_frame = parquet_ensemble.source.copy(), parquet_ensemble.object.copy()

    # Verify their metadata was preserved in the copy()
    assert source_frame.label == SOURCE_LABEL
    assert source_frame.ensemble is parquet_ensemble
    assert object_frame.label == OBJECT_LABEL
    assert object_frame.ensemble is parquet_ensemble

    # Join a SourceFrame (left) with an ObjectFrame (right)
    # Validate that metadata is preserved and the outputted object is a SourceFrame
    joined_source = source_frame.join(object_frame, how="left")
    assert joined_source.label is SOURCE_LABEL
    assert type(joined_source) is SourceFrame
    assert joined_source.ensemble is parquet_ensemble

    # Now the same form of join (in terms of left/right) but produce an ObjectFrame. This is
    # because frame1.join(frame2) will yield frame1's type regardless of left vs right.
    assert type(object_frame.join(source_frame, how="right")) is ObjectFrame

    # Join an ObjectFrame (left) with a SourceFrame (right)
    # Validate that metadata is preserved and the outputted object is an ObjectFrame
    joined_object = object_frame.join(source_frame, how="left")
    assert joined_object.label is OBJECT_LABEL
    assert type(joined_object) is ObjectFrame
    assert joined_object.ensemble is parquet_ensemble

    # Now the same form of join (in terms of left/right) but produce a SourceFrame. This is
    # because frame1.join(frame2) will yield frame1's type regardless of left vs right.
    assert type(source_frame.join(object_frame, how="right")) is SourceFrame


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

    ens.source.coalesce(["flux1", "flux2", "flux3"], "flux", drop_inputs=drop_inputs).update_ensemble()

    # Coalesce should return this exact flux array
    assert list(ens.source["flux"].values.compute()) == [5.0, 3.0, 4.0, 10.0, 7.0]

    if drop_inputs:
        # The column mapping should be updated
        assert ens.make_column_map().map["flux_col"] == "flux"

        # The columns to drop should be dropped
        for col in ["flux1", "flux2", "flux3"]:
            assert col not in ens.source.columns

        # Test for the drop warning
        with pytest.warns(UserWarning):
            ens.source.coalesce(["time", "flux"], "bad_col", drop_inputs=drop_inputs)

    else:
        # The input columns should still be present
        for col in ["flux1", "flux2", "flux3"]:
            assert col in ens.source.columns


def test_partition_slicing(parquet_ensemble_with_divisions):
    """
    Test that partition slicing propagates EnsembleFrame metadata
    """
    ens = parquet_ensemble_with_divisions

    ens.source.repartition(npartitions=10).update_ensemble()
    ens.object.repartition(npartitions=5).update_ensemble()

    prior_obj_len = len(ens.object)
    prior_src_len = len(ens.source)

    # slice on object
    ens.object.partitions[0:3].update_ensemble()
    ens._lazy_sync_tables("all")  # sync needed as len() won't trigger one

    assert ens.object.npartitions == 3  # should return exactly 3 partitions
    assert len(ens.source) < prior_src_len  # should affect source

    prior_obj_len = len(ens.object)
    prior_src_len = len(ens.source)

    # slice on source
    ens.source.partitions[0:2].update_ensemble()
    ens._lazy_sync_tables("all")  # sync needed as len() won't trigger one

    assert ens.source.npartitions == 2  # should return exactly 2 partitions
    assert len(ens.object) < prior_src_len  # should affect objects
