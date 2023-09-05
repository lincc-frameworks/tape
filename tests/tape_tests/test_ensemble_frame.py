""" Test EnsembleFrame (inherited from Dask.DataFrame) creation and manipulations. """
import numpy as np
import pandas as pd
from tape import ColumnMapper, EnsembleFrame, ObjectFrame, SourceFrame, TapeObjectFrame, TapeSourceFrame, TapeFrame

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
    ens_frame = EnsembleFrame.from_dict(data, 
                                        npartitions=1)

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
    ens_frame = EnsembleFrame.from_tapeframe(frame, 
                                          label=TEST_LABEL, 
                                          ensemble=ens,
                                          npartitions=1)

    assert isinstance(ens_frame, EnsembleFrame)
    assert isinstance(ens_frame._meta, TapeFrame)
    assert ens_frame.label == TEST_LABEL
    assert ens_frame.ensemble is ens

    # The calculation for finding the max flux from the data. Note that the
    # inherited dask compute method must be called to obtain the result. 
    assert ens_frame.flux.max().compute() == 80.6


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
    ens_frame = EnsembleFrame.from_dict(data,
                                        npartitions=1)
    # Set a label and ensemble for the frame and copies/transformations retain them.
    ens_frame.label = TEST_LABEL
    ens_frame.ensemble=ens

    # Create a copy of an EnsembleFrame and verify that it's still a proper
    # EnsembleFrame with appropriate metadata propagated.
    copied_frame = ens_frame.copy()
    assert isinstance(copied_frame, EnsembleFrame)
    assert isinstance(copied_frame._meta, TapeFrame)
    assert copied_frame.label == TEST_LABEL
    assert copied_frame.ensemble == ens

    # Test that a filtered EnsembleFrame is still an EnsembleFrame.
    filtered_frame = ens_frame[["id", "time"]]
    assert isinstance(filtered_frame, EnsembleFrame)
    assert isinstance(filtered_frame._meta, TapeFrame)
    assert filtered_frame.label == TEST_LABEL
    assert filtered_frame.ensemble == ens

    # Test that the output of an EnsembleFrame query is still an EnsembleFrame
    queried_rows = ens_frame.query("flux > 3.0")
    assert isinstance(filtered_frame, EnsembleFrame)
    assert isinstance(filtered_frame._meta, TapeFrame)
    assert filtered_frame.label == TEST_LABEL
    assert filtered_frame.ensemble == ens

    # Test that head returns a subset of the underlying TapeFrame.
    h = ens_frame.head(5)
    assert isinstance(h, TapeFrame)
    assert len(h) == 5

    # Test that the inherited dask.DataFrame.compute method returns
    # the underlying TapeFrame. 
    assert isinstance(ens_frame.compute(), TapeFrame)
    assert len(ens_frame) == len(ens_frame.compute())

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
        "parquet_files_and_ensemble_without_client",
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

    # Perform a series of operations on the SourceFrame and then verify the result is still a
    # proper SourceFrame with appropriate metadata propagated.
    mean_ps_flux = source_frame["psFlux"].mean().compute()
    result_source_frame = source_frame.copy()[["psFlux", "psFluxErr"]]
    assert isinstance(result_source_frame, SourceFrame)
    assert isinstance(result_source_frame._meta, TapeSourceFrame)
    assert len(result_source_frame) > 0
    assert result_source_frame.label == SOURCE_LABEL
    assert result_source_frame.ensemble is not None
    assert result_source_frame.ensemble is ens

    # Create an ObjectFrame from a parquet file
    object_frame = ObjectFrame.from_parquet(
        object_file,
        ensemble=ens,
        index="ps1_objid",
    )

    assert isinstance(object_frame, EnsembleFrame)
    assert isinstance(object_frame, ObjectFrame)
    assert isinstance(object_frame._meta, TapeObjectFrame)

    # Perform a series of operations on the ObjectFrame and then verify the result is still a
    # proper ObjectFrame with appropriate metadata propagated.
    result_object_frame = object_frame.copy()[["nobs_g", "nobs_total"]]
    assert isinstance(result_object_frame, ObjectFrame)
    assert isinstance(result_object_frame._meta, TapeObjectFrame)
    assert result_object_frame.label == OBJECT_LABEL
    assert result_object_frame.ensemble is ens