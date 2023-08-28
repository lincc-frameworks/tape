""" Test EnsembleFrame (inherited from Dask.DataFrame) creation and manipulations. """
import pandas as pd
from tape import Ensemble, EnsembleFrame, TapeFrame

import pytest

TEST_LABEL = "test_frame"

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
    assert ens_frame.flux.max().compute() == 5.0

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
    assert ens_frame.flux.max().compute() == 5.0


@pytest.mark.parametrize(
    "data_fixture",
    [
        "ensemble_from_source_dict",
    ],
)
def test_frame_propagation(data_fixture, request):
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
    assert isinstance(queried_rows, EnsembleFrame)
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