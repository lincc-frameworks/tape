""" Test EnsembleFrame (inherited from Dask.DataFrame) creation and manipulations. """
import pandas as pd
from tape import Ensemble, EnsembleFrame, TapeFrame

import pytest

# Create some fake lightcurve data with two IDs (8001, 8002), two bands ["g", "b"]
# and a few time steps.
SAMPLE_LC_DATA = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }
TEST_LABEL = "test_frame"
TEST_ENSEMBLE = Ensemble()

def test_from_dict():
    """
    Test creating an EnsembleFrame from a dictionary and verify that dask lazy evaluation was appropriately inherited.
    """
    ens_frame = EnsembleFrame.from_dict(SAMPLE_LC_DATA, 
                                        label=TEST_LABEL, 
                                        ensemble=TEST_ENSEMBLE,
                                        npartitions=1)

    assert isinstance(ens_frame, EnsembleFrame)
    assert isinstance(ens_frame._meta, TapeFrame)
    assert ens_frame.label == TEST_LABEL
    assert ens_frame.ensemble is TEST_ENSEMBLE

    # The calculation for finding the max flux from the data. Note that the
    # inherited dask compute method must be called to obtain the result. 
    assert ens_frame.flux.max().compute() == 5.0

def test_from_pandas():
    """
    Test creating an EnsembleFrame from a Pandas dataframe and verify that dask lazy evaluation was appropriately inherited.
    """
    frame = TapeFrame(SAMPLE_LC_DATA)
    ens_frame = EnsembleFrame.from_tapeframe(frame, 
                                          label=TEST_LABEL, 
                                          ensemble=TEST_ENSEMBLE,
                                          npartitions=1)

    assert isinstance(ens_frame, EnsembleFrame)
    assert isinstance(ens_frame._meta, TapeFrame)
    assert ens_frame.label == TEST_LABEL
    assert ens_frame.ensemble is TEST_ENSEMBLE

    # The calculation for finding the max flux from the data. Note that the
    # inherited dask compute method must be called to obtain the result. 
    assert ens_frame.flux.max().compute() == 5.0


def test_frame_propagation():
    """
    Test ensuring that slices and copies of an EnsembleFrame or still the same class.
    """
    ens_frame = EnsembleFrame.from_dict(SAMPLE_LC_DATA,
                                        label=TEST_LABEL,
                                        ensemble=TEST_ENSEMBLE,
                                        npartitions=1)

    # Create a copy of an EnsembleFrame and verify that it's still a proper
    # EnsembleFrame with appropriate metadata propagated.
    copied_frame = ens_frame.copy()
    assert isinstance(copied_frame, EnsembleFrame)
    assert isinstance(copied_frame._meta, TapeFrame)
    assert copied_frame.label == TEST_LABEL
    assert copied_frame.ensemble == TEST_ENSEMBLE

    # Test that a filtered EnsembleFrame is still an EnsembleFrame.
    filtered_frame = ens_frame[["id", "time"]]
    assert isinstance(filtered_frame, EnsembleFrame)
    assert isinstance(filtered_frame._meta, TapeFrame)
    assert filtered_frame.label == TEST_LABEL
    assert filtered_frame.ensemble == TEST_ENSEMBLE

    # Test that head returns a subset of the underlying TapeFrame.
    h = ens_frame.head(5)
    assert isinstance(h, TapeFrame)
    assert len(h) == 5

    # Test that the inherited dask.DataFrame.compute method returns
    # the underlying TapeFrame. 
    assert isinstance(ens_frame.compute(), TapeFrame)
    assert len(ens_frame) == len(ens_frame.compute())