"""Test ensemble manipulations"""

import pandas as pd

from tape import TimeSeries


def test_from_dict():
    """
    Test that we can build a TimeSeries from a dict
    """
    test_dict1 = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "flux": [10.0, 11.0, 10.5, 12.0, 11.5],
        "flux_err": [0.1, 0.1, 0.2, 0.3, 0.1],
        "band": ["r", "g", "g", "r", "r"],
    }
    ts1 = TimeSeries()
    ts1.from_dict(test_dict1)

    # The flux lists will be ordered by the band
    assert ts1.flux.tolist() == [11.0, 10.5, 10.0, 12.0, 11.5]
    assert ts1.flux_err.tolist() == [0.1, 0.2, 0.1, 0.3, 0.1]
    assert ts1.time.tolist() == [2.0, 3.0, 1.0, 4.0, 5.0]

    # Try with alternate names and no flux error.
    test_dict2 = {
        "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "col2": [10.0, 11.0, 10.5, 12.0, 11.5],
        "col4": ["r", "g", "g", "r", "r"],
    }
    ts2 = TimeSeries()
    ts2.from_dict(test_dict2, time_label="col1", flux_label="col2", band_label="col4", err_label=None)

    # The flux lists will be ordered by the band
    assert ts2.flux.tolist() == [11.0, 10.5, 10.0, 12.0, 11.5]
    assert ts2.flux_err is None
    assert ts2.time.tolist() == [2.0, 3.0, 1.0, 4.0, 5.0]


def test_from_dataframe():
    """
    Test that we can build a TimeSeries from a pandas dataframe.
    """
    test_dict1 = {
        "time_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "flux_col": [10.0, 11.0, 10.5, 12.0, 11.5],
        "flux_err_col": [0.1, 0.1, 0.2, 0.3, 0.1],
        "band_col": ["r", "g", "g", "r", "r"],
    }
    pdf = pd.DataFrame(test_dict1)

    ts = TimeSeries().from_dataframe(
        pdf,
        "my_object_id",
        time_label="time_col",
        flux_label="flux_col",
        err_label="flux_err_col",
        band_label="band_col",
    )

    # The flux lists will be in the order of the DataFrame.
    assert ts.meta["id"] == "my_object_id"
    assert ts.flux.tolist() == [10.0, 11.0, 10.5, 12.0, 11.5]
    assert ts.flux_err.tolist() == [0.1, 0.1, 0.2, 0.3, 0.1]
    assert ts.time.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_build_index():
    """
    Test that ensemble indexing returns expected behavior
    """
    bands = ["u", "u", "u", "g", "g", "u", "u"]

    ts = TimeSeries()
    result = ts._build_index(bands)
    assert len(result.levels) == 2

    result_bands = list(result.get_level_values(0))
    assert result_bands == bands

    result_ids = list(result.get_level_values(1))
    target = [0, 1, 2, 0, 1, 3, 4]
    assert result_ids == target
