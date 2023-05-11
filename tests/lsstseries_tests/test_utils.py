import pytest
from lsstseries.utils import ColumnMapper


def test_column_mapper():
    """
    Test the basic elements of the ColumnMapper object
    """

    col_map = ColumnMapper()  # create empty column map

    ready, needed = col_map.is_ready(show_needed=True)

    assert not ready  # the col_map should not be ready for an ensemble

    for item in col_map.required.items():
        if item[1]:
            assert item[0] in needed  # all required columns should be captured here

    # Assign required columns
    col_map.assign(id_col="id", flux_col="flux", err_col="err", band_col="band")

    assert col_map.is_ready()  # col_map should now be ready

    # Assign the remaining columns
    col_map.assign(
        provenance_col="provenance", nobs_total_col="nobs_total", nobs_band_cols=["nobs_g", "nobs_r"]
    )

    expected_map = {
        "id_col": "id",
        "flux_col": "flux",
        "err_col": "err",
        "band_col": "band",
        "provenance_col": "provenance",
        "nobs_total_col": "nobs_total",
        "nobs_band_cols": ["nobs_g", "nobs_r"],
    }

    assert col_map.map == expected_map  # The expected mapping


def test_column_mapper_init():
    """
    Test the mapping at init of a ColumnMapper object
    """

    col_map = ColumnMapper(
        id_col="id",
        flux_col="flux",
        err_col="err",
        band_col="band",
        provenance_col="provenance",
        nobs_total_col="nobs_total",
        nobs_band_cols=["nobs_g", "nobs_r"],
    )

    assert col_map.is_ready()  # col_map should be ready

    expected_map = {
        "id_col": "id",
        "flux_col": "flux",
        "err_col": "err",
        "band_col": "band",
        "provenance_col": "provenance",
        "nobs_total_col": "nobs_total",
        "nobs_band_cols": ["nobs_g", "nobs_r"],
    }

    assert col_map.map == expected_map  # The expected mapping
