import pytest
from tape.utils import ColumnMapper


def test_column_mapper():
    """
    Test the basic elements of the ColumnMapper object
    """

    col_map = ColumnMapper()  # create empty column map

    ready, needed = col_map.is_ready(show_needed=True)

    assert not ready  # the col_map should not be ready for an ensemble

    for col in col_map.required:
        if col.is_required:
            assert col.name in needed  # all required columns should be captured here

    # Assign required columns
    col_map.assign(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")

    assert col_map.is_ready()  # col_map should now be ready

    expected_map = {
        "id_col": "id",
        "time_col": "time",
        "flux_col": "flux",
        "err_col": "err",
        "band_col": "band",
    }

    assert col_map.map == expected_map  # The expected mapping


def test_column_mapper_init():
    """
    Test the mapping at init of a ColumnMapper object
    """

    col_map = ColumnMapper(
        id_col="id",
        time_col="time",
        flux_col="flux",
        err_col="err",
        band_col="band",
    )

    assert col_map.is_ready()  # col_map should be ready

    expected_map = {
        "id_col": "id",
        "time_col": "time",
        "flux_col": "flux",
        "err_col": "err",
        "band_col": "band",
    }

    assert col_map.map == expected_map  # The expected mapping

    @pytest.mark.parametrize("map_id", ["ZTF", "Grundor"])
    def test_use_known_map(map_id):
        """test a known mapping"""
        cmap = ColumnMapper()

        if map_id == "Grundor":
            with pytest.raises(ValueError):
                cmap.use_known_map(map_id)
        else:
            cmap.use_known_map(map_id)
            assert cmap.is_ready()
