import pytest
from tape.utils import ColumnMapper, IndexCallable


def test_index_callable(parquet_ensemble):
    """
    Test the basic function of the IndexCallable object
    """

    ens = parquet_ensemble

    source_ic = IndexCallable(ens.source._partitions, True, "ensemble")

    # grab the first (and only) source partition
    sliced_source_frame = source_ic[0]

    # ensure that the metadata propagates to the result
    assert sliced_source_frame.dirty is True
    assert sliced_source_frame.ensemble == "ensemble"


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


@pytest.mark.parametrize("map_id", ["ZTF", "PS1", "ztf", "Grundor"])
@pytest.mark.parametrize("hipscat", [True, False])
def test_use_known_map(map_id, hipscat):
    """test a known mapping"""
    cmap = ColumnMapper()

    if map_id == "Grundor":
        with pytest.raises(ValueError):
            cmap = cmap.use_known_map(map_id, hipscat=hipscat)
    else:
        cmap = cmap.use_known_map(map_id, hipscat=hipscat)
        assert cmap.is_ready()
