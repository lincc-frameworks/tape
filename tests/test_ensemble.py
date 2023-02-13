"""Test ensemble manipulations"""
import pytest
from lsstseries import ensemble
from lsstseries.analysis.stetsonj import calc_stetson_J
from lsstseries.analysis.structurefunction2 import calc_sf2


@pytest.fixture
def parquet_data():
    ens = ensemble()
    ens.from_parquet("tests/data/test_subset.parquet", id_col='ps1_objid', band_col='filterName',
                     flux_col='psFlux', err_col='psFluxErr')

    yield ens
    ens.client.close()


def test_from_parquet(parquet_data):
    """
    Test that ensemble.from_parquet() successfully loads a parquet file
    """
    ens = parquet_data

    assert ens.data is not None  # Check to make sure the data property was actually set
    ens.data = ens.data.compute()
    for col in [ens._time_col, ens._flux_col, ens._err_col, ens._band_col]:
        ens.data[col]  # Check to make sure the critical quantity labels are bound to real columns


def test_counts(parquet_data):
    """
    Test that ensemble.count() runs and returns the first five values correctly
    """
    ens = parquet_data

    count = ens.count()
    assert list(count.values[0:5]) == [499, 343, 337, 195, 188]


def test_prune(parquet_data):
    """
    Test that ensemble.prune() appropriately filters the dataframe
    """
    ens = parquet_data

    threshold = 10
    ens.prune(threshold)
    assert ens.count(ascending=False).values[0] >= threshold


def test_batch(parquet_data):
    """
    Test that ensemble.batch() returns the correct values of the first result
    """
    ens = parquet_data
    result = ens.prune(10).dropna(1).batch(calc_stetson_J, band_to_calc=None)

    assert pytest.approx(result.values[0]['g'], 0.001) == -0.04174282
    assert pytest.approx(result.values[0]['r'], 0.001) == 0.6075282


def test_to_timeseries(parquet_data):
    """
    Test that ensemble.to_timeseries() runs and assigns the correct metadata
    """
    ens = parquet_data
    ts = ens.to_timeseries('88480000290704349')

    assert ts.meta['id'] == '88480000290704349'


def test_build_index():
    """
    Test that ensemble indexing returns expected behavior
    """

    obj_ids = [1, 1, 1, 2, 1, 2, 2]
    bands = ["u", "u", "u", "g", "g", "u", "u"]

    ens = ensemble()
    result = list(ens._build_index(obj_ids, bands).get_level_values(2))
    target = [0, 1, 2, 0, 0, 0, 1]
    assert result == target
    ens.client.close()


@pytest.mark.parametrize("method", ["size", "length", "loglength"])
@pytest.mark.parametrize("combine", [True, False])
@pytest.mark.parametrize("sthresh", [50, 100])
def test_sf2(parquet_data, method, combine, sthresh):
    """
    Test calling sf2 from the ensemble
    """

    ens = parquet_data

    res_sf2 = ens.sf2(combine=combine, method=method, sthresh=sthresh)
    res_batch = ens.batch(calc_sf2, combine=combine, method=method, sthresh=sthresh)

    if combine:
        assert not res_sf2.equals(res_batch)  # output should be different
    else:
        assert res_sf2.equals(res_batch)  # output should be identical
