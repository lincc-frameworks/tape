"""Test ensemble manipulations"""
import pytest

from lsstseries import Ensemble
from lsstseries.analysis.stetsonj import calc_stetson_J
from lsstseries.analysis.structurefunction2 import calc_sf2


# pylint: disable=protected-access
def test_with():
    """Test that we open and close a client on enter and exit."""
    with Ensemble() as ens:
        ens.from_parquet(
            "tests/lsstseries_tests/data/test_subset.parquet",
            id_col="ps1_objid",
            band_col="filterName",
            flux_col="psFlux",
            err_col="psFluxErr",
        )
        assert ens._data is not None


def test_from_parquet(parquet_ensemble):
    """
    Test that ensemble.from_parquet() successfully loads a parquet file
    """
    # Check to make sure the data property was actually set
    assert parquet_ensemble._data is not None

    parquet_ensemble._data = parquet_ensemble.compute()
    for col in [
        parquet_ensemble._time_col,
        parquet_ensemble._flux_col,
        parquet_ensemble._err_col,
        parquet_ensemble._band_col,
    ]:
        # Check to make sure the critical quantity labels are bound to real columns
        assert parquet_ensemble._data[col] is not None


def test_core_wrappers(parquet_ensemble):
    """
    Test that the core wrapper functions execute without errors
    """
    # Just test if these execute successfully
    parquet_ensemble.info()
    parquet_ensemble.columns()
    parquet_ensemble.head(5)
    parquet_ensemble.tail(5)
    parquet_ensemble.compute()


def test_counts(parquet_ensemble):
    """
    Test that ensemble.count() runs and returns the first five values correctly
    """
    count = parquet_ensemble.count()
    assert list(count.values[0:5]) == [499, 343, 337, 195, 188]


def test_prune(parquet_ensemble):
    """
    Test that ensemble.prune() appropriately filters the dataframe
    """
    threshold = 10
    parquet_ensemble.prune(threshold)
    assert parquet_ensemble.count(ascending=False).values[0] >= threshold


def test_batch(parquet_ensemble):
    """
    Test that ensemble.batch() returns the correct values of the first result
    """
    result = (
        parquet_ensemble.prune(10).dropna(1).batch(calc_stetson_J, band_to_calc=None)
    )

    assert pytest.approx(result.values[0]["g"], 0.001) == -0.04174282
    assert pytest.approx(result.values[0]["r"], 0.001) == 0.6075282


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
    result = list(ens._build_index(obj_ids, bands).get_level_values(2))
    target = [0, 1, 2, 0, 0, 0, 1]
    assert result == target


@pytest.mark.parametrize("method", ["size", "length", "loglength"])
@pytest.mark.parametrize("combine", [True, False])
@pytest.mark.parametrize("sthresh", [50, 100])
def test_sf2(parquet_ensemble, method, combine, sthresh):
    """
    Test calling sf2 from the ensemble
    """

    res_sf2 = parquet_ensemble.sf2(combine=combine, method=method, sthresh=sthresh)
    res_batch = parquet_ensemble.batch(
        calc_sf2, combine=combine, method=method, sthresh=sthresh
    )

    if combine:
        assert not res_sf2.equals(res_batch)  # output should be different
    else:
        assert res_sf2.equals(res_batch)  # output should be identical
