"""Test ensemble manipulations"""

import copy
import os
import lsdb

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import tape

from tape import (
    Ensemble,
    EnsembleFrame,
    EnsembleSeries,
    ObjectFrame,
    SourceFrame,
    TapeFrame,
    TapeSeries,
    TapeObjectFrame,
    TapeSourceFrame,
    TimeSeries,
)
from tape.analysis.stetsonj import calc_stetson_J
from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structurefunction2 import calc_sf2
from tape.utils import ColumnMapper


# pylint: disable=protected-access
def test_with_client():
    """Test that we open and close a client on enter and exit."""
    with Ensemble() as ens:
        ens.from_parquet(
            "tests/tape_tests/data/source/test_source.parquet",
            id_col="ps1_objid",
            band_col="filterName",
            flux_col="psFlux",
            err_col="psFluxErr",
        )
        assert ens._data is not None


@pytest.mark.parametrize("on", [None, ["ps1_objid", "filterName"], ["filterName", "ps1_objid"]])
@pytest.mark.parametrize("func_label", ["mean", "bounds"])
def test_batch_by_band(parquet_ensemble, func_label, on):
    """
    Test that ensemble.batch(by_band=True) works as intended.
    """

    if func_label == "mean":

        def my_mean(flux):
            """returns a single value"""
            return np.mean(flux)

        res = parquet_ensemble.batch(my_mean, parquet_ensemble._flux_col, on=on, by_band=True)

        parquet_ensemble.source.query(f"{parquet_ensemble._band_col}=='g'").update_ensemble()
        filter_res = parquet_ensemble.batch(my_mean, parquet_ensemble._flux_col, on=on, by_band=False)

        # An EnsembleFrame should be returned
        assert isinstance(res, EnsembleFrame)

        # Make sure we get all the expected columns
        assert all([col in res.columns for col in ["result_g", "result_r"]])

        # These should be equivalent
        # [expr] need this TODO: investigate typing issue
        filter_res.index = filter_res.index.astype("int")
        assert (
            res.loc[88472935274829959]["result_g"]
            .compute()
            .equals(filter_res.loc[88472935274829959]["result"].compute())
        )

    elif func_label == "bounds":

        def my_bounds(flux):
            """returns a series"""
            return pd.Series({"min": np.min(flux), "max": np.max(flux)})

        res = parquet_ensemble.batch(
            my_bounds, "psFlux", on=on, by_band=True, meta={"min": float, "max": float}
        )

        parquet_ensemble.source.query(f"{parquet_ensemble._band_col}=='g'").update_ensemble()
        filter_res = parquet_ensemble.batch(
            my_bounds, "psFlux", on=on, by_band=False, meta={"min": float, "max": float}
        )

        # An EnsembleFrame should be returned
        assert isinstance(res, EnsembleFrame)

        # Make sure we get all the expected columns
        assert all([col in res.columns for col in ["max_g", "max_r", "min_g", "min_r"]])

        # These should be equivalent

        # [expr] need this TODO: investigate typing issue
        filter_res.index = filter_res.index.astype("int")

        assert (
            res.loc[88472935274829959]["max_g"]
            .compute()
            .equals(filter_res.loc[88472935274829959]["max"].compute())
        )
        assert (
            res.loc[88472935274829959]["min_g"]
            .compute()
            .equals(filter_res.loc[88472935274829959]["min"].compute())
        )

    # Meta should reflect the actual columns, this can get out of sync
    # whenever multi-indexes are involved, which batch tries to handle
    assert all([col in res.columns for col in res.compute().columns])
