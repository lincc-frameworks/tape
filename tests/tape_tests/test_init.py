import pytest
from importlib import reload
import dask
import tape


def test_expr_config():
    """test that the query-planning config is set back to True on package import"""
    reload(dask)

    dask.config.set({"dataframe.query-planning": False})

    reload(tape)

    assert dask.config.get("dataframe.query-planning")
