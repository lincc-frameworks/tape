"""
Contains the base class for analysis functions.
"""

from abc import ABC, abstractmethod
from typing import Callable, List

import pandas as pd


class AnalysisFunction(ABC, Callable):
    """Base class for analysis functions.

    Analysis functions are functions that take few arrays representing
    an object and return a single pandas.Series representing the result.

    Methods
    -------
    cols(ens) -> List[str]
        Return the columns that the analysis function takes as input.
    meta(ens) -> pd.DataFrame
        Return the metadata pandas.DataFrame required by Dask to pre-build
        a computation graph. It is basically the schema for calculate() method
        output.
    on(ens) -> List[str]
        Return the columns to group source table by.
        Typically, `[ens._id_col]`.
    __call__(*cols, **kwargs)
        Calculate the analysis function.
    """

    @abstractmethod
    def cols(self, ens: "Ensemble") -> List[str]:
        """
        Return the column names that the analysis function takes as input.

        Parameters
        ----------
        ens : Ensemble
            The ensemble object, it could be required to get column names of
            the "special" columns like `ens._time_col` or `ens._err_col`.

        Returns
        -------
        List[str]
            The column names to select and pass to .calculate() method.
            For example `[ens._time_col, ens._flux_col]`.
        """
        raise NotImplementedError

    @abstractmethod
    def meta(self, ens: "Ensemble"):
        """
        Return the schema of the analysis function output.

        Parameters
        ----------
        ens : Ensemble
            The ensemble object.

        Returns
        -------
        pd.DataFrame or (str, dtype) tuple or {str: dtype} dictionary
            Dask meta, for example
            `pd.DataFrame(columns=['x', 'y'], dtype=float)`.
        """
        raise NotImplementedError

    @abstractmethod
    def on(self, ens: "Ensemble") -> List[str]:
        """Return the columns to group source table by.

        Parameters
        ----------
        ens : Ensemble
            The ensemble object.

        Returns
        --------
        List[str]
            The column names to group by. Typically, `[ens._id_col]`.
        """
        return [ens._id_col]

    @abstractmethod
    def __call__(self, *cols, **kwargs):
        """Calculate the analysis function.

        Parameters
        ----------
        *cols : array_like
            The columns to calculate the analysis function on. It must be
            consistent with .cols(ens) output.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        pd.Series or pd.DataFrame or array or value
            The result, it must be consistent with .meta() output.
        """
        raise NotImplementedError
