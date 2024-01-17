from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from tape.analysis.base import AnalysisFunction


__all__ = ["calc_stetson_J", "StetsonJ"]


class StetsonJ(AnalysisFunction):
    """Compute the StetsonJ statistic on data from one or several bands"""

    def cols(self, ens: "Ensemble") -> List[str]:
        return [ens._flux_col, ens._err_col, ens._band_col]

    def meta(self, ens: "Ensemble"):
        return "stetsonJ", float

    def on(self, ens: "Ensemble") -> List[str]:
        return [ens._id_col]

    def __call__(
        self,
        flux: np.ndarray,
        err: np.ndarray,
        band: np.ndarray,
        *,
        band_to_calc: Union[str, Iterable[str], None] = None,
        check_nans: bool = False,
    ):
        """Compute the StetsonJ statistic on data from one or several bands

        Parameters
        ----------
        flux : `numpy.ndarray` (N,)
            Array of flux/magnitude measurements
        err : `numpy.ndarray` (N,)
            Array of associated flux/magnitude errors
        band : `numpy.ndarray` (N,)
            Array of associated band labels
        band_to_calc : `str` or `list` of `str`
            Bands to calculate StetsonJ on. Single band descriptor, or list
            of such descriptors.
        check_nans : `bool`
            Boolean to run a check for NaN values and filter them out.

        Returns
        -------
        stetsonJ : `dict`
            StetsonJ statistic for each of input bands.

        Note
        ----------
        In case that no value for `band_to_calc` is passed, the function is
        executed on all available bands in `band`.
        """

        # NaN filtering
        if check_nans:
            f_mask = np.isnan(flux)
            e_mask = np.isnan(err)  # always mask out nan errors?
            nan_mask = np.logical_or(f_mask, e_mask)

            flux = flux[~nan_mask]
            err = err[~nan_mask]
            band = band[~nan_mask]

        unq_band = np.unique(band)

        if band_to_calc is None:
            band_to_calc = unq_band
        if isinstance(band_to_calc, str):
            band_to_calc = [band_to_calc]

        assert isinstance(band_to_calc, Iterable) is True

        stetsonJ = {}
        for b in band_to_calc:
            if b in unq_band:
                mask = band == b
                fluxes = flux[mask]
                errors = err[mask]
                stetsonJ[b] = _stetson_J_single(fluxes, errors)
            else:
                stetsonJ[b] = np.nan

        return stetsonJ


calc_stetson_J = StetsonJ()
calc_stetson_J.__doc__ = StetsonJ.__call__.__doc__


def _stetson_J_single(fluxes, errors):
    """Compute the single band stetsonJ statistic.

    Parameters
    ----------
    fluxes : `numpy.ndarray` (N,)
        Lightcurve flux values.
    errors : `numpy.ndarray` (N,)
        Errors on the lightcurve fluxes.

    Returns
    -------
    stetsonJ : `float`
        StetsonJ statistic

    References
    ----------
    .. [1] Stetson, P. B., "On the Automatic Determination of Light-Curve
    Parameters for Cepheid Variables", PASP, 108, 851S, 1996

    Note
    ----------
    Taken from
    https://github.com/lsst/meas_base/blob/main/python/lsst/meas/base/diaCalculationPlugins.py
    Using the function on random gaussian distribution gives result of -0.2
    instead of expected result of 0?
    """

    n_points = len(fluxes)
    if n_points <= 1:
        return np.nan
    flux_mean = _stetson_J_mean(fluxes, errors)
    delta_val = np.sqrt(n_points / (n_points - 1)) * (fluxes - flux_mean) / errors
    p_k = delta_val**2 - 1
    return np.mean(np.sign(p_k) * np.sqrt(np.fabs(p_k)))


def _stetson_J_mean(values, errors, mean=None, alpha=2.0, beta=2.0, n_iter=20, tol=1e-6):
    """Compute the stetson mean of the fluxes which down-weights outliers.

    Weighted biased on an error weighted difference scaled by a constant
    (1/``a``) and raised to the power beta. Higher betas more harshly
    penalize outliers and ``a`` sets the number of sigma where a weighted
    difference of 1 occurs.

    Parameters
    ----------
    values : `numpy.dnarray`, (N,)
        Input values to compute the mean of.
    errors : `numpy.ndarray`, (N,)
        Errors on the input values.
    mean : `float`
        Starting mean value or None.
    alpha : `float`
        Scalar down-weighting of the fractional difference. lower->more
        clipping. (Default value is 2.)
    beta : `float`
        Power law slope of the used to down-weight outliers. higher->more
        clipping. (Default value is 2.)
    n_iter : `int`
        Number of iterations of clipping.
    tol : `float`
        Fractional and absolute tolerance goal on the change in the mean
        before exiting early. (Default value is 1e-6)

    Returns
    -------
    mean : `float`
        Weighted stetson mean result.

    References
    ----------
    .. [1] Stetson, P. B., "On the Automatic Determination of Light-Curve
    Parameters for Cepheid Variables", PASP, 108, 851S, 1996

    Note
    ----------
    Taken from
    https://github.com/lsst/meas_base/blob/main/python/lsst/meas/base/diaCalculationPlugins.py
    """
    n_points = len(values)
    n_factor = np.sqrt(n_points / (n_points - 1))
    inv_var = 1 / errors**2
    if mean is None:
        mean = np.average(values, weights=inv_var)
    for iter_idx in range(n_iter):
        chi = np.fabs(n_factor * (values - mean) / errors)
        tmp_mean = np.average(values, weights=inv_var / (1 + (chi / alpha) ** beta))
        diff = np.fabs(tmp_mean - mean)
        mean = tmp_mean
        if mean == 0:  # catch divide by zero in diff / mean
            break
        elif diff / mean < tol and diff < tol:
            break
    return mean
