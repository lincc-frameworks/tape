"""
Auxiliary code for time-series feature extraction with "light-curve"  package
"""

from typing import List

import numpy as np
import pandas as pd
from light_curve.light_curve_ext import _FeatureEvaluator as BaseLightCurveFeature

from tape.analysis.base import AnalysisFunction


__all__ = ["FeatureExtractor", "BaseLightCurveFeature"]


class FeatureExtractor(AnalysisFunction):
    """Apply light-curve package feature extractor to a light curve

    Parameters
    ----------
    feature : light_curve.light_curve_ext._FeatureEvaluator
        Feature extractor to apply, see "light-curve" package for more details.

    Attributes
    ----------
    feature : light_curve.light_curve_ext._FeatureEvaluator
        Feature extractor to apply, see "light-curve" package for more details.
    """

    def __init__(self, feature: BaseLightCurveFeature):
        self.feature = feature

    def cols(self, ens: "Ensemble") -> List[str]:
        return [ens._time_col, ens._flux_col, ens._err_col, ens._band_col]

    def meta(self, ens: "Ensemble") -> pd.DataFrame:
        """Return the schema of the analysis function output.

        It always returns a pandas.DataFrame with the same columns as
        `self.feature.names` and dtype `np.float64`. However, if
        input columns are all single precision floats then the output dtype
        will be `np.float32`.
        """
        return pd.DataFrame(dtype=np.float64, columns=self.feature.names)

    def on(self, ens: "Ensemble") -> List[str]:
        return [ens._id_col]

    def __call__(self, time, flux, err, band, *, band_to_calc: str, **kwargs) -> pd.DataFrame:
        """
        Apply a feature extractor to a light curve, concatenating the results over
        all bands.

        Parameters
        ----------
        time : `numpy.ndarray`
            Time values
        flux : `numpy.ndarray`
            Brightness values, flux or magnitudes
        err : `numpy.ndarray`
            Errors for "flux"
        band : `numpy.ndarray`
            Passband names.
        band_to_calc : `str` or `int` or `None`
            Name of the passband to calculate features for, usually a string
            like "g" or "r", or an integer. If None, then features are
            calculated for all sources - band is ignored.
        **kwargs : `dict`
            Additional keyword arguments to pass to the feature extractor.

        Returns
        -------
        features : pandas.DataFrame
            Feature values for each band, dtype is a common type for input arrays.
        """

        # Select passband to calculate
        if band_to_calc is not None:
            band_mask = band == band_to_calc
            time, flux, err = (a[band_mask] for a in (time, flux, err))

        # Sort inputs by time if not already sorted
        if not kwargs.get("sorted", False):
            sort_idx = np.argsort(time)
            time, flux, err, band = (a[sort_idx] for a in (time, flux, err, band))
            # Now we can update the kwargs for better performance
            kwargs = kwargs.copy()
            kwargs["sorted"] = True

        # Convert the numerical arrays to a common dtype
        dtype = np.find_common_type([a.dtype for a in (time, flux, err)], [])
        time, flux, err = (a.astype(dtype) for a in (time, flux, err))

        values = self.feature(time, flux, err, **kwargs)

        series = pd.Series(dict(zip(self.feature.names, values)))
        return series
