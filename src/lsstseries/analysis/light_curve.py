import numpy as np


class LightCurve:
    """This base class is meant to support various analysis routines and be
    extended as needed. (Hence it's location in the `analysis` package.)

    The base class ensures that the data for a single lightcurve is well formed.
    Namely that the input data is all of the same length, with NaN's removed and
    that there are enough observations to perform a given analysis.
    """

    def __init__(
        self, times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, minimum_observations: int = 0
    ):
        """Calls the processing functions to clean and validate the input data.
        All of the input arrays should have the same length.

        Parameters
        ----------
        times : np.ndarray
            The timestamps values of the data.
        fluxes : np.ndarray
            The flux/magnitude values of the data.
        errors : np.ndarray
            The error in the measurements.
        minimum_observations : int, optional
            The minimum number of data points needed to successfully process
            the lightcurve, by default 0
        """
        # The primary data
        self._times = times
        self._fluxes = fluxes
        self._errors = errors
        self._minimum_observations = minimum_observations

        self._process_input_data()

    def _process_input_data(self):
        """Cleaning and validation occurs here, ideally by calling
        sub-methods for specific checks and cleaning tasks.
        """
        self._check_input_data_size_is_equal()  # make sure arrays the same length
        self._filter_nans()
        self._check_input_data_length_is_sufficient()  # make sure we have enough values still

    def _filter_nans(self):
        """Mask out any NaN values from time, flux and error arrays"""

        t_mask = np.isnan(self._times)
        f_mask = np.isnan(self._fluxes)
        e_mask = np.isnan(self._errors)

        # np.logical_or takes at most 2 arguments, using reduce will collapse
        # the first two arguments into one, and apply it to the third.
        nan_mask = np.logical_or.reduce((t_mask, f_mask, e_mask))

        self._times = self._times[~nan_mask]
        self._fluxes = self._fluxes[~nan_mask]
        self._errors = self._errors[~nan_mask]

    def _check_input_data_size_is_equal(self):
        """Make sure that the three input np.arrays have the same size"""
        if self._times.size != self._fluxes.size or self._times.size != self._errors.size:
            raise ValueError("Input np.arrays are expected to have the same size.")

    def _check_input_data_length_is_sufficient(self):
        """Make sure that we have enough data after cleaning and filtering
        to be able to perform Structure Function calculations.
        """
        if self._times.size < self._minimum_observations:
            raise ValueError("Too few observations provided to create `LightCurve`.")
