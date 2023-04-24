from typing import List

import numpy as np

from lsstseries.analysis.structure_function_argument_containers.structure_function_argument_container import (
    StructureFunctionArgumentContainer,
)


class StructureFunctionCalculator:
    """This is the base class from which all other Structure Function calculators
    will inherit
    """

    def __init__(
        self,
        time: List[float],
        flux: List[float],
        err: List[float],
        argument_container: StructureFunctionArgumentContainer,
    ):
        # do the work here to initialize

        self._time = time
        self._flux = flux
        self._err = err
        self._argument_container = argument_container

        self._binning_method = argument_container.bin_method
        self._bin_count_target = argument_container.bin_count_target
        self._dts = []
        return

    def calculate(self):
        """Abstract method that must be implemented by the child class."""
        raise (NotImplementedError, "Must be implemented by the child class")

    def _bin_dts(self, dts):
        """Bin an input array of delta times (dt). Supports several binning
        schemes.

        Parameters
        ----------
        dts : `numpy.ndarray` (N,)
            1-d array of delta times to bin

        Returns
        -------
        bins : `numpy.ndarray` (N,)
            The returned bins array.
        """

        num_bins = int(np.ceil(len(dts) / self._bin_count_target))
        dts_unique = np.unique(dts)
        if self._binning_method == "size":
            quantiles = np.linspace(0.0, 1.0, num_bins + 1)
            self._bins = np.quantile(dts_unique, quantiles)

        elif self._binning_method == "length":
            # Compute num_bins equally spaced bins.
            min_val = dts_unique.min()
            max_val = dts_unique.max()
            self._bins = np.linspace(min_val, max_val, num_bins + 1)

            # Extend the start of the first bin by 0.1% of the range to
            # include the first element. Note this is also done to match
            # Panda's cut function.
            self._bins[0] -= 0.001 * (max_val - min_val)

        elif self._binning_method == "loglength":
            log_vals = np.log(dts_unique)

            # Compute num_bins equally spaced bins in log space.
            min_val = log_vals.min()
            max_val = log_vals.max()
            self._bins = np.linspace(min_val, max_val, num_bins + 1)

            # Extend the start of the first bin by 0.1% of the range to
            # include the first element. Note this is also done to match
            # Panda's cut function.
            self._bins[0] -= 0.001 * (max_val - min_val)

        else:
            raise ValueError(f"Method '{self._binning_method}' not recognized")

    @staticmethod
    def name_id():
        raise (NotImplementedError, "Must be implemented as a static method by the child class")

    @staticmethod
    def expected_argument_container():
        raise (NotImplementedError, "Must be implemented as a static method by the child class")
