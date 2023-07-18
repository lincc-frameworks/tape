import numpy as np

from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structure_function.base_calculator import StructureFunctionCalculator

# MacLeod et al. 2012, Erratum 2014ApJ...782..119M
CONVERSION_TO_SIGMA = 0.74


class Macleod2012StructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    MacLeod et al. 2012, 2012ApJ...753..106M [https://arxiv.org/abs/1112.0679]
    MacLeod et al. 2012, Erratum 2014ApJ...782..119M

    `SF_obs(deltaT) = 0.74 * IQR`

    Where `IQR` is the interquartile range between 25% and 75% of the sorted
    (y(t) - y(t+delta_t)) distribution.

    Note that the return value is structure function squared.

    Additional references:
    Kozlowski 2016, 2016ApJ...826..118K [https://arxiv.org/abs/1604.05858]
    """

    def calculate(self):
        dts, sfs = self._calculate_binned_statistics(statistic_to_apply=self.calculate_iqr_sf2_statistic)

        return dts, sfs

    def calculate_iqr_sf2_statistic(self, input):
        """For a given set of binned metrics (in this case delta fluxes) calculate
        the interquartile range.

        Parameters
        ----------
        input : `np.ndarray` (N,)
            The delta flux values that correspond to a given delta time bin.

        Returns
        -------
        float
            Result of calculation defined in MacLeod et al. 2012, Erratum 2014ApJ...782..119M:

            `SF(dt) = 0.74 * IQR`
        """
        # calculate interquartile range between 25% and 75%.
        iqr = np.subtract(*np.percentile(input, [75, 25]))

        return (CONVERSION_TO_SIGMA * iqr) ** 2

    @staticmethod
    def name_id() -> str:
        return "macleod_2012"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
