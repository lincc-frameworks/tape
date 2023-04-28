from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class StructureFunctionArgumentContainer:
    """This is a container class for less-often-used configuration arguments for
    Structure Function.

    Args:
        band : `list[str]`, optional
            The band information for each lightcurve to be processed. If a value
            is provided for the `band` argument when calling `calc_sf2` directly,
            this value will be ignored. By default None.
        lc_id : `int` or `list[int]`, optional
            The object id for each lightcurve to be processed. If a value is
            provided for the `lc_id` argument when calling `calc_sf2` directly,
            this value will be ignored. By default None.
        sf_method : {'basic'}, optional
            This defined which Structure Function calculation method to use when
            processing lightcurves. By default 'basic'.
        band_to_calc : `str` or `list[str]`, optional
            The bands of the input lightcurves to operate on. If no value is
            provided, all input bands will be processed. By default None.
        combine : `bool`, optional
            Boolean to determine whether structure function is computed for each
            light curve independently (combine=False; the default), or computed
            for all light curves together (combine=True). By default False.
        bins : `list[floats]`, optional
            Manually provided time difference bins, if not provided then bins
            are computed using the `bin_method` defined. By default None.
        bin_method : {'size', 'length', 'loglength'}, optional
            The binning method to apply, choices of 'size'; which seeks an even
            distribution of samples per bin using quantiles, 'length'; which
            creates bins of equal length in time and 'loglength'; which creates
            bins of equal length in log time. By default 'size'.
        bin_count_target : `int`, optional
            Target number of samples per time difference bin. By default 100.
        ignore_timestamps : `bool`, optional
            Used to ignore the use of any provided timestamps, instead assuming
            all measurements are taken at equi-distant times.

    Notes:
        It may be necessary to extend this dataclass to support new Structure
        Function calculation methods. When doing so, all new properties *must*
        have a default value defined.
    """

    band: List[str] = None
    lc_id: Union[int, List[int]] = None
    sf_method: str = "basic"
    band_to_calc: Union[str, List[str]] = None
    combine: bool = False
    bins: List[float] = None
    bin_method: str = "size"
    bin_count_target: int = 100
    ignore_timestamps: bool = False

    def __post_init__(self):
        # Nothing here yet
        return
