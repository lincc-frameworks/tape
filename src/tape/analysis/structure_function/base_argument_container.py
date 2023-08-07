from dataclasses import dataclass
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
            all measurements are taken at equi-distant times. By default `False`.
        random_seed: `int`, optional
            Used when randomly sampling lightcurves to ensure reproducibility.
            By default None.
        equally_weight_lightcurves: `bool`, optional
            Used to ensure that no lightcurves completely dominate the structure
            function calculation. For instance if lightcurve LC_1 has N=10
            observations and LC_2 has N=100, setting `equally_weight_lightcurves=True`
            will calculate all of the time and flux differences (45 for LC_1 and
            4950 for LC_2), then randomly sample (without replacement) 45 from
            LC_2 when calculating the Structure Function. Note that to bin the
            results, we would use the bins calculated based on the 4950
            differences or the user provided bins.
            By default `False`.
        number_lightcurve_samples: `int`, optional
            Used to specify the number of time and flux differences to select
            from a lightcurve. For Structure Function calculators that inherit
            from `StructureFunctionCalculator` and do not implement their own
            `_equally_weight_lightcurves` method, the value defined here will
            only be used when `equally_weight_lightcurves = True`.
            If a value is not provided here, then the default number of
            lightcurve samples will be equal to the least number of differences
            in the available lightcurves. By default None.
        estimate_err: `bool`, optional
            Specifies if the structure function errors are to be estimated,
            via bootstraping the sample and taking note of 16 and 84 percentile
            of the resulting distribution
        calculation_repetitions: `int`, optional
            Specifies the number of times to repeat the structure function
            calculation. Typically this would be used when setting
            `estimate_err = True`. By default 1 when not estimating errors,
            and 100 when estimating errors.
        lower_error_quantile: `float`, optional
            When calculation_repetitions > 1 we will calculate the
            `lower_error_quantile` and `upper_error_quantile` quantiles of the
            results of the structure function calculation and report the
            difference/2 as 1_sigma_error. Value must be between 0 and 1.
            By default 0.16.
        upper_error_quantile: `float`, optional
            When calculation_repetitions > 1 we will calculate the
            `lower_error_quantile` and `upper_error_quantile` quantiles of the
            results of the structure function calculation and report the
            difference/2 as 1_sigma_error. Value must be between 0 and 1.
            By default 0.84.
        report_upper_lower_error_separately: `bool`, optional
            When true, upper_error_quantile - median and median - lower_error_quantile
            will be reported separately. Note, when using `Ensemble.batch`,
            additional metadata information will need to be provided.
            By default False.

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
    random_seed: int = None
    equally_weight_lightcurves: bool = False
    number_lightcurve_samples: int = None
    estimate_err: bool = False
    if estimate_err:
        calculation_repetitions: int = 100
    else:
        calculation_repetitions: int = 1
    lower_error_quantile: float = 0.16
    upper_error_quantile: float = 0.84
    report_upper_lower_error_separately: bool = False

    def __post_init__(self):
        # Nothing here yet
        return
