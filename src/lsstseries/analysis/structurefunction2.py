import numpy as np
import pandas as pd

from lsstseries.analysis.structure_function import SF_METHODS
from lsstseries.analysis.structure_function.light_curve import StructureFunctionLightCurve


def calc_sf2(time, flux, err=None, band=None, lc_id=None, sf_method="basic", argument_container=None):
    """Calculate structure function squared using one of a variety of structure
    function calculation methods defined by the input argument `sf_method`, or
    in the argument container object.


    Parameters
    ----------
    time : `numpy.ndarray` (N,) or `None`
        Array of times when measurements were taken. If all array values are
        `None` or if a scalar `None` is provided, then equidistant time between
        measurements is assumed.
    flux : `numpy.ndarray` (N,)
        Array of flux/magnitude measurements.
    err : `numpy.ndarray` (N,), `float`, or `None`, optional
        Array of associated flux/magnitude errors. If a scalar value is provided
        we assume that error for all measurements. If `None` is provided, we
        assume all errors are 0. By default None
    band : `numpy.ndarray` (N,), optional
        Array of associated band labels, by default None
    lc_id : `numpy.ndarray` (N,), optional
        Array of lightcurve ids per data point. By default None
    sf_method : str, optional
        The structure function calculation method to be used, by default "basic".
    argument_container : StructureFunctionArgumentContainer, optional
        Container object for additional configuration options, by default None.

    Returns
    -------
    sf2 : `pandas.DataFrame`
        Structure function squared for each of input bands.

    Notes
    ----------
    In case that no value for `band_to_calc` is passed, the function is
    executed on all available bands in `band`.
    """

    argument_container = _create_arg_container_if_needed(sf_method, argument_container)

    band = _validate_band(band, flux, argument_container)

    lc_id = _validate_lightcurve_id(lc_id, flux, argument_container)

    sf_method = _validate_sf_method(sf_method, argument_container)

    _validate_argument_container(argument_container)

    unq_band = np.unique(band)
    unq_ids = np.unique(lc_id)

    band_to_calc = argument_container.band_to_calc
    if band_to_calc is None:
        band_to_calc = unq_band
    if isinstance(band_to_calc, str):
        band_to_calc = [band_to_calc]

    assert hasattr(band_to_calc, "__iter__") is True

    ids = []
    dts = []
    bands = []
    sf2s = []
    for b in band_to_calc:
        if b in unq_band:
            band_mask = band == b

            # Mask on band
            times = _extract_time(time=time, band_mask=band_mask, argument_container=argument_container)

            errors = _extract_error(err=err, band_mask=band_mask)

            fluxes = np.array(flux)[band_mask]
            lc_ids = np.array(lc_id)[band_mask]

            # Create stacks of critical quantities, indexed by id
            id_masks = [lc_ids == lc for lc in unq_ids]

            lightcurves = []
            for mask in id_masks:
                if sum(mask) > 1:
                    sf_lc = StructureFunctionLightCurve(
                        times=times[mask], fluxes=fluxes[mask], errors=errors[mask]
                    )
                    lightcurves.append(sf_lc)

            #! This doesn't seem to behave the way I expect when I run the Ensemble
            #! tests. I would expect this to just pass through, but it gets caught
            #! in the lambda functions in `Ensemble.batch` because there is nothing
            #! to concatenate.

            if len(lightcurves) == 0:
                continue

            sf_calculator = SF_METHODS[sf_method](lightcurves, argument_container)

            rng = np.random.default_rng(argument_container.random_seed)
            for _ in range(argument_container.calculation_repetitions):
                if argument_container.equally_weight_lightcurves:
                    sf_calculator._equally_weight_lightcurves(random_generator=rng)

                res = sf_calculator.calculate()

                # ! Need to do some more work to aggregate the results here
                # ! Take the median, also calculate 16/83 quantiles to report as 1 sigma.

            res_ids = [[str(unq_ids[i])] * len(arr) for i, arr in enumerate(res[0])]
            res_bands = [[b] * len(arr) for arr in res[0]]

            ids.append(np.hstack(res_ids))
            bands.append(np.hstack(res_bands))
            dts.append(np.hstack(res[0]))
            sf2s.append(np.hstack(res[1]))

    idstack = np.hstack(ids)
    if argument_container.combine:
        idstack = ["combined"] * len(idstack)

    sf2_df = pd.DataFrame(
        {"lc_id": idstack, "band": np.hstack(bands), "dt": np.hstack(dts), "sf2": np.hstack(sf2s)}
    )
    return sf2_df


def _create_arg_container_if_needed(sf_method, argument_container):
    """If no argument container was provided, we'll create a default container
    with the default values using the value of `sf_method` to determine the correct
    type.

    Parameters
    ----------
    sf_method : str
        The structure function calculation method to be used.
    argument_container : StructureFunctionArgumentContainer
        Container object for additional configuration options.

    Returns
    -------
    StructureFunctionArgumentContainer
        Container object for additional configuration options.
    """

    if argument_container is None:
        argument_container_type = SF_METHODS[sf_method].expected_argument_container()
        argument_container = argument_container_type()
        argument_container.sf_method = sf_method

    return argument_container


def _validate_band(band, flux, argument_container):
    """The argument `band` can be set as an input argument or inside the
    `argument_container`. If `band` is provided in the function call with
    non-default values, we'll use those. Otherwise, we'll look inside
    `argument_container` and use the values found there. If we still can't find
    non-`None` values, we'll create a minimal fallback array of 1's with length
    equal to the input `flux` array.

    Parameters
    ----------
    band : `numpy.ndarray` (N,)
        Array of associated band labels.
    flux : `numpy.ndarray` (N,)
        Array of flux/magnitude measurements.
    argument_container : StructureFunctionArgumentContainer
        Container object for additional configuration options.

    Returns
    -------
    `numpy.ndarray` (N,)
        The band information to be used for structure function calculations.

    Raises
    ------
    ValueError
        If the provided band and flux arrays have different lengths, raise an
        exception.
    """

    if band is None:
        band = argument_container.band

    # if band is still `None`, create a fake array of bands. We'll use numpy int8
    # values to use as little memory as possible.
    if band is None:
        band = np.zeros(len(flux), dtype=np.int8)

    if len(band) != len(flux):
        raise ValueError(
            "Value of `band` should be `None` or array with the same length as the `flux` array."
        )

    return band


def _validate_lightcurve_id(lc_id, flux, argument_container):
    """The argument `lc_id` can be set as an input argument or inside the
    `argument_container`. If `lc_id` is provided in the function call with
    non-default values, we'll use those. Otherwise, we'll look inside
    `argument_container` and use the values found there. If we still can't find
    non-`None` values, we'll create a minimal fallback array of 1's with length
    equal to the input `flux` array.

    Parameters
    ----------
    lc_id : `numpy.ndarray` (N,), optional
        Array of lightcurve ids per data point.
    flux : `numpy.ndarray` (N,)
        Array of flux/magnitude measurements.
    argument_container : StructureFunctionArgumentContainer
        Container object for additional configuration options.

    Returns
    -------
    `numpy.ndarray` (N,)
        The lightcurve id information to be used for structure function
        calculations.

    Raises
    ------
    ValueError
        If the provided lc_id and flux arrays have different lengths, raise an
        exception.
    """

    if lc_id is None:
        lc_id = argument_container.lc_id

    # if the light curve value is still `None`, create an array of all 0's.
    # Using numpy.int8 values to save memory.
    if lc_id is None:
        lc_id = np.zeros(len(flux), dtype=np.int8)

    if len(lc_id) != len(flux):
        raise ValueError(
            "Value of `lc_id` should be `None` or array with the same length as the `flux` array."
        )

    return lc_id


def _validate_sf_method(sf_method, argument_container):
    """The argument `sf_method` can be set as an input argument or inside the
    `argument_container`. If `sf_method` is provided in the function call with
    a non-default value, we'll use that. If the default value for `sf_method`
    ("basic") has been provided, we'll check `argument_container` to see if the
    user provided a different `sf_method` value there.

    Parameters
    ----------
    sf_method : str, optional
        The structure function calculation method to be used, by default "basic".
    argument_container : StructureFunctionArgumentContainer
        Container object for additional configuration options.

    Returns
    -------
    `str`
        The band information to be used for structure function calculations.
    """

    if sf_method == "basic":
        sf_method = argument_container.sf_method

    if sf_method not in SF_METHODS:
        raise ValueError("Unknown structure function calculator method: " + str(sf_method))

    return sf_method


def _validate_argument_container(argument_container):
    """Raise an exception if this type of argument container does not match the
    type of argument container required by the structure function calculation
    method
    """

    if type(argument_container) is not SF_METHODS[argument_container.sf_method].expected_argument_container():
        raise TypeError("Argument container does not match Structure Function calculator method")


def _extract_time(time, band_mask, argument_container):
    """This will process the input time values, apply a band mask and return a
    numpy array of timestamps.

    Parameters
    ----------
    time : `numpy.ndarray` (N,) or `None`
        Array of times when measurements were taken. If all array values are
        `None` or if a scalar `None` is provided, then equidistant time between
        measurements is assumed.
    band_mask : `numpy mask` (N,)
        Array of associated band labels, by default None
    argument_container : StructureFunctionArgumentContainer, optional
        Container object for additional configuration options, by default None.
    """

    times = None

    # if the user passed in a scalar `None` value, or if explicitly ignoring
    # timestamps using the argument_container flag, create a numpy array
    # with a single `None` element. Otherwise assume the user passed an
    # array of timestamps to be masked with `band_mask`.
    # Note: some or all timestamps could be `None`.
    if time is None or argument_container.ignore_timestamps:
        times = np.array(None)
    else:
        times = np.array(time)[band_mask]

    # if all elements in `times` are `None`, we assume equidistant times
    # between measurements. To do so, we'll create an array of integers
    # from 0 to N-1 where N is the number of flux values for this band.
    if np.all(np.equal(times, None)):
        times = np.arange(sum(band_mask), dtype=int)

    return times


def _extract_error(err, band_mask):
    """Process the input err value(s), apply a band_mask, and return a numpy
    array of error values.

    Parameters
    ----------
    err : `numpy.ndarray` (N,), `float`, or `None`
        Array of associated flux/magnitude errors. If a scalar value is provided
        we assume that error for all measurements. If `None` is provided, we
        assume all errors are 0. By default None
    band_mask : `numpy mask` (N,)
        Array of associated band labels, by default None
    """

    errors = None
    # assume all errors are 0 if `None` is provided
    if err is None:
        errors = np.zeros(sum(band_mask))

    # assume the same error for all measurements if a scalar value is
    # provided
    elif np.isscalar(err):
        errors = np.ones(sum(band_mask)) * err

    # otherwise assume one error value per measurement
    else:
        errors = np.array(err)[band_mask]

    return errors
