from scipy.stats import binned_statistic
import numpy as np


def calc_sf2(time, flux, err, bins, band, band_to_calc=None):
    """Compute structure function squared on one or many bands

    Parameters
    ----------
    time : `numpy.ndarray` (N,)
        Array of times when measurements were taken.
    flux : `numpy.ndarray` (N,)
        Array of flux/magnitude measurements.
    err : `numpy.ndarray` (N,)
        Array of associated flux/magnitude errors.
    bins : `numpy.ndarray` (N,)
        Array of time bins in which to calculate structure function
    band : `numpy.ndarray` (N,)
        Array of associated band labels,
    band_to_calc : `str` or `list` of `str`
        Bands to calculate StetsonJ on. Single band descriptor, or list
        of such descriptors.

    Returns
    -------

    Notes
    ----------
    In case that no value for `band_to_calc` is passed, the function is
    executed on all available bands in `band`.
    """

    unq_band = np.unique(band)

    if band_to_calc is None:
        band_to_calc = unq_band
    if isinstance(band_to_calc, str):
        band_to_calc = [band_to_calc]

    assert hasattr(band_to_calc, "__iter__") is True

    sf2 = {}
    for b in band_to_calc:
        if b in unq_band:
            mask = band == b
            times = time[mask]
            fluxes = flux[mask]
            errors = err[mask]
            sf2[b] = _sf2_single(times, fluxes, errors, bins)
        else:
            sf2[b] = np.nan
    return sf2


def _sf2_single(times, fluxes, errors, bins):
    """Calculate structure function squared

    Calculate structure function squared from the avaliable data. This is
    AGN-type definition, i.e., variance of the distribution of the
    differences of measurments at different times

    Parameters
    ----------
    time : `np.array` [`float`]
        Times at which the measurment was conducted.
    y : `np.array` [`float`]
        Measurment values
    yerr : `np.array` [`float`]
        Measurment errors.
    bins :  `np.array` [`float`]:
        Edges of time bin in which data is grouped together.

    Returns
    ----------
    bin_mean: `np.array` [`float`]
        Mean of the time bins.
    SF: `np.array` [`float`],
        Structure function.

    TODO:
    ----------
    - allow user to not specify bins - automatically assume ``reasonable bins''
    - allow user to not specify times - assume equidistant times
    - allow multiple inputs, with same <t> at once
    - ability to create SF2 from multiple lightcurves at once (ensamble)
    - allow for different definitions of SF2
    """

    times = times.values
    fluxes = fluxes.values
    errors = errors.values

    # compute dt (difference of times) and
    # dm (difference of magnitudes) for all gaps
    # d_times - difference of times
    dt_matrix = times.reshape((1, times.size)) - times.reshape((times.size, 1))
    d_times = dt_matrix[dt_matrix > 0].flatten().astype(np.float16)

    # d_fluxes - difference of fluxes
    df_matrix = fluxes.reshape((1, fluxes.size)) - fluxes.reshape((fluxes.size, 1))
    d_fluxes = df_matrix[dt_matrix > 0].flatten().astype(np.float16)

    # err^2 - errors squared
    err2_matrix = errors.reshape((1, errors.size))**2 \
        + errors.reshape((errors.size, 1))**2
    err2s = err2_matrix[dt_matrix > 0].flatten().astype(np.float16)

    # corrected each pair of observations
    cor_flux2 = d_fluxes**2 - err2s

    # structure function at specific dt
    # the line below will throw error if the bins are not covering the whole range
    SFs, bin_edgs, _ = binned_statistic(d_times, cor_flux2, 'mean', bins)

    return (bin_edgs[0:-1] + bin_edgs[1:])/2, SFs
