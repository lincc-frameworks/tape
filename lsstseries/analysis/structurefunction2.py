from scipy.stats import binned_statistic
import numpy as np
import pandas as pd


def calc_sf2(lc_id, time, flux, err, band, bins=None,
             band_to_calc=None, combine=False, method='size', sthresh=100):
    """Compute structure function squared on one or many bands

    Parameters
    ----------
    lc_id : 'numpy.ndarray' (N,)
        Array of lightcurve ids per data point.
    time : `numpy.ndarray` (N,)
        Array of times when measurements were taken.
    flux : `numpy.ndarray` (N,)
        Array of flux/magnitude measurements.
    err : `numpy.ndarray` (N,)
        Array of associated flux/magnitude errors.
    band : `numpy.ndarray` (N,)
        Array of associated band labels,
    bins : `numpy.ndarray`
        Manually provided bins, if not provided then bins are computed using
        the `method` kwarg
    band_to_calc : `str` or `list` of `str`
        Bands to calculate structure function on. Single band descriptor,
        or list of such descriptors.
    combine : 'bool'
        Boolean to determine whether structure function is computed for each
        light curve independently (combine=False), or computed for all light
        curves together (combine=True).
    method : 'str'
        The binning method to apply, choices of 'size'; which seeks an even
        distribution of samples per bin using quantiles, 'length'; which
        creates bins of equal length in time and 'loglength'; which creates
        bins of equal length in log time.
    sthresh : 'int'
        Target number of samples per bin.

    Returns
    -------
    sf2 : `pandas.DataFrame`
        Structure function squared for each of input bands.

    Notes
    ----------
    In case that no value for `band_to_calc` is passed, the function is
    executed on all available bands in `band`.
    """

    unq_band = np.unique(band)
    unq_ids = np.unique(lc_id)

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
            times = np.array(time)[band_mask]
            fluxes = np.array(flux)[band_mask]
            errors = np.array(err)[band_mask]
            lc_ids = np.array(lc_id)[band_mask]

            # Create stacks of critical quantities, indexed by id
            id_masks = [lc_ids == lc for lc in unq_ids]
            times_2d = [times[mask] for mask in id_masks]
            fluxes_2d = [fluxes[mask] for mask in id_masks]
            errors_2d = [errors[mask] for mask in id_masks]

            res = _sf2_single(times_2d, fluxes_2d, errors_2d, bins=bins,
                              combine=combine, method=method, sthresh=sthresh)

            res_ids = [[str(unq_ids[i])]*len(arr) for i, arr in enumerate(res[0])]
            res_bands = [[b]*len(arr) for arr in res[0]]

            ids.append(np.hstack(res_ids))
            bands.append(np.hstack(res_bands))
            dts.append(np.hstack(res[0]))
            sf2s.append(np.hstack(res[1]))

    if combine:
        idstack = ["combined"] * len(np.hstack(ids))
    else:
        idstack = np.hstack(ids)
    sf2_df = pd.DataFrame({"lc_id": idstack, "band": np.hstack(bands),
                           "dt": np.hstack(dts), "sf2": np.hstack(sf2s)})
    return sf2_df


def _sf2_single(times, fluxes, errors, bins=None,
                combine=False, method='size', sthresh=100):
    """Calculate structure function squared

    Calculate structure function squared from the available data. This is
    AGN-type definition, i.e., variance of the distribution of the
    differences of measurements at different times

    Parameters
    ----------
    times : `np.array` [`float`]
        Times at which the measurements were conducted.
    fluxes : `np.array` [`float`]
        Measurements values
    errors : `np.array` [`float`]
        Measurements errors.
    bins : `np.array` [`float`]
        Manually provided bins, if not provided then bins are computed using
        the `method` kwarg
    combine : 'bool'
        Boolean to determine whether structure function is computed for each
        light curve independently (combine=False), or computed for all light
        curves together (combine=True).
    method : 'str'
        The binning method to apply, choices of 'size'; which seeks an even
        distribution of samples per bin using quantiles, 'length'; which
        creates bins of equal length in time and 'loglength'; which creates
        bins of equal length in log time.
    sthresh : `int`
        Target number of samples per bin.

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

    d_times_all = []
    cor_flux2_all = []
    for lc_idx in range(len(times)):
        lc_times = times[lc_idx]
        lc_fluxes = fluxes[lc_idx]
        lc_errors = errors[lc_idx]

        # mask out any nan values
        t_mask = np.isnan(lc_times)
        f_mask = np.isnan(lc_fluxes)
        e_mask = np.isnan(lc_errors)  # always mask out nan errors?
        lc_mask = np.logical_or(t_mask, f_mask, e_mask)

        lc_times = lc_times[~lc_mask]
        lc_fluxes = lc_fluxes[~lc_mask]
        lc_errors = lc_errors[~lc_mask]

        # compute d_times (difference of times) and
        # d_fluxes (difference of magnitudes, i.e., fluxes) for all gaps
        # d_times - difference of times
        dt_matrix = lc_times.reshape((1, lc_times.size)) - lc_times.reshape((lc_times.size, 1))
        d_times = dt_matrix[dt_matrix > 0].flatten()

        # d_fluxes - difference of fluxes
        df_matrix = lc_fluxes.reshape((1, lc_fluxes.size)) - lc_fluxes.reshape((lc_fluxes.size, 1))
        d_fluxes = df_matrix[dt_matrix > 0].flatten()

        # err^2 - errors squared
        err2_matrix = lc_errors.reshape((1, lc_errors.size))**2 \
            + lc_errors.reshape((lc_errors.size, 1))**2
        err2s = err2_matrix[dt_matrix > 0].flatten()

        # corrected each pair of observations
        cor_flux2 = d_fluxes**2 - err2s

        # build stack of times and fluxes
        d_times_all.append(d_times)
        cor_flux2_all.append(cor_flux2)

    # combining treats all lightcurves as one when calculating the structure function
    if combine and len(times) > 1:
        d_times_all = np.hstack(np.array(d_times_all, dtype='object'))
        cor_flux2_all = np.hstack(np.array(cor_flux2_all, dtype='object'))

        # binning
        if bins is None:
            bins = _bin_dts(d_times_all, method=method, sthresh=sthresh)

        # structure function at specific dt
        # the line below will throw error if the bins are not covering the whole range
        sfs, bin_edgs, _ = binned_statistic(d_times_all, cor_flux2_all, 'mean', bins)
        return [(bin_edgs[0:-1] + bin_edgs[1:])/2], [sfs]
    # Not combining calculates structure function for each light curve independently
    else:
        # may want to raise warning if len(times) <=1 and combine was set true
        sfs_all = []
        t_all = []
        for lc_idx in range(len(d_times_all)):
            if len(d_times_all[lc_idx]) > 1:

                # binning
                bins = _bin_dts(d_times_all[lc_idx], method=method, sthresh=sthresh)
                sfs, bin_edgs, _ = binned_statistic(d_times_all[lc_idx], cor_flux2_all[lc_idx], 'mean', bins)
                sfs_all.append(sfs)
                t_all.append((bin_edgs[0:-1] + bin_edgs[1:])/2)
            else:
                sfs_all.append(np.array([]))
                t_all.append(np.array([]))
        return t_all, sfs_all


def _bin_dts(dts, method='size', sthresh=100):
    """Bin an input array of delta times (dt). Supports several binning
    schemes.

    Parameters
    ----------
    dts : 'numpy.ndarray' (N,)
        1-d array of delta times to bin
    method : 'str'
        The binning method to apply, choices of 'size'; which seeks an even
        distribution of samples per bin using quantiles, 'length'; which
        creates bins of equal length in time and 'loglength'; which creates
        bins of equal length in log time.
    sthresh : 'int'
        Target number of samples per bin.

    Returns
    -------
    bins : 'numpy.ndarray' (N,)
        The returned bins array.
    """

    if method == 'size':
        quantiles = int(np.ceil(len(dts)/sthresh))
        _, bins = pd.qcut(dts, q=quantiles, retbins=True, duplicates='drop')
        return bins

    elif method == 'length':
        nbins = int(np.ceil(len(dts)/sthresh))
        _, bins = pd.cut(dts, bins=nbins, retbins=True, duplicates='drop')
        return bins

    elif method == 'loglength':
        nbins = int(np.ceil(len(dts)/sthresh))
        _, bins = pd.cut(np.log(dts), bins=nbins, retbins=True, duplicates='drop')
        return np.exp(bins)

    else:
        raise ValueError(f"Method '{method}' not recognized")
