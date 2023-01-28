from scipy.stats import binned_statistic
import numpy as np
import pandas as pd


def calc_sf2(lc_id, time, flux, err, bins, band,
             band_to_calc=None, combine=False):
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
    bins : `numpy.ndarray` (N,)
        Array of time bins in which to calculate structure function
    band : `numpy.ndarray` (N,)
        Array of associated band labels,
    band_to_calc : `str` or `list` of `str`
        Bands to calculate structure function on. Single band descriptor,
        or list of such descriptors.

    Returns
    -------
    sf2 : `dict`
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

            res = _sf2_single(times_2d, fluxes_2d, errors_2d, bins, combine=combine)

            res_ids = [[str(unq_ids[i])]*len(arr) for i, arr in enumerate(res[0])]
            res_bands = [[b]*len(arr) for arr in res[0]]

            ids.append(np.hstack(res_ids))
            bands.append(np.hstack(res_bands))
            dts.append(np.hstack(res[0]))
            sf2s.append(np.hstack(res[1]))

    sf2_df = pd.DataFrame({"lc_id": np.hstack(ids), "band": np.hstack(bands),
                           "dt": np.hstack(dts), "sf2": np.hstack(sf2s)})
    sf2_df = sf2_df.set_index('lc_id')
    return sf2_df


def _sf2_single(times, fluxes, errors, bins, combine=False):
    """Calculate structure function squared

    Calculate structure function squared from the available data. This is
    AGN-type definition, i.e., variance of the distribution of the
    differences of measurements at different times

    Parameters
    ----------
    time : `np.array` [`float`]
        Times at which the measurements were conducted.
    y : `np.array` [`float`]
        Measurements values
    yerr : `np.array` [`float`]
        Measurements errors.
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

    d_times_all = []
    cor_flux2_all = []
    for lc_idx in range(len(times)):
        lc_times = times[lc_idx]
        lc_fluxes = fluxes[lc_idx]
        lc_errors = errors[lc_idx]

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
    if combine:
        d_times_all = np.hstack(np.array(d_times_all, dtype='object'))
        cor_flux2_all = np.hstack(np.array(cor_flux2_all, dtype='object'))

        # structure function at specific dt
        # the line below will throw error if the bins are not covering the whole range
        sfs, bin_edgs, _ = binned_statistic(d_times_all, cor_flux2_all, 'mean', bins)
        return [(bin_edgs[0:-1] + bin_edgs[1:])/2], [sfs]
    # Not combining calculates structure function for each light curve independently
    else:
        sfs_all = []
        t_all = []
        for lc_idx in range(len(d_times_all)):
            if len(d_times_all[lc_idx]) > 1:
                sfs, bin_edgs, _ = binned_statistic(d_times_all[lc_idx], cor_flux2_all[lc_idx], 'mean', bins)
                sfs_all.append(sfs)
                t_all.append((bin_edgs[0:-1] + bin_edgs[1:])/2)
            else:
                sfs_all.append(np.array([]))
                t_all.append(np.array([]))
        return t_all, sfs_all


def _patch_empty(res):
    """Patches empty result with appropriate dt_bins and sf nans arrays"""

    # first look for populated bins to use as truth array
    not_found = True
    i = 0
    while not_found:
        if len(res[0][i]) > 0:
            dt_bins = res[0][i]
            not_found = False
        else:
            i += 1

    # find indices of empty results and patch in the bins/nans
    missing = np.where(np.array([len(tbins) for tbins in res[0]]) == 0)[0]
    for idx in missing:
        res[0][idx] = dt_bins
        res[1][idx] = np.array([np.nan]*len(dt_bins))

    return res
