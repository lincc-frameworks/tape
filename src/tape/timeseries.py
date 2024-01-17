import pandas as pd

from tape.analysis.stetsonj import calc_stetson_J
from tape.analysis.structurefunction2 import calc_sf2


class TimeSeries:
    """Represent and analyze Rubin TimeSeries data"""

    def __init__(self, data=None):
        self.data = data
        self.meta = {"id": None}  # metadata dict
        self.colmap = {"time": None, "flux": None, "flux_err": None}  # column mapping

    # I/O
    def from_dict(
        self,
        data_dict,
        time_label="time",
        flux_label="flux",
        err_label="flux_err",
        band_label="band",
    ):
        """Build dataframe from a python dictionary

        Parameters
        ----------
        data_dict : `dict`
            Dictionary contaning the data.
        time_label: `str`
            Name for column containing time information.
        flux_label: `str`
            Name for column containing signal
            (flux, magnitude, etc) information.
        err_label: `str`
            Name for column containing error information.
        band_label: `str`
            Name for column containing filter information.
        """

        try:
            data_dict[band_label]
        except KeyError as exc:
            raise KeyError(f"The indicated label '{band_label}' was not found.") from exc
        index = self._build_index(data_dict[band_label])
        data_dict = {key: data_dict[key] for key in data_dict if key != band_label}
        self.data = pd.DataFrame(data=data_dict, index=index).sort_index()

        labels = [time_label, flux_label, err_label]
        for label, quantity in zip(labels, list(self.colmap.keys())):
            if (quantity == "flux_err") and (label is None):  # flux_err is optional
                continue

            if label in self.data.columns:
                self.colmap[quantity] = label
            else:
                raise KeyError(f"The indicated label '{label}' was not found.")

        return self

    def dropna(self, **kwargs):
        """Handle NaN values, wrapper for pandas.DataFrame.dropna"""
        self.data = self.data.dropna(**kwargs)
        return self

    def from_dataframe(
        self, data, object_id, time_label="time", flux_label="flux", err_label="flux_err", band_label="band"
    ):
        """Loader function for inputing data from a dataframe.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The data for the time serires.
        object_id : `str`
            The ID of the current object.
        time_label: `str`
            Name for column containing time information.
        flux_label: `str`
            Name for column containing signal
            (flux, magnitude, etc) information.
        err_label: `str`
            Name for column containing error information.
        band_label: `str`
            Name for column containing filter information.
        """
        self.data = data
        self.meta["id"] = object_id

        # Index the timeseries on band.
        index = self._build_index(self.data[band_label])
        self.data.index = index

        labels = [time_label, flux_label, err_label]
        for label, quantity in zip(labels, list(self.colmap.keys())):
            if (quantity == "flux_err") and (label is None):  # flux_err is optional
                continue

            if label in self.data.columns:
                self.colmap[quantity] = label
            else:
                raise KeyError(f"The indicated label '{label}' was not found.")

        return self

    @property
    def time(self):
        """Time values stored as a Pandas Series"""
        return self.data[self.colmap["time"]]

    @property
    def flux(self):
        """Flux values stored as a Pandas Series"""
        return self.data[self.colmap["flux"]]

    @property
    def flux_err(self):
        """Flux error values stored as a Pandas Series"""
        if self.colmap["flux_err"] is not None:  # Errors are not mandatory
            return self.data[self.colmap["flux_err"]]
        return None

    @property
    def band(self):
        """Band labels stored as a Pandas Index"""
        return self.data.index.get_level_values("band")

    def _build_index(self, band):
        """Build pandas multiindex from band array"""
        count_dict = {}
        idx = []
        for b in band:
            count = count_dict.get(b, 0)
            idx.append(count)

            # Increment count for this band or insert 1 there wasn't an ongoing count.
            count_dict[b] = count + 1
        tuples = zip(band, idx)
        index = pd.MultiIndex.from_tuples(tuples, names=["band", "index"])
        return index

    def stetson_J(self, band=None):
        """Compute the stetsonJ statistic on data from one or several bands

        Parameters
        ----------
        band : `str` or `list` of `str`
            Single band descriptor, or list of such descriptors.

        Returns
        -------
        stetsonJ : `dict`
            StetsonJ statistic for each of input bands.

        Note
        ----------
        In case that no value for band is passed, the function is executed
        on all available bands.
        """
        return calc_stetson_J(self.flux, self.flux_err, self.band, band_to_calc=band)

    def sf2(self, sf_method="basic", argument_container=None):
        """Compute the structure function squared statistic on data

        Parameters
        ----------
        bins : `numpy.array` or `list`
            Manually provided bins, if not provided then bins are computed using
            the `method` kwarg
        band_to_calc : `str` or `list` of `str`
            Single band descriptor, or list of such descriptors.
        method : 'str'
            The binning method to apply, choices of 'size'; which seeks an even
            distribution of samples per bin using quantiles, 'length'; which
            creates bins of equal length in time and 'loglength'; which creates
            bins of equal length in log time.
        sthresh : 'int'
            Target number of samples per bin.

        Returns
        -------
        stetsonJ : `dict`
            Structure function squared statistic for each of input bands.

        Note
        ----------
        In case that no value for band_to_calc is passed, the function is executed
        on all available bands.
        """
        if self.meta["id"]:
            lc_id = [self.meta["id"]] * len(self.time)
        else:
            lc_id = [0] * len(self.time)
        return calc_sf2(
            time=self.time,
            flux=self.flux,
            err=self.flux_err,
            band=self.band,
            lc_id=lc_id,
            sf_method=sf_method,
            argument_container=argument_container,
        )
