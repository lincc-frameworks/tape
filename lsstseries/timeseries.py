import pandas as pd


class timeseries():
    """represent and analyze Rubin timeseries data"""
    def __init__(self, data=None):
        self.data = data
        self.meta = {'id': None} # metadata dict
        self.colmap = {'time': None, 'flux': None, 'flux_err': None} # column mapping

    # I/O
    def from_dict(self, data_dict, time_label='time', flux_label='flux', err_label='flux_err', 
                  band_label='band'):
        """Build dataframe from a python dictionary

        Parameters
        ----------
        data_dict : `dict`
            Dictionary contaning the data
        time_label: `str`
            Name for column containing time information
        flux_label: `str`
            Name for column containing signal (flux, magnitude, etc) information
        err_label: `str`
            Name for column containing error information
        band_label: `str`
            Name for column containing filter information
        """

        try:
            data_dict[band_label]
        except KeyError:
            raise KeyError(f"The indicated label '{band_label}' was not found.")
        index = self._build_index(data_dict[band_label])
        data_dict = {key: data_dict[key] for key in data_dict if key != band_label}
        self.data = pd.DataFrame(data=data_dict, index=index).sort_index()

        labels = [time_label, flux_label, err_label]
        
        for label, quantity in zip(labels, list(self.colmap.keys())):

            if (quantity == 'flux_err') and (label is None): # flux_err is optional
                continue

            try:
                self.data[label]
                self.colmap[quantity] = label
            except KeyError:
                raise KeyError(f"The indicated label '{label}' was not found.")
            
        return self      

    def _from_ensemble(self, data, object_id, time_label='time', flux_label='flux', err_label='flux_err'):
        """Loader function for inputing data from an ensemble"""
        self.cols = list(data.columns)
        self.data = data
        self.meta['id'] = object_id

        labels = [time_label, flux_label, err_label]
        
        for label, quantity in zip(labels, list(self.colmap.keys())):

            if (quantity == 'flux_err') and (label is None): # flux_err is optional
                continue

            try:
                self.data[label]
                self.colmap[quantity] = label
            except KeyError:
                raise KeyError(f"The indicated label '{label}' was not found.")

        return self

    @property
    def time(self):
        """Time values stored as a Pandas Series"""
        return self.data[self.colmap['time']]

    @property
    def flux(self):
        """Flux values stored as a Pandas Series"""
        return self.data[self.colmap['flux']]

    @property
    def flux_err(self):
        """Flux error values stored as a Pandas Series"""
        if self.colmap['flux_err'] is not None: # Errors are not mandatory
            return self.data[self.colmap['flux_err']]
        else:
            return None

    @property
    def band(self):
        """Band labels stored as a Pandas Index"""
        return self.data.index.get_level_values('band')

    def _build_index(self, band):
        """Build pandas multiindex from band array"""
        # Create a multiindex
        tuples = zip(band, range(len(band)))
        index = pd.MultiIndex.from_tuples(tuples, names=["band", "index"])
        return index
