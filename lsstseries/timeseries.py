import pandas as pd
import numpy as np

class timeseries():
    def __init__(self,data=None):
        self.data = data
    
    #I/O
    def from_dict(self,data_dict,band_label='band'):
        """Build dataframe from a python dictionary"""
        index = self._build_index(data_dict[band_label])
        data_dict = {key: data_dict[key] for key in data_dict if key != band_label}
        self.data = pd.DataFrame(data=data_dict,index=index).sort_index()
        return self
    
    @property
    def time(self):
        """Time values stored as a Pandas Series"""
        return self.data["time"]
    
    @property
    def flux(self):
        """Flux values stored as a Pandas Series"""
        return self.data["flux"]
    
    @property
    def flux_err(self):
        """Flux error values stored as a Pandas Series"""
        return self.data["flux_err"]
    
    @property
    def band(self):
        """Band labels stored as a Pandas Series from Index"""
        return self.data.index.get_level_values('band')
    
    def _build_index(self,band):
        """Build pandas multiindex from band array"""
        #Create a multiindex
        tuples = zip(band,range(len(band)))
        index = pd.MultiIndex.from_tuples(tuples, names=["band", "index"])
        return index