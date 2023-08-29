import dask.dataframe as dd

import dask
from dask.dataframe.dispatch import make_meta_dispatch
from dask.dataframe.backends import _nonempty_index, meta_nonempty, meta_nonempty_dataframe

from dask.dataframe.core import get_parallel_type
from dask.dataframe.extensions import make_array_nonempty

import numpy as np
import pandas as pd

class _Frame(dd.core._Frame):
    """Base class for extensions of Dask Dataframes that track additional Ensemble-related metadata."""

    def __init__(self, dsk, name, meta, divisions, label=None, ensemble=None):
        super().__init__(dsk, name, meta, divisions)
        self.label = label # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble # The Ensemble object containing this frame.

    @property
    def _args(self):
        # Ensure our Dask extension can correctly be used by pickle.
        # See https://github.com/geopandas/dask-geopandas/issues/237
        return super()._args + (self.label, self.ensemble)

    def _propagate_metadata(self, new_frame):
        """Propagatees any relevant metadata to a new frame.

        Parameters
        ----------
        new_frame: `_Frame`
        |   A frame to propage metadata to

        Returns
        ----------
        new_frame: `_Frame`
            The modifed frame
        """
        new_frame.label = self.label
        new_frame.ensemble = self.ensemble
        return new_frame

    def copy(self):
        self_copy = super().copy()
        return self._propagate_metadata(self_copy)

class TapeSeries(pd.Series):
    """A barebones extension of a Pandas series to be used for underlying Ensmeble data.
    
    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """
    @property
    def _constructor(self):
        return TapeSeries
    
    @property
    def _constructor_sliced(self):
        return TapeSeries
    
class TapeFrame(pd.DataFrame):
    """A barebones extension of a Pandas frame to be used for underlying Ensmeble data.
    
    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """
    @property
    def _constructor(self):
        return TapeFrame
    
    @property
    def _constructor_expanddim(self):
        return TapeFrame
    
    
class EnsembleSeries(_Frame, dd.core.Series):
    """A barebones extension of a Dask Series for Ensemble data.
    """
    _partition_type = TapeSeries # Tracks the underlying data type

class EnsembleFrame(_Frame, dd.core.DataFrame):
    """An extension for a Dask Dataframe for data used by a lightcurve Ensemble.

    The underlying non-parallel dataframes are TapeFrames and TapeSeries which extend Pandas frames.

    Example
    ----------
    import tape
    ens = tape.Ensemble()
    data = {...} # Some data you want tracked by the Ensemble
    ensemble_frame = tape.EnsembleFrame.from_dict(data, label="my_frame", ensemble=ens)
    """
    _partition_type = TapeFrame # Tracks the underlying data type

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, _Frame):
            # Ensures that we have any 
            result = self._propagate_metadata(result)
        return result

    @classmethod
    def from_tapeframe(
        cls, data, npartitions=None, chunksize=None, sort=True, label=None, ensemble=None
    ):
        """ Returns an EnsembleFrame constructed from a TapeFrame.
        Parameters
        ----------
        data: `TapeFrame`
            Frame containing the underlying data fro the EnsembleFram
        npartitions: `int`, optional
            The number of partitions of the index to create. Note that depending on
            the size and index of the dataframe, the output may have fewer
            partitions than requested.
        chunksize: `int`, optional
            Size of the individual chunks of data in non-parallel objects that make up Dask frames.
        sort: `bool`, optional
            Whether to sort the frame by a default index.
        label: `str`, optional
        |   The label used to by the Ensemble to identify the frame.
        ensemble: `tape.Ensemble`, optional
        |   A linnk to the Ensmeble object that owns this frame.
        Returns
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        result = dd.from_pandas(data, npartitions=npartitions, chunksize=chunksize, sort=sort)
        result.label = label
        result.ensemble = ensemble
        return result
    
    def convert_flux_to_mag(self, 
                            flux_col, 
                            zero_point, 
                            err_col=None, 
                            zp_form="mag", 
                            out_col_name=None, 
                            ):
        """Converts this EnsembleFrame's flux column into a magnitude column, returning a new
        EnsembleFrame.

        Parameters
        ----------
        flux_col: 'str'
            The name of the EnsembleFrame flux column to convert into magnitudes.
        zero_point: 'str'
            The name of the EnsembleFrame column containing the zero point
            information for column transformation.
        err_col: 'str', optional
            The name of the EnsembleFrame column containing the errors to propagate.
            Errors are propagated using the following approximation:
            Err= (2.5/log(10))*(flux_error/flux), which holds mainly when the
            error in flux is much smaller than the flux.
        zp_form: `str`, optional
            The form of the zero point column, either "flux" or
            "magnitude"/"mag". Determines how the zero point (zp) is applied in
            the conversion. If "flux", then the function is applied as
            mag=-2.5*log10(flux/zp), or if "magnitude", then
            mag=-2.5*log10(flux)+zp.
        out_col_name: 'str', optional
            The name of the output magnitude column, if None then the output
            is just the flux column name + "_mag". The error column is also
            generated as the out_col_name + "_err".
        Returns
        ----------
        result: `tape.EnsembleFrame`
            A new EnsembleFrame object with a new magnitude (and error) column.
        """
        if out_col_name is None:
            out_col_name = flux_col + "_mag"

        result = None
        if zp_form == "flux":  # mag = -2.5*np.log10(flux/zp)
            result = self.assign(
                **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col] / x[zero_point])}
            )

        elif zp_form == "magnitude" or zp_form == "mag":  # mag = -2.5*np.log10(flux) + zp
            result = self.assign(
                **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col]) + x[zero_point]}
            )
        else:
            raise ValueError(f"{zp_form} is not a valid zero_point format.")

        # Calculate Errors
        if err_col is not None:
            result = result.assign(
                **{out_col_name + "_err": lambda x: (2.5 / np.log(10)) * (x[err_col] / x[flux_col])}
            )

        return result

"""
Dask Dataframes are constructed indirectly using method dispatching and inference on the
underlying data. So to ensure our subclasses behave correctly, we register the methods
below.

For more information, see https://docs.dask.org/en/latest/dataframe-extend.html

The following should ensure that any Dask Dataframes which use TapeSeries or TapeFrames as their
underlying data will be resolved as EnsembleFrames or EnsembleSeries as their parrallel
counterparts. The underlying Dask Dataframe _meta will be a TapeSeries or TapeFrame.
"""
get_parallel_type.register(TapeSeries, lambda _: EnsembleSeries)
get_parallel_type.register(TapeFrame, lambda _: EnsembleFrame)

@make_meta_dispatch.register(TapeSeries)
def make_meta_series(x, index=None):
    # Create an empty TapeSeries to use as Dask's underlying object meta.
    result = x.head(0)
    # Re-index if requested
    if index is not None:
        result = result.reindex(index[:0])
    return result

@make_meta_dispatch.register(TapeFrame)
def make_meta_frame(x, index=None):
    # Create an empty TapeFrame to use as Dask's underlying object meta.
    result = x.head(0)
    # Re-index if requested
    if index is not None:
        result = result.reindex(index[:0])
    return result

@meta_nonempty.register(TapeSeries)
def _nonempty_tapeseries(x, index=None):
    # Construct a new TapeSeries with the same underlying data.
    if index is None:
        index = _nonempty_index(x.index)
    data = make_array_nonempty(x.dtype)
    return TapeSeries(data, name=x.name, crs=x.crs)

@meta_nonempty.register(TapeFrame)
def _nonempty_tapeseries(x, index=None):
    # Construct a new TapeFrame with the same underlying data.
    df = meta_nonempty_dataframe(x)
    return TapeFrame(df)