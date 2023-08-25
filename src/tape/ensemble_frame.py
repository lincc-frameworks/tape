import dask.dataframe as dd

from packaging.version import Version
import dask
DASK_2021_06_0 = Version(dask.__version__) >= Version("2021.06.0")
DASK_2022_06_0 = Version(dask.__version__) >= Version("2022.06.0")
if DASK_2021_06_0:
    from dask.dataframe.dispatch import make_meta_dispatch
    from dask.dataframe.backends import _nonempty_index, meta_nonempty, meta_nonempty_dataframe
else:
    from dask.dataframe.core import make_meta as make_meta_dispatch
    from dask.dataframe.utils import _nonempty_index, meta_nonempty, meta_nonempty_dataframe

from dask.dataframe.core import get_parallel_type
from dask.dataframe.extensions import make_array_nonempty

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
        result = dd.from_pandas(data, npartitions=npartitions, chunksize=chunksize, sort=sort, name="fdsafdfasd")
        result.label = label
        result.ensemble = ensemble
        return result

    @classmethod
    def from_dict(
        cls, data, npartitions=None, orient="columns", dtype=None, columns=None, label=None, 
        ensemble=None,
    ):
        """Returns an EnsembleFrame constructed from a Python Dictionary.
        Parameters
        ----------
        data: `TapeFrame`
            Frame containing the underlying data fro the EnsembleFram
        npartitions: `int`, optional
            The number of partitions of the index to create. Note that depending on
            the size and index of the dataframe, the output may have fewer
            partitions than requested.
        orient: `str`, optional
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass 'columns'
            (default). Otherwise if the keys should be rows, pass 'index'.
            If 'tight', assume a dict with keys
            ['index', 'columns', 'data', 'index_names', 'column_names'].
        dtype: `bool`, optional
            Data type to force, otherwise infer.
        columns: `str`, optional
            Column labels to use when ``orient='index'``. Raises a ValueError
            if used with ``orient='columns'`` or ``orient='tight'``.
        label: `str`, optional
        |   The label used to by the Ensemble to identify the frame.
        ensemble: `tape.Ensemble`, optional
        |   A linnk to the Ensmeble object that owns this frame.
        Returns
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        frame = TapeFrame.from_dict(data, orient, dtype, columns)
        return EnsembleFrame.from_tapeframe(frame,
            label=label, ensemble=ensemble, npartitions=npartitions
        )

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