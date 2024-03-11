from packaging.version import Version
import warnings

import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
import dask.array as da

# from dask.dataframe import map_partitions
from dask.highlevelgraph import HighLevelGraph
from dask.utils import M, OperatorMethodMixin, derived_from, ignore_warning
from dask.base import tokenize

from dask.dataframe.core import get_parallel_type
from dask.dataframe.utils import meta_nonempty
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.base import normalize_token
from dask.dataframe.dispatch import make_meta_dispatch, pyarrow_schema_dispatch
from dask.dataframe.backends import _nonempty_index, meta_nonempty_dataframe, _nonempty_series

import dask_expr as dx
from dask_expr import (
    elemwise,
    from_graph,
    get_collection_type,
)
from dask_expr._collection import new_collection
from dask_expr._expr import _emulate, ApplyConcatApply

from .ensemble_frame import TapeFrame, TapeSeries

SOURCE_FRAME_LABEL = "source"  # Reserved label for source table
OBJECT_FRAME_LABEL = "object"  # Reserved label for object table.

__all__ = [
    "EnsembleFrame",
    "EnsembleSeries",
    "ObjectFrame",
    "SourceFrame",
    "TapeFrame",
    "TapeObjectFrame",
    "TapeSourceFrame",
    "TapeSeries",
]

from functools import partial
from dask.dataframe.io.parquet.arrow import (
    ArrowDatasetEngine as DaskArrowDatasetEngine,
)


class TapeArrowEngine(DaskArrowDatasetEngine):
    """
    Engine for reading parquet files into Tape and assigning the appropriate Dask meta.

    Based off of the approach used in dask_geopandas.io
    """

    @classmethod
    def _creates_meta(cls, meta, schema):
        """
        Converts the meta to a TapeFrame.
        """
        return TapeFrame(meta)

    @classmethod
    def _create_dd_meta(cls, dataset_info, use_nullable_dtypes=False):
        """Overriding private method for dask >= 2021.10.0"""
        meta = super()._create_dd_meta(dataset_info)

        schema = dataset_info["schema"]
        if not schema.names and not schema.metadata:
            if len(list(dataset_info["ds"].get_fragments())) == 0:
                raise ValueError(
                    "No dataset parts discovered. Use dask.dataframe.read_parquet "
                    "to read it as an empty DataFrame"
                )
        meta = cls._creates_meta(meta, schema)
        return meta


class TapeSourceArrowEngine(TapeArrowEngine):
    """
    Barebones subclass of TapeArrowEngine for assigning the meta when loading from a parquet file
    of source data.
    """

    @classmethod
    def _creates_meta(cls, meta, schema):
        """
        Convert meta to a TapeSourceFrame
        """
        return TapeSourceFrame(meta)


class TapeObjectArrowEngine(TapeArrowEngine):
    """
    Barebones subclass of TapeArrowEngine for assigning the meta when loading from a parquet file
    of object data.
    """

    @classmethod
    def _creates_meta(cls, meta, schema):
        """
        Convert meta to a TapeObjectFrame
        """
        return TapeObjectFrame(meta)


class _Frame(dx.FrameBase, OperatorMethodMixin):
    """Superclass for DataFrame and Series
    Parameters
    ----------
    dsk : dict
        The dask graph to compute this DataFrame
    name : str
        The key prefix that specifies which keys in the dask comprise this
        particular DataFrame / Series
    meta : geopandas.GeoDataFrame, geopandas.GeoSeries
        An empty geopandas object with names, dtypes, and indices matching the
        expected output.
    divisions : tuple of index values
        Values along which we partition our blocks on the index
    """

    _partition_type = TapeFrame

    def __init__(self, expr, label=None, ensemble=None):

        # We define relevant object fields before super().__init__ since that call may lead to a
        # map_partitions call which will assume these fields exist.
        self.label = label  # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble  # The Ensemble object containing this frame.
        self.dirty = False  # True if the underlying data is out of sync with the Ensemble

        super().__init__(expr)

    def is_dirty(self):
        return self.dirty

    def set_dirty(self, dirty):
        self.dirty = dirty

    @property
    def _args(self):
        # Ensure our Dask extension can correctly be used by pickle.
        # See https://github.com/geopandas/dask-geopandas/issues/237
        return super()._args + (self.label, self.ensemble)

    def optimize(self, fuse: bool = True):
        result = new_collection(self.expr.optimize(fuse=fuse))
        return result

    def __dask_postpersist__(self):
        func, args = super().__dask_postpersist__()

        return self._rebuild, (func, args)

    def _rebuild(self, graph, func, args):
        collection = func(graph, *args)
        return collection

    def _propagate_metadata(self, new_frame):
        """Propagates any relevant metadata to a new frame.

        Parameters
        ----------
        new_frame: `_Frame`
            A frame to propage metadata to

        Returns
        ----------
        new_frame: `_Frame`
            The modifed frame
        """
        new_frame.label = self.label
        new_frame.ensemble = self.ensemble
        new_frame.set_dirty(self.is_dirty())
        return new_frame

    def map_partitions(self, func, *args, **kwargs):
        """Apply Python function on each DataFrame partition."""

        result = super().map_partitions(func, *args, **kwargs)
        # if isinstance(result, self.__class__):
        # If the output of func is another _Frame, let's propagate any metadata.
        #    return self._propagate_metadata(result)
        return result

    def set_index(
        self,
        other,
        drop=True,
        sorted=False,
        npartitions=None,
        divisions=None,
        inplace=False,
        sort=True,
        **kwargs,
    ):
        result = super().set_index(other, drop, sorted, npartitions, divisions, inplace, sort, **kwargs)
        return self._propagate_metadata(result)


class TapeSeries(pd.Series):
    """A barebones extension of a Pandas series to be used for underlying Ensemble data.

    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """

    @property
    def _constructor(self):
        return TapeSeries

    @property
    def _constructor_sliced(self):
        return TapeSeries


class TapeFrame(pd.DataFrame):
    """A barebones extension of a Pandas frame to be used for underlying Ensemble data.

    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """

    @property
    def _constructor(self):
        return TapeFrame

    @property
    def _constructor_expanddim(self):
        return TapeFrame


class EnsembleSeries(_Frame, dx.Series):
    """A barebones extension of a Dask Series for Ensemble data."""

    _partition_type = TapeSeries  # Tracks the underlying data type


class EnsembleFrame(
    _Frame, dx.DataFrame
):  # should be able to use dd.DataFrame, but that's causing recursion issues
    """An extension for a Dask Dataframe for data used by a lightcurve Ensemble.

    The underlying non-parallel dataframes are TapeFrames and TapeSeries which extend Pandas frames.

    Examples
    ----------
    Instatiation::

        import tape
        ens = tape.Ensemble()
        data = {...} # Some data you want tracked by the Ensemble
        ensemble_frame = tape.EnsembleFrame.from_dict(data, label="my_frame", ensemble=ens)
    """

    _partition_type = TapeFrame  # Tracks the underlying data type

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, _Frame):
            # Ensures that any _Frame metadata is propagated.
            result = self._propagate_metadata(result)
        return result

    @classmethod
    def from_dask_dataframe(cl, df, ensemble=None, label=None):
        """Returns an EnsembleFrame constructed from a Dask dataframe.

        Parameters
        ----------
        df: `dask.dataframe.DataFrame` or `list`
            a Dask dataframe to convert to an EnsembleFrame
        ensemble: `tape.ensemble.Ensemble`, optional
            A link to the Ensemble object that owns this frame.
        label: `str`, optional
            The label used to by the Ensemble to identify the frame.

        Returns
        ----------
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        # Create a EnsembleFrame by mapping the partitions to the appropriate meta, TapeFrame
        # TODO(wbeebe@uw.edu): Determine if there is a better method
        result = df.map_partitions(TapeFrame)
        result.ensemble = ensemble
        result.label = label
        return result


class TapeSourceFrame(TapeFrame):
    """A barebones extension of a Pandas frame to be used for underlying Ensemble source data

    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """

    @property
    def _constructor(self):
        return TapeSourceFrame

    @property
    def _constructor_expanddim(self):
        return TapeSourceFrame


class TapeObjectFrame(TapeFrame):
    """A barebones extension of a Pandas frame to be used for underlying Ensemble object data.

    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """

    @property
    def _constructor(self):
        return TapeObjectFrame

    @property
    def _constructor_expanddim(self):
        return TapeObjectFrame


class SourceFrame(EnsembleFrame):
    """A subclass of EnsembleFrame for Source data."""

    _partition_type = TapeSourceFrame  # Tracks the underlying data type

    def __init__(self, expr, ensemble=None):

        # We define relevant object fields before super().__init__ since that call may lead to a
        # map_partitions call which will assume these fields exist.
        self.label = SOURCE_FRAME_LABEL  # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble  # The Ensemble object containing this frame.
        self.dirty = False  # True if the underlying data is out of sync with the Ensemble

        super().__init__(expr)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, _Frame):
            # Ensures that we have any metadata
            result = self._propagate_metadata(result)
        return result

    @classmethod
    def from_dask_dataframe(cl, df, ensemble=None):
        """Returns a SourceFrame constructed from a Dask dataframe.

        Parameters
        ----------
        df: `dask.dataframe.DataFrame` or `list`
            a Dask dataframe to convert to a SourceFrame
        ensemble: `tape.ensemble.Ensemble`, optional
            A link to the Ensemble object that owns this frame.

        Returns
        ----------
        result: `tape.SourceFrame`
            The constructed SourceFrame object.
        """
        # Create a SourceFrame by mapping the partitions to the appropriate meta, TapeSourceFrame
        # TODO(wbeebe@uw.edu): Determine if there is a better method
        result = df.map_partitions(TapeSourceFrame)
        result.ensemble = ensemble
        result.label = SOURCE_FRAME_LABEL
        return result


class ObjectFrame(EnsembleFrame):
    """A subclass of EnsembleFrame for Object data."""

    _partition_type = TapeObjectFrame  # Tracks the underlying data type

    def __init__(self, expr, ensemble=None):

        # We define relevant object fields before super().__init__ since that call may lead to a
        # map_partitions call which will assume these fields exist.
        self.label = OBJECT_FRAME_LABEL  # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble  # The Ensemble object containing this frame.
        self.dirty = False  # True if the underlying data is out of sync with the Ensemble

        super().__init__(expr)

    @classmethod
    def from_dask_dataframe(cl, df, ensemble=None):
        """Returns an ObjectFrame constructed from a Dask dataframe.

        Parameters
        ----------
        df: `dask.dataframe.DataFrame` or `list`
            a Dask dataframe to convert to an ObjectFrame
        ensemble: `tape.ensemble.Ensemble`, optional
            A link to the Ensemble object that owns this frame.

        Returns
        ----------
        result: `tape.ObjectFrame`
            The constructed ObjectFrame object.
        """
        # Create an ObjectFrame by mapping the partitions to the appropriate meta, TapeObjectFrame
        # TODO(wbeebe@uw.edu): Determine if there is a better method
        result = df.map_partitions(TapeObjectFrame)
        result.ensemble = ensemble
        result.label = OBJECT_FRAME_LABEL
        return result


# Dask Dataframes are constructed indirectly using method dispatching and inference on the
# underlying data. So to ensure our subclasses behave correctly, we register the methods
# below.
#
# For more information, see https://docs.dask.org/en/latest/dataframe-extend.html
#
# The following should ensure that any Dask Dataframes which use TapeSeries or TapeFrames as their
# underlying data will be resolved as EnsembleFrames or EnsembleSeries as their parrallel
# counterparts. The underlying Dask Dataframe _meta will be a TapeSeries or TapeFrame.


get_collection_type.register(TapeSeries, lambda _: EnsembleSeries)
get_collection_type.register(TapeFrame, lambda _: EnsembleFrame)
get_collection_type.register(TapeObjectFrame, lambda _: ObjectFrame)
get_collection_type.register(TapeSourceFrame, lambda _: SourceFrame)


@make_meta_dispatch.register(TapeSeries)
def make_meta_series(x, index=None):
    # Create an empty TapeSeries to use as Dask's underlying object meta.
    result = x.head(0)
    return result


@make_meta_dispatch.register(TapeFrame)
def make_meta_frame(x, index=None):
    # Create an empty TapeFrame to use as Dask's underlying object meta.
    result = x.head(0)
    return result


@meta_nonempty.register(TapeSeries)
def _nonempty_tapeseries(x, index=None):
    # Construct a new TapeSeries with the same underlying data.
    data = _nonempty_series(x)
    return TapeSeries(data)


@meta_nonempty.register(TapeFrame)
def _nonempty_tapeseries(x, index=None):
    # Construct a new TapeFrame with the same underlying data.
    df = meta_nonempty_dataframe(x)
    return TapeFrame(df)


@make_meta_dispatch.register(TapeObjectFrame)
def make_meta_frame(x, index=None):
    # Create an empty TapeObjectFrame to use as Dask's underlying object meta.
    result = x.head(0)
    return result


@meta_nonempty.register(TapeObjectFrame)
def _nonempty_tapesourceframe(x, index=None):
    # Construct a new TapeObjectFrame with the same underlying data.
    df = meta_nonempty_dataframe(x)
    return TapeObjectFrame(df)


@make_meta_dispatch.register(TapeSourceFrame)
def make_meta_frame(x, index=None):
    # Create an empty TapeSourceFrame to use as Dask's underlying object meta.
    result = x.head(0)
    return result


@meta_nonempty.register(TapeSourceFrame)
def _nonempty_tapesourceframe(x, index=None):
    # Construct a new TapeSourceFrame with the same underlying data.
    df = meta_nonempty_dataframe(x)
    return TapeSourceFrame(df)
