from collections.abc import Sequence

import dask.dataframe as dd

import dask
from dask.dataframe.dispatch import make_meta_dispatch
from dask.dataframe.backends import _nonempty_index, meta_nonempty, meta_nonempty_dataframe

from dask.dataframe.core import get_parallel_type
from dask.dataframe.extensions import make_array_nonempty

import numpy as np
import pandas as pd

from typing import Literal


from functools import partial
from dask.dataframe.io.parquet.arrow import (
        ArrowDatasetEngine as DaskArrowDatasetEngine,
    )

SOURCE_FRAME_LABEL = "source" # Reserved label for source table
OBJECT_FRAME_LABEL = "object" # Reserved label for object table.

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
    
    def assign(self, **kwargs):
        """Assign new columns to a DataFrame.

        This docstring was copied from dask.dataframe.DataFrame.assign.

        Some inconsistencies with the Dask version may exist.

        Returns a new object with all original columns in addition to new ones. Existing columns
        that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs: `dict`
            The column names are keywords. If the values are callable, they are computed on the
            DataFrame and assigned to the new columns. The callable must not change input DataFrame
            (though pandas doesn’t check it). If the values are not callable, (e.g. a Series,
            scalar, or array), they are simply assigned.

        Returns
        ----------
        result: `tape._Frame`
            The modifed frame
        """
        result = super().assign(**kwargs)
        return self._propagate_metadata(result)
    
    def query(self, expr, **kwargs):
        """Filter dataframe with complex expression

        Doc string below derived from dask.dataframe.core

        Blocked version of pd.DataFrame.query

        Parameters
        ----------
        expr: str
            The query string to evaluate.
            You can refer to column names that are not valid Python variable names
            by surrounding them in backticks.
            Dask does not fully support referring to variables using the '@' character,
            use f-strings or the ``local_dict`` keyword argument instead.
        **kwargs: `dict`
            See the documentation for eval() for complete details on the keyword arguments accepted
            by pandas.DataFrame.query().

        Returns
        ----------
        result: `tape._Frame`
            The modifed frame
            
        Notes
        -----
        This is like the sequential version except that this will also happen
        in many threads.  This may conflict with ``numexpr`` which will use
        multiple threads itself.  We recommend that you set ``numexpr`` to use a
        single thread:

        .. code-block:: python

            import numexpr
            numexpr.set_num_threads(1)
        """
        result = super().query(expr, **kwargs)
        return self._propagate_metadata(result)
    
    def set_index(
        self,
        other: str | pd.Series,
        drop: bool = True,
        sorted: bool = False,
        npartitions: int | Literal["auto"] | None = None,
        divisions: Sequence | None = None,
        inplace: bool = False,
        sort: bool = True,
        **kwargs,
    ):

        """Set the DataFrame index (row labels) using an existing column.

        Doc string below derived from dask.dataframe.core

        If ``sort=False``, this function operates exactly like ``pandas.set_index``
        and sets the index on the DataFrame. If ``sort=True`` (default),
        this function also sorts the DataFrame by the new index. This can have a
        significant impact on performance, because joins, groupbys, lookups, etc.
        are all much faster on that column. However, this performance increase
        comes with a cost, sorting a parallel dataset requires expensive shuffles.
        Often we ``set_index`` once directly after data ingest and filtering and
        then perform many cheap computations off of the sorted dataset.

        With ``sort=True``, this function is much more expensive. Under normal
        operation this function does an initial pass over the index column to
        compute approximate quantiles to serve as future divisions. It then passes
        over the data a second time, splitting up each input partition into several
        pieces and sharing those pieces to all of the output partitions now in
        sorted order.

        In some cases we can alleviate those costs, for example if your dataset is
        sorted already then we can avoid making many small pieces or if you know
        good values to split the new index column then we can avoid the initial
        pass over the data. For example if your new index is a datetime index and
        your data is already sorted by day then this entire operation can be done
        for free. You can control these options with the following parameters.

        Parameters
        ----------
        other: string or Dask Series
            Column to use as index.
        drop: boolean, default True
            Delete column to be used as the new index.
        sorted: bool, optional
            If the index column is already sorted in increasing order.
            Defaults to False
        npartitions: int, None, or 'auto'
            The ideal number of output partitions. If None, use the same as
            the input. If 'auto' then decide by memory use.
            Only used when ``divisions`` is not given. If ``divisions`` is given,
            the number of output partitions will be ``len(divisions) - 1``.
        divisions: list, optional
            The "dividing lines" used to split the new index into partitions.
            For ``divisions=[0, 10, 50, 100]``, there would be three output partitions,
            where the new index contained [0, 10), [10, 50), and [50, 100), respectively.
            See https://docs.dask.org/en/latest/dataframe-design.html#partitions.
            If not given (default), good divisions are calculated by immediately computing
            the data and looking at the distribution of its values. For large datasets,
            this can be expensive.
            Note that if ``sorted=True``, specified divisions are assumed to match
            the existing partitions in the data; if this is untrue you should
            leave divisions empty and call ``repartition`` after ``set_index``.
        inplace: bool, optional
            Modifying the DataFrame in place is not supported by Dask.
            Defaults to False.
        sort: bool, optional
            If ``True``, sort the DataFrame by the new index. Otherwise
            set the index on the individual existing partitions.
            Defaults to ``True``.
        shuffle: {'disk', 'tasks', 'p2p'}, optional
            Either ``'disk'`` for single-node operation or ``'tasks'`` and
            ``'p2p'`` for distributed operation.  Will be inferred by your
            current scheduler.
        compute: bool, default False
            Whether or not to trigger an immediate computation. Defaults to False.
            Note, that even if you set ``compute=False``, an immediate computation
            will still be triggered if ``divisions`` is ``None``.
        partition_size: int, optional
            Desired size of each partitions in bytes.
            Only used when ``npartitions='auto'``
  
        Returns
        ----------
        result: `tape._Frame`
            The indexed frame
        """
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

    _is_dirty = False # True if the underlying data is out of sync with the Ensemble

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, _Frame):
            # Ensures that any _Frame metadata is propagated.
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
        |   A link to the Ensemble object that owns this frame.
        Returns
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        result = dd.from_pandas(data, npartitions=npartitions, chunksize=chunksize, sort=sort)
        result.label = label
        result.ensemble = ensemble
        return result

    def update_ensemble(self):
        """ Updates the Ensemble linked by the `EnsembelFrame.ensemble` property to track this frame.

        Returns
        result: `tape.Ensemble`
            The Ensemble object which tracks this frame, `None` if no such Ensemble.
        """
        if self.ensemble is None:
            return None
        # Update the Ensemble to track this frame and return the ensemble.
        return self.ensemble.update_frame(self)
    
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

    @classmethod
    def from_parquet(
        cl,
        path,
        index=None,
        columns=None,
        ensemble=None,
    ):
        """ Returns an EnsembleFrame constructed from loading a parquet file.
        Parameters
        ----------
        path: `str` or `list`
            Source directory for data, or path(s) to individual parquet files. Prefix with a
            protocol like s3:// to read from alternative filesystems. To read from multiple
            files you can pass a globstring or a list of paths, with the caveat that they must all
            have the same protocol.
        columns: `str` or `list`, optional
            Field name(s) to read in as columns in the output. By default all non-index fields will
            be read (as determined by the pandas parquet metadata, if present). Provide a single
            field name instead of a list to read in the data as a Series.
        index: `str`, `list`, `False`, optional
            Field name(s) to use as the output frame index. Default is None and index will be
            inferred from the pandas parquet file metadata, if present. Use False to read all
            fields as columns.
        ensemble: `tape.ensemble.Ensemble`, optional
        |   A link to the Ensemble object that owns this frame.
        Returns
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        # Read the parquet file with an engine that will assume the meta is a TapeFrame which Dask will
        # instantiate as EnsembleFrame via its dispatcher.
        result = dd.read_parquet(
            path, index=index, columns=columns, split_row_groups=True, engine=TapeArrowEngine,
        )
        result.ensemble=ensemble

        return result
    
    def is_dirty(self):
        return self._is_dirty
    
    def set_dirty(self, is_dirty):
        self._is_dirty = is_dirty

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
    """ A subclass of EnsembleFrame for Source data. """

    _partition_type = TapeSourceFrame # Tracks the underlying data type

    def __init__(self, dsk, name, meta, divisions, ensemble=None):
        super().__init__(dsk, name, meta, divisions)
        self.label = SOURCE_FRAME_LABEL # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble # The Ensemble object containing this frame.

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, _Frame):
            # Ensures that we have any metadata
            result = self._propagate_metadata(result)
        return result

    @classmethod
    def from_parquet(
        cl,
        path,
        index=None,
        columns=None,
        ensemble=None,
    ):
        """ Returns a SourceFrame constructed from loading a parquet file.
        Parameters
        ----------
        path: `str` or `list`
            Source directory for data, or path(s) to individual parquet files. Prefix with a
            protocol like s3:// to read from alternative filesystems. To read from multiple
            files you can pass a globstring or a list of paths, with the caveat that they must all
            have the same protocol.
        columns: `str` or `list`, optional
            Field name(s) to read in as columns in the output. By default all non-index fields will
            be read (as determined by the pandas parquet metadata, if present). Provide a single
            field name instead of a list to read in the data as a Series.
        index: `str`, `list`, `False`, optional
            Field name(s) to use as the output frame index. Default is None and index will be
            inferred from the pandas parquet file metadata, if present. Use False to read all
            fields as columns.
        ensemble: `tape.ensemble.Ensemble`, optional
        |   A link to the Ensemble object that owns this frame.
        Returns
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        # Read the source parquet file with an engine that will assume the meta is a 
        # TapeSourceFrame which tells Dask to instantiate a SourceFrame via its
        # dispatcher.
        result = dd.read_parquet(
            path, index=index, columns=columns, split_row_groups=True, engine=TapeSourceArrowEngine,
        )
        result.ensemble=ensemble
        result.label = SOURCE_FRAME_LABEL

        return result
    
class ObjectFrame(EnsembleFrame):
    """ A subclass of EnsembleFrame for Object data. """

    _partition_type = TapeObjectFrame # Tracks the underlying data type

    def __init__(self, dsk, name, meta, divisions, ensemble=None):
        super().__init__(dsk, name, meta, divisions)
        self.label = OBJECT_FRAME_LABEL # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble # The Ensemble object containing this frame.

    @classmethod
    def from_parquet(
        cl,
        path,
        index=None,
        columns=None,
        ensemble=None,
    ):
        """ Returns an ObjectFrame constructed from loading a parquet file.
        Parameters
        ----------
        path: `str` or `list`
            Source directory for data, or path(s) to individual parquet files. Prefix with a
            protocol like s3:// to read from alternative filesystems. To read from multiple
            files you can pass a globstring or a list of paths, with the caveat that they must all
            have the same protocol.
        columns: `str` or `list`, optional
            Field name(s) to read in as columns in the output. By default all non-index fields will
            be read (as determined by the pandas parquet metadata, if present). Provide a single
            field name instead of a list to read in the data as a Series.
        index: `str`, `list`, `False`, optional
            Field name(s) to use as the output frame index. Default is None and index will be
            inferred from the pandas parquet file metadata, if present. Use False to read all
            fields as columns.
        ensemble: `tape.ensemble.Ensemble`, optional
        |   A link to the Ensemble object that owns this frame.
        Returns
        result: `tape.ObjectFrame`
            The constructed ObjectFrame object.
        """
        # Read in the object Parquet file
        result = dd.read_parquet(
            path, index=index, columns=columns, split_row_groups=True, engine=TapeObjectArrowEngine,
        )
        result.ensemble = ensemble
        result.label= OBJECT_FRAME_LABEL

        return result

    @classmethod
    def from_dask_dataframe(cl, df, ensemble=None):
        """ Returns an ObjectFrame constructed from a Dask dataframe..
        Parameters
        ----------
        df: `dask.dataframe.DataFrame` or `list`
            a Dask dataframe to convert to an ObjectFrame
        ensemble: `tape.ensemble.Ensemble`, optional
        |   A link to the Ensemble object that owns this frame.
        Returns
        result: `tape.ObjectFrame`
            The constructed ObjectFrame object.
        """
        # Create an ObjectFrame by mapping the partitions to the appropriate meta, TapeObjectFrame
        # TODO(wbeebe@uw.edu): Determine if there is a better method
        result = df.map_partitions(TapeObjectFrame) 
        result.set_ensemble = ensemble
        result.set_label = OBJECT_FRAME_LABEL
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
get_parallel_type.register(TapeObjectFrame, lambda _: ObjectFrame)
get_parallel_type.register(TapeSourceFrame, lambda _: SourceFrame)

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
    return TapeSeries(data, name=x.name, crs=x.crs)

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