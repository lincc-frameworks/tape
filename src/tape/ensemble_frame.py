from collections.abc import Sequence

import warnings

import dask.dataframe as dd

import dask
from dask.dataframe.dispatch import make_meta_dispatch
from dask.dataframe.backends import _nonempty_index, meta_nonempty, meta_nonempty_dataframe, _nonempty_series
from tape.utils import IndexCallable
from dask.dataframe.core import get_parallel_type
from dask.dataframe.extensions import make_array_nonempty

import numpy as np
import pandas as pd

from typing import Literal


from functools import partial
from dask.dataframe.io.parquet.arrow import (
    ArrowDatasetEngine as DaskArrowDatasetEngine,
)

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
        # We define relevant object fields before super().__init__ since that call may lead to a
        # map_partitions call which will assume these fields exist.
        self.label = label  # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble  # The Ensemble object containing this frame.
        self.dirty = False  # True if the underlying data is out of sync with the Ensemble

        super().__init__(dsk, name, meta, divisions)

    def is_dirty(self):
        return self.dirty

    def set_dirty(self, dirty):
        self.dirty = dirty

    @property
    def _args(self):
        # Ensure our Dask extension can correctly be used by pickle.
        # See https://github.com/geopandas/dask-geopandas/issues/237
        return super()._args + (self.label, self.ensemble)

    @property
    def partitions(self):
        """Slice dataframe by partitions

        This allows partitionwise slicing of a TAPE EnsembleFrame.  You can perform normal
        Numpy-style slicing, but now rather than slice elements of the array you
        slice along partitions so, for example, ``df.partitions[:5]`` produces a new
        Dask Dataframe of the first five partitions. Valid indexers are integers, sequences
        of integers, slices, or boolean masks.

        Examples
        --------
        >>> df.partitions[0]  # doctest: +SKIP
        >>> df.partitions[:3]  # doctest: +SKIP
        >>> df.partitions[::10]  # doctest: +SKIP

        Returns
        -------
        A TAPE EnsembleFrame Object
        """
        self.set_dirty(True)
        return IndexCallable(self._partitions, self.is_dirty(), self.ensemble)

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
            (though pandas doesn't check it). If the values are not callable, (e.g. a Series,
            scalar, or array), they are simply assigned.

        Returns
        ----------
        result: `tape._Frame`
            The modifed frame
        """
        result = self._propagate_metadata(super().assign(**kwargs))
        result.set_dirty(True)
        return result

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
        result = self._propagate_metadata(super().query(expr, **kwargs))
        result.set_dirty(True)
        return result

    def sample(self, **kwargs):
        """Random sample of items from a Dataframe.

        Doc string below derived from dask.dataframe.core

        Parameters
        ----------
        frac: float, optional
            Approximate fraction of objects to return. This sampling fraction
            is applied to all partitions equally. Note that this is an
            approximate fraction. You should not expect exactly len(df) * frac
            items to be returned, as the exact number of elements selected will
            depend on how your data is partitioned (but should be pretty close
            in practice).
        replace: boolean, optional
            Sample with or without replacement. Default = False.
        random_state: int or np.random.RandomState
            If an int, we create a new RandomState with this as the seed;
            Otherwise we draw from the passed RandomState.

        Returns
        ----------
        result: `tape._Frame`
            The modifed frame

        """
        result = self._propagate_metadata(super().sample(**kwargs))
        result.set_dirty(True)
        return result

    def merge(self, right, **kwargs):
        """Merge the Dataframe with another DataFrame

        Doc string below derived from dask.dataframe.core

        This will merge the two datasets, either on the indices, a certain column
        in each dataset or the index in one dataset and the column in another.

        Parameters
        ----------
        right: dask.dataframe.DataFrame
        how : {'left', 'right', 'outer', 'inner'}, default: 'inner'
            How to handle the operation of the two objects:

            - left: use calling frame's index (or column if on is specified)
            - right: use other frame's index
            - outer: form union of calling frame's index (or column if on is
              specified) with other frame's index, and sort it
              lexicographically
            - inner: form intersection of calling frame's index (or column if
              on is specified) with other frame's index, preserving the order
              of the calling's one

        on : label or list
            Column or index level names to join on. These must be found in both
            DataFrames. If on is None and not merging on indexes then this
            defaults to the intersection of the columns in both DataFrames.
        left_on : label or list, or array-like
            Column to join on in the left DataFrame. Other than in pandas
            arrays and lists are only support if their length is 1.
        right_on : label or list, or array-like
            Column to join on in the right DataFrame. Other than in pandas
            arrays and lists are only support if their length is 1.
        left_index : boolean, default False
            Use the index from the left DataFrame as the join key.
        right_index : boolean, default False
            Use the index from the right DataFrame as the join key.
        suffixes : 2-length sequence (tuple, list, ...)
            Suffix to apply to overlapping column names in the left and
            right side, respectively
        indicator : boolean or string, default False
            If True, adds a column to output DataFrame called "_merge" with
            information on the source of each row. If string, column with
            information on source of each row will be added to output DataFrame,
            and column will be named value of string. Information column is
            Categorical-type and takes on a value of "left_only" for observations
            whose merge key only appears in `left` DataFrame, "right_only" for
            observations whose merge key only appears in `right` DataFrame,
            and "both" if the observation's merge key is found in both.
        npartitions: int or None, optional
            The ideal number of output partitions. This is only utilised when
            performing a hash_join (merging on columns only). If ``None`` then
            ``npartitions = max(lhs.npartitions, rhs.npartitions)``.
            Default is ``None``.
        shuffle: {'disk', 'tasks', 'p2p'}, optional
            Either ``'disk'`` for single-node operation or ``'tasks'`` and
            ``'p2p'``` for distributed operation.  Will be inferred by your
            current scheduler.
        broadcast: boolean or float, optional
            Whether to use a broadcast-based join in lieu of a shuffle-based
            join for supported cases.  By default, a simple heuristic will be
            used to select the underlying algorithm. If a floating-point value
            is specified, that number will be used as the ``broadcast_bias``
            within the simple heuristic (a large number makes Dask more likely
            to choose the ``broacast_join`` code path). See ``broadcast_join``
            for more information.

        Notes
        -----

        There are three ways to join dataframes:

        1. Joining on indices. In this case the divisions are
           aligned using the function ``dask.dataframe.multi.align_partitions``.
           Afterwards, each partition is merged with the pandas merge function.

        2. Joining one on index and one on column. In this case the divisions of
           dataframe merged by index (:math:`d_i`) are used to divide the column
           merged dataframe (:math:`d_c`) one using
           ``dask.dataframe.multi.rearrange_by_divisions``. In this case the
           merged dataframe (:math:`d_m`) has the exact same divisions
           as (:math:`d_i`). This can lead to issues if you merge multiple rows from
           (:math:`d_c`) to one row in (:math:`d_i`).

        3. Joining both on columns. In this case a hash join is performed using
           ``dask.dataframe.multi.hash_join``.

        In some cases, you may see a ``MemoryError`` if the ``merge`` operation requires
        an internal ``shuffle``, because shuffling places all rows that have the same
        index in the same partition. To avoid this error, make sure all rows with the
        same ``on``-column value can fit on a single partition.
        """
        result = super().merge(right, **kwargs)
        return self._propagate_metadata(result)

    def join(self, other, **kwargs):
        """Join columns of another DataFrame. Note that if `other` is a different type,
        we expect the result to have the type of this object regardless of the value
        of the`how` parameter.

        This docstring was copied from pandas.core.frame.DataFrame.join.

        Some inconsistencies with this version may exist.

        Join columns with `other` DataFrame either on index or on a key
        column. Efficiently join multiple DataFrame objects by index at once by
        passing a list.

        Parameters
        ----------
        other : DataFrame, Series, or a list containing any combination of them
            Index should be similar to one of the columns in this one. If a
            Series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined DataFrame.
        on : str, list of str, or array-like, optional
            Column or index level name(s) in the caller to join on the index
            in `other`, otherwise joins index-on-index. If multiple
            values given, the `other` DataFrame must have a MultiIndex. Can
            pass an array as the join key if it is not already contained in
            the calling DataFrame. Like an Excel VLOOKUP operation.
        how : {'left', 'right', 'outer', 'inner', 'cross'}, default 'left'
            How to handle the operation of the two objects.

            * left: use calling frame's index (or column if on is specified)
            * right: use `other`'s index.
            * outer: form union of calling frame's index (or column if on is
              specified) with `other`'s index, and sort it lexicographically.
            * inner: form intersection of calling frame's index (or column if
              on is specified) with `other`'s index, preserving the order
              of the calling's one.
            * cross: creates the cartesian product from both frames, preserves the order
              of the left keys.
        lsuffix : str, default ''
            Suffix to use from left frame's overlapping columns.
        rsuffix : str, default ''
            Suffix to use from right frame's overlapping columns.
        sort : bool, default False
            Order result DataFrame lexicographically by the join key. If False,
            the order of the join key depends on the join type (how keyword).
        validate : str, optional
            If specified, checks if join is of specified type.

            * "one_to_one" or "1:1": check if join keys are unique in both left
              and right datasets.
            * "one_to_many" or "1:m": check if join keys are unique in left dataset.
            * "many_to_one" or "m:1": check if join keys are unique in right dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.

        Returns
        -------
        result: `tape._Frame`
            A TAPE dataframe containing columns from both the caller and `other`.

        """
        result = super().join(other, **kwargs)
        return self._propagate_metadata(result)

    def drop(self, labels=None, axis=0, columns=None, errors="raise"):
        """Drop specified labels from rows or columns.

        Doc string below derived from dask.dataframe.core

        Remove rows or columns by specifying label names and corresponding
        axis, or by directly specifying index or column names. When using a
        multi-index, labels on different levels can be removed by specifying
        the level. See the :ref:`user guide <advanced.shown_levels>`
        for more information about the now unused levels.

        Parameters
        ----------
        labels : single label or list-like
            Index or column labels to drop. A tuple will be used as a single
            label and not treated as a list-like.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') or
            columns (1 or 'columns').
            is equivalent to ``index=labels``.
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are
            dropped.

        Returns
        -------
        result: `tape._Frame`
            Returns the frame or None with the specified
            index or column labels removed or None if inplace=True.
        """
        result = self._propagate_metadata(
            super().drop(labels=labels, axis=axis, columns=columns, errors=errors)
        )
        result.set_dirty(True)
        return result

    def dropna(self, **kwargs):
        """
        Remove missing values.

        Doc string below derived from dask.dataframe.core

        Parameters
        ----------

        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we have
            at least one NA or all NA.

            * 'any' : If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.

        thresh : int, optional
            Require that many non-NA values. Cannot be combined with how.
        subset : column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.

        Returns
        ----------
        result: `tape._Frame`
            The modifed frame with NA entries dropped from it or None if ``inplace=True``.
        """
        result = self._propagate_metadata(super().dropna(**kwargs))
        result.set_dirty(True)
        return result

    def persist(self, **kwargs):
        """Persist this dask collection into memory

        Doc string below derived from dask.base

        This turns a lazy Dask collection into a Dask collection with the same
        metadata, but now with the results fully computed or actively computing
        in the background.

        The action of function differs significantly depending on the active
        task scheduler.  If the task scheduler supports asynchronous computing,
        such as is the case of the dask.distributed scheduler, then persist
        will return *immediately* and the return value's task graph will
        contain Dask Future objects.  However if the task scheduler only
        supports blocking computation then the call to persist will *block*
        and the return value's task graph will contain concrete Python results.

        This function is particularly useful when using distributed systems,
        because the results will be kept in distributed memory, rather than
        returned to the local process as with compute.

        Parameters
        ----------
        **kwargs
            Extra keywords to forward to the scheduler function.

        Returns
        -------
        result: `tape._Frame`
            The modifed frame backed by in-memory data
        """
        result = super().persist(**kwargs)
        return self._propagate_metadata(result)

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

    def map_partitions(self, func, *args, **kwargs):
        """Apply Python function on each DataFrame partition.

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
        """
        result = super().map_partitions(func, *args, **kwargs)
        if isinstance(result, self.__class__):
            # If the output of func is another _Frame, let's propagate any metadata.
            return self._propagate_metadata(result)
        return result

    def compute(self, **kwargs):
        """Compute this Dask collection, returning the underlying dataframe or series.
        If tracked by an `Ensemble`, the `Ensemble` is informed of this operation and
        is given the opportunity to sync any of its tables prior to this Dask collection
        being computed.

        Doc string below derived from dask.dataframe.DataFrame.compute

        This turns a lazy Dask collection into its in-memory equivalent. For example
        a Dask array turns into a NumPy array and a Dask dataframe turns into a
        Pandas dataframe. The entire dataset must fit into memory before calling
        this operation.

        Parameters
        ----------
        scheduler: `string`, optional
            Which scheduler to use like “threads”, “synchronous” or “processes”.
            If not provided, the default is to check the global settings first,
            and then fall back to the collection defaults.
        optimize_graph: `bool`, optional
            If True [default], the graph is optimized before computation.
            Otherwise the graph is run as is. This can be useful for debugging.
        **kwargs: `dict`, optional
            Extra keywords to forward to the scheduler function.
        """
        if self.ensemble is not None:
            self.ensemble._lazy_sync_tables_from_frame(self)
        return super().compute(**kwargs)

    def repartition(
        self,
        divisions=None,
        npartitions=None,
        partition_size=None,
        freq=None,
        force=False,
    ):
        """Repartition dataframe along new divisions

        Doc string below derived from dask.dataframe.DataFrame

        Parameters
        ----------
        divisions : list, optional
            The "dividing lines" used to split the dataframe into partitions.
            For ``divisions=[0, 10, 50, 100]``, there would be three output partitions,
            where the new index contained [0, 10), [10, 50), and [50, 100), respectively.
            See https://docs.dask.org/en/latest/dataframe-design.html#partitions.
            Only used if npartitions and partition_size isn't specified.
            For convenience if given an integer this will defer to npartitions
            and if given a string it will defer to partition_size (see below)
        npartitions : int, optional
            Approximate number of partitions of output. Only used if partition_size
            isn't specified. The number of partitions used may be slightly
            lower than npartitions depending on data distribution, but will never be
            higher.
        partition_size: int or string, optional
            Max number of bytes of memory for each partition. Use numbers or
            strings like 5MB. If specified npartitions and divisions will be
            ignored. Note that the size reflects the number of bytes used as
            computed by ``pandas.DataFrame.memory_usage``, which will not
            necessarily match the size when storing to disk.

            .. warning::

               This keyword argument triggers computation to determine
               the memory size of each partition, which may be expensive.

        freq : str, pd.Timedelta
            A period on which to partition timeseries data like ``'7D'`` or
            ``'12h'`` or ``pd.Timedelta(hours=12)``.  Assumes a datetime index.
        force : bool, default False
            Allows the expansion of the existing divisions.
            If False then the new divisions' lower and upper bounds must be
            the same as the old divisions'.

        Notes
        -----
        Exactly one of `divisions`, `npartitions`, `partition_size`, or `freq`
        should be specified. A ``ValueError`` will be raised when that is
        not the case.

        Also note that ``len(divisons)`` is equal to ``npartitions + 1``. This is because ``divisions``
        represents the upper and lower bounds of each partition. The first item is the
        lower bound of the first partition, the second item is the lower bound of the
        second partition and the upper bound of the first partition, and so on.
        The second-to-last item is the lower bound of the last partition, and the last
        (extra) item is the upper bound of the last partition.

        Examples
        --------
        >>> df = df.repartition(npartitions=10)  # doctest: +SKIP
        >>> df = df.repartition(divisions=[0, 5, 10, 20])  # doctest: +SKIP
        >>> df = df.repartition(freq='7d')  # doctest: +SKIP
        """
        result = super().repartition(
            divisions=divisions,
            npartitions=npartitions,
            partition_size=partition_size,
            freq=freq,
            force=force,
        )
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
    """A barebones extension of a Dask Series for Ensemble data."""

    _partition_type = TapeSeries  # Tracks the underlying data type


class EnsembleFrame(_Frame, dd.core.DataFrame):
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
    def from_tapeframe(cls, data, npartitions=None, chunksize=None, sort=True, label=None, ensemble=None):
        """Returns an EnsembleFrame constructed from a TapeFrame.

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
            The label used to by the Ensemble to identify the frame.
        ensemble: `tape.Ensemble`, optional
            A link to the Ensemble object that owns this frame.

        Returns
        ----------
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        result = dd.from_pandas(data, npartitions=npartitions, chunksize=chunksize, sort=sort)
        result.label = label
        result.ensemble = ensemble
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

    def update_ensemble(self):
        """Updates the Ensemble linked by the `EnsembelFrame.ensemble` property to track this frame.

        Returns
        ----------
        result: `tape.Ensemble`
            The Ensemble object which tracks this frame, `None` if no such Ensemble.
        """
        if self.ensemble is None:
            return None
        # Update the Ensemble to track this frame and return the ensemble.
        return self.ensemble.update_frame(self)

    def convert_flux_to_mag(
        self,
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
            result = self.assign(**{out_col_name: lambda x: -2.5 * np.log10(x[flux_col] / x[zero_point])})

        elif zp_form == "magnitude" or zp_form == "mag":  # mag = -2.5*np.log10(flux) + zp
            result = self.assign(**{out_col_name: lambda x: -2.5 * np.log10(x[flux_col]) + x[zero_point]})
        else:
            raise ValueError(f"{zp_form} is not a valid zero_point format.")

        # Calculate Errors
        if err_col is not None:
            result = result.assign(
                **{out_col_name + "_err": lambda x: (2.5 / np.log(10)) * (x[err_col] / x[flux_col])}
            )

        return result

    def coalesce(self, input_cols, output_col, drop_inputs=False):
        """Combines multiple input columns into a single output column, with
        values equal to the first non-nan value encountered in the input cols.

        Parameters
        ----------
        input_cols: `list`
            The list of column names to coalesce into a single column.
        output_col: `str`, optional
            The name of the coalesced output column.
        drop_inputs: `bool`, optional
            Determines whether the input columns are dropped or preserved. If
            a mapped column is an input and dropped, the output column is
            automatically assigned to replace that column mapping internally.

        Returns
        -------
        ensemble: `tape.ensemble.Ensemble`
            An ensemble object.
        """

        def coalesce_partition(df, input_cols, output_col):
            """Coalescing function for a single partition (pandas dataframe)"""

            # Create a subset dataframe per input column
            # Rename column to output to allow combination
            input_dfs = []
            for col in input_cols:
                col_df = df[[col]]
                input_dfs.append(col_df.rename(columns={col: output_col}))

            # Combine each dataframe
            coal_df = input_dfs.pop()
            while input_dfs:
                coal_df = coal_df.combine_first(input_dfs.pop())

            # Assign the output column to the partition dataframe
            out_df = df.assign(**{output_col: coal_df[output_col]})

            return out_df

        table_ddf = self.map_partitions(lambda x: coalesce_partition(x, input_cols, output_col))

        # Drop the input columns if wanted
        if drop_inputs:
            if self.ensemble is not None:
                # First check to see if any dropped columns were critical columns
                current_map = self.ensemble.make_column_map().map
                cols_to_update = [key for key in current_map if current_map[key] in input_cols]

                # Theoretically a user could assign multiple critical columns in the input cols, this is very
                # likely to be a mistake, so we throw a warning here to alert them.
                if len(cols_to_update) > 1:
                    warnings.warn(
                        """Warning: Coalesce (with column dropping) is needing to update more than one
                    critical column mapping, please check that the resulting mapping is set as intended"""
                    )

                # Update critical columns to the new output column as needed
                if len(cols_to_update):  # if not zero
                    new_map = current_map
                    for col in cols_to_update:
                        new_map[col] = output_col

                    new_colmap = self.ensemble.make_column_map()
                    new_colmap.map = new_map

                    # Update the mapping
                    self.ensemble.update_column_mapping(new_colmap)

            table_ddf = table_ddf.drop(columns=input_cols)

        return table_ddf

    @classmethod
    def from_parquet(cl, path, index=None, columns=None, label=None, ensemble=None, **kwargs):
        """Returns an EnsembleFrame constructed from loading a parquet file.

        Parameters
        ----------
        path: `str` or `list`
            Source directory for data, or path(s) to individual parquet files. Prefix with a
            protocol like s3:// to read from alternative filesystems. To read from multiple
            files you can pass a globstring or a list of paths, with the caveat that they must all
            have the same protocol.
        index: `str`, `list`, `False`, optional
            Field name(s) to use as the output frame index. Default is None and index will be
            inferred from the pandas parquet file metadata, if present. Use False to read all
            fields as columns.
        columns: `str` or `list`, optional
            Field name(s) to read in as columns in the output. By default all non-index fields will
            be read (as determined by the pandas parquet metadata, if present). Provide a single
            field name instead of a list to read in the data as a Series.
        label: `str`, optional
            The label used to by the Ensemble to identify the frame.
        ensemble: `tape.ensemble.Ensemble`, optional
            A link to the Ensemble object that owns this frame.

        Returns
        ----------
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        # Read the parquet file with an engine that will assume the meta is a TapeFrame which Dask will
        # instantiate as EnsembleFrame via its dispatcher.
        result = dd.read_parquet(
            path, index=index, columns=columns, split_row_groups=True, engine=TapeArrowEngine, **kwargs
        )
        result.label = label
        result.ensemble = ensemble

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

    def __init__(self, dsk, name, meta, divisions, ensemble=None):
        super().__init__(dsk, name, meta, divisions)
        self.label = SOURCE_FRAME_LABEL  # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble  # The Ensemble object containing this frame.

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
        """Returns a SourceFrame constructed from loading a parquet file.

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
            A link to the Ensemble object that owns this frame.

        Returns
        ----------
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        # Read the source parquet file with an engine that will assume the meta is a
        # TapeSourceFrame which tells Dask to instantiate a SourceFrame via its
        # dispatcher.
        result = dd.read_parquet(
            path,
            index=index,
            columns=columns,
            split_row_groups=True,
            engine=TapeSourceArrowEngine,
        )
        result.ensemble = ensemble
        result.label = SOURCE_FRAME_LABEL

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

    def __init__(self, dsk, name, meta, divisions, ensemble=None):
        super().__init__(dsk, name, meta, divisions)
        self.label = OBJECT_FRAME_LABEL  # A label used by the Ensemble to identify this frame.
        self.ensemble = ensemble  # The Ensemble object containing this frame.

    @classmethod
    def from_parquet(
        cl,
        path,
        index=None,
        columns=None,
        ensemble=None,
    ):
        """Returns an ObjectFrame constructed from loading a parquet file.

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
            A link to the Ensemble object that owns this frame.

        Returns
        ----------
        result: `tape.ObjectFrame`
            The constructed ObjectFrame object.
        """
        # Read in the object Parquet file
        result = dd.read_parquet(
            path,
            index=index,
            columns=columns,
            split_row_groups=True,
            engine=TapeObjectArrowEngine,
        )
        result.ensemble = ensemble
        result.label = OBJECT_FRAME_LABEL

        return result

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
