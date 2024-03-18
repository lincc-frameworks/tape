import glob
import os
import json
import shutil
import warnings
import requests
import lsdb
import dask.dataframe as dd
import numpy as np
import pandas as pd

from dask.distributed import Client
from collections import Counter
from collections.abc import Iterable

from .analysis.base import AnalysisFunction
from .analysis.feature_extractor import BaseLightCurveFeature, FeatureExtractor
from .analysis.structure_function import SF_METHODS
from .analysis.structurefunction2 import calc_sf2
from .ensemble_frame import (
    EnsembleFrame,
    EnsembleSeries,
    ObjectFrame,
    SourceFrame,
    TapeFrame,
    TapeObjectFrame,
    TapeSourceFrame,
    TapeSeries,
)
from .timeseries import TimeSeries
from .utils import ColumnMapper

SOURCE_FRAME_LABEL = "source"
OBJECT_FRAME_LABEL = "object"

DEFAULT_FRAME_LABEL = "result"  # A base default label for an Ensemble's result frames.

METADATA_FILENAME = "ensemble_metadata.json"


class Ensemble:
    """Ensemble object is a collection of light curve ids"""

    def __init__(self, client=False, **kwargs):
        """Constructor of an Ensemble instance.

        Parameters
        ----------
        client: `dask.distributed.client` or `bool`, optional
            Accepts an existing `dask.distributed.Client`, or creates one if
            `client=True`, passing any additional kwargs to a
             dask.distributed.Client constructor call. If `client=False`, the
             Ensemble is created without a distributed client (default).

        """
        self.result = None  # holds the latest query

        self.frames = {}  # Frames managed by this Ensemble, keyed by label

        # A unique ID to allocate new result frame labels.
        self.default_frame_id = 1

        self.source = None  # Source Table EnsembleFrame
        self.object = None  # Object Table EnsembleFrame

        self._source_temp = []  # List of temporary columns in Source
        self._object_temp = []  # List of temporary columns in Object

        # Default to removing empty objects.
        self.keep_empty_objects = kwargs.get("keep_empty_objects", False)

        # Initialize critical column quantities
        self._id_col = None
        self._time_col = None
        self._flux_col = None
        self._err_col = None
        self._band_col = None

        self.client = None
        self.cleanup_client = False

        # Setup Dask Distributed Client if Provided
        if isinstance(client, Client):
            self.client = client
            self.cleanup_client = True
        elif client:
            self.client = Client(**kwargs)  # arguments passed along to Client
            self.cleanup_client = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cleanup_client:
            self.client.close()
        return self

    def __del__(self):
        if self.cleanup_client:
            self.client.close()
        return self

    def add_frame(self, frame, label):
        """Adds a new frame for the Ensemble to track.

        Parameters
        ----------
        frame: `tape.ensemble_frame.EnsembleFrame`
            The frame object for the Ensemble to track.
        label: `str`
            The label for the Ensemble to use to track the frame.

        Returns
        -------
        Ensemble

        Raises
        ------
        ValueError
            if the label is "source", "object", or already tracked by the Ensemble.
        """
        if label == SOURCE_FRAME_LABEL or label == OBJECT_FRAME_LABEL:
            raise ValueError(f"Unable to add frame with reserved label " f"'{label}'")
        if label in self.frames:
            raise ValueError(f"Unable to add frame: a frame with label " f"'{label}'" f"is in the Ensemble.")
        # Assign the frame to the requested tracking label.
        frame.label = label
        # Update the ensemble to track this labeled frame.
        self.update_frame(frame)
        return self

    def update_frame(self, frame):
        """Updates a frame tracked by the Ensemble or otherwise adds it to the Ensemble.
        The frame is tracked by its `EnsembleFrame.label` field.

        Parameters
        ----------
        frame: `tape.ensemble.EnsembleFrame`
            The frame for the Ensemble to update. If not already tracked, it is added.

        Returns
        -------
        Ensemble

        Raises
        ------
        ValueError
            if the `frame.label` is unpopulated, or if the frame is not a SourceFrame or ObjectFrame
            but uses the reserved labels.
        """
        if frame.label is None:
            raise ValueError(f"Unable to update frame with no populated `EnsembleFrame.label`.")
        if isinstance(frame, SourceFrame) or isinstance(frame, ObjectFrame):
            expected_label = SOURCE_FRAME_LABEL if isinstance(frame, SourceFrame) else OBJECT_FRAME_LABEL
            if frame.label != expected_label:
                raise ValueError(f"Unable to update frame with reserved label " f"'{frame.label}'")
            if isinstance(frame, SourceFrame):
                self.source = frame
            elif isinstance(frame, ObjectFrame):
                self.object = frame

        # Ensure this frame is assigned to this Ensemble.
        frame.ensemble = self
        self.frames[frame.label] = frame
        return self

    def drop_frame(self, label):
        """Drops a frame tracked by the Ensemble.

        Parameters
        ----------
        label: `str`
            The label of the frame to be dropped by the Ensemble.

        Returns
        -------
        Ensemble

        Raises
        ------
        ValueError
            if the label is "source", or "object".
        KeyError
            if the label is not tracked by the Ensemble.
        """
        if label == SOURCE_FRAME_LABEL or label == OBJECT_FRAME_LABEL:
            raise ValueError(f"Unable to drop frame with reserved label " f"'{label}'")
        if label not in self.frames:
            raise KeyError(f"Unable to drop frame: no frame with label " f"'{label}'" f"is in the Ensemble.")
        del self.frames[label]
        return self

    def select_frame(self, label):
        """Selects and returns frame tracked by the Ensemble.

        Parameters
        ----------
        label: `str`
            The label of a frame tracked by the Ensemble to be selected.

        Returns
        -------
        tape.ensemble.EnsembleFrame

        Raises
        ------
        KeyError
            if the label is not tracked by the Ensemble.
        """
        if label not in self.frames:
            raise KeyError(
                f"Unable to select frame: no frame with label" f"'{label}'" f" is in the Ensemble."
            )
        return self.frames[label]

    def frame_info(self, labels=None, verbose=True, memory_usage=True, **kwargs):
        """Wrapper for calling dask.dataframe.DataFrame.info() on frames tracked by the Ensemble.

        Parameters
        ----------
        labels: `list`, optional
            A list of labels for Ensemble frames to summarize.
            If None, info is printed for all tracked frames.
        verbose: `bool`, optional
            Whether to print the whole summary
        memory_usage: `bool`, optional
            Specifies whether total memory usage of the DataFrame elements
            (including the index) should be displayed.
        **kwargs:
            keyword arguments passed along to
            `dask.dataframe.DataFrame.info()`
        Returns
        -------
        None

        Raises
        ------
        KeyError
            if a label in labels is not tracked by the Ensemble.
        """
        if labels is None:
            labels = self.frames.keys()
        for label in labels:
            if label not in self.frames:
                raise KeyError(
                    f"Unable to get frame info: no frame with label " f"'{label}'" f" is in the Ensemble."
                )
            print(label, "Frame")
            print(self.frames[label].info(verbose=verbose, memory_usage=memory_usage, **kwargs))

    def _generate_frame_label(self):
        """Generates a new unique label for a result frame."""
        result = DEFAULT_FRAME_LABEL + "_" + str(self.default_frame_id)
        self.default_frame_id += 1  # increment to guarantee uniqueness
        while result in self.frames:
            # If the generated label has been taken by a user, increment again.
            # In most workflows, we expect the number of frames to be O(100) so it's unlikely for
            # the performance cost of this method to be high.
            result = DEFAULT_FRAME_LABEL + "_" + str(self.default_frame_id)
            self.default_frame_id += 1
        return result

    def insert_sources(
        self,
        obj_ids,
        bands,
        timestamps,
        fluxes,
        flux_errs=None,
        force_repartition=False,
        **kwargs,
    ):
        """Manually insert sources into the ensemble.

        Requires, at a minimum, the object's ID and the band, timestamp,
        and flux of the observation.

        Note
        ----
        This function is expensive and is provides mainly for testing purposes.
        Care should be used when incorporating it into the core of an analysis.

        Parameters
        ----------
        obj_ids: `list`
            A list of the sources' object ID.
        bands: `list`
            A list of the bands of the observation.
        timestamps: `list`
            A list of the times the sources were observed.
        fluxes: `list`
            A list of the fluxes of the observations.
        flux_errs: `list`, optional
            A list of the errors in the flux.
        force_repartition: `bool` optional
            Do an immediate repartition of the dataframes.
        """
        # Check the lists are all the same sizes.
        num_inserting: int = len(obj_ids)
        if num_inserting != len(bands):
            raise ValueError(f"Incorrect bands length during insert" f"{num_inserting} != {len(bands)}")
        if num_inserting != len(timestamps):
            raise ValueError(
                f"Incorrect timestamps length during insert" f"{num_inserting} != {len(timestamps)}"
            )
        if num_inserting != len(fluxes):
            raise ValueError(f"Incorrect fluxes length during insert" f"{num_inserting} != {len(fluxes)}")
        if flux_errs is not None and num_inserting != len(flux_errs):
            raise ValueError(
                f"Incorrect flux_errs length during insert" f"{num_inserting} != {len(flux_errs)}"
            )

        # Create a dictionary with the new information.
        rows = {
            self._id_col: obj_ids,
            self._band_col: bands,
            self._time_col: timestamps,
            self._flux_col: fluxes,
        }
        if flux_errs is not None:
            rows[self._err_col] = flux_errs

        # Add any other supplied columns to the dictionary.
        for key, value in kwargs.items():
            if key in self._data.columns:
                rows[key] = value

        # Create the new row and set the paritioning to match the original dataframe.
        df2 = dd.DataFrame.from_dict(rows, npartitions=1)
        df2 = df2.set_index(self._id_col, drop=True, sort=True)

        # Save the divisions and number of partitions.
        prev_div = self.source.divisions
        prev_num = self.source.npartitions

        # Append the new rows to the correct divisions.
        self.update_frame(dd.concat([self.source, df2], axis=0, interleave_partitions=True))
        self.source.set_dirty(True)

        # Do the repartitioning if requested. If the divisions were set, reuse them.
        # Otherwise, use the same number of partitions.
        if force_repartition:
            if all(prev_div):
                self.update_frame(self.source.repartition(divisions=prev_div))
            elif self.source.npartitions != prev_num:
                self.update_frame(self.source.repartition(npartitions=prev_num))

        return self

    def client_info(self):
        """Calls the Dask Client, which returns cluster information

        Parameters
        ----------
        None

        Returns
        ----------
        self.client: `distributed.client.Client`
            Dask Client information
        """
        return self.client  # Prints Dask dashboard to screen

    def info(self, verbose=True, memory_usage=True, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.info() for the Source and Object tables

        Parameters
        ----------
        verbose: `bool`, optional
            Whether to print the whole summary
        memory_usage: `bool`, optional
            Specifies whether total memory usage of the DataFrame elements
            (including the index) should be displayed.

        Returns
        ----------
        None
        """
        # Sync tables if user wants to retrieve their information
        self._lazy_sync_tables(table="all")

        print("Object Table")
        self.object.info(verbose=verbose, memory_usage=memory_usage, **kwargs)
        print("Source Table")
        self.source.info(verbose=verbose, memory_usage=memory_usage, **kwargs)

    def check_sorted(self, table="object"):
        """Checks to see if an Ensemble Dataframe is sorted (increasing) on the index.

        Parameters
        ----------
        table: `str`, optional
            The table to check.

        Returns
        -------
        boolean
            indicating whether the index is sorted (True) or not (False)
        """
        if table == "object":
            idx = self.object.index
        elif table == "source":
            idx = self.source.index
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

        # Use the existing index function to check if it's sorted (increasing)
        return idx.is_monotonic_increasing.compute()

    def check_lightcurve_cohesion(self):
        """Checks to see if lightcurves are split across multiple partitions.

        With partitioned data, and source information represented by rows, it
        is possible that when loading data or manipulating it in some way (most
        likely a repartition) that the sources for a given object will be split
        among multiple partitions. This function will check to see if all
        lightcurves are "cohesive", meaning the sources for that object only
        live in a single partition of the dataset.

        Returns
        -------
        boolean
            indicates whether the sources tied to a given object are only found
            in a single partition (True), or if they are split across multiple
            partitions (False)
        """
        idx = self.source.index
        counts = idx.map_partitions(lambda a: Counter(a.unique())).compute()

        unq_counter = counts[0]
        for i in range(1, len(counts)):
            unq_counter += counts[i]
            if any(c >= 2 for c in unq_counter.values()):
                return False
        return True

    def sort_lightcurves(self, by_band=True):
        """Sorts each Source partition first by the indexed ID column and then by
        the time column, each in ascending order.

        This allows for efficient access of lightcurves by their indexed object ID
        while still giving easy access to the sorted time series.

        Note that if the lightcurves are split across multiple partitions, this operation
        only sorts on a per-partition basis, and the table will not be globally sorted.

        You can check that no lightcurves are not split across multiple partitions by
        seeing if `Ensemble.check_lightcurve_cohesion()` is `True`.

        Parameters
        ----------
        by_band: `bool`, optional
            If True, the lightcurves are still sorted first by the indexed ID column,
            but then by band and then by timestamp, all in ascending order.

        Returns
        -------
        Ensemble
        """
        self._lazy_sync_tables(table="source")

        # Dask lacks support for multi-column sorting and indices, but if we have
        # lightcurve cohesion, we can sort each partition individually since
        # each lightcurve should only be in a single partition. We sort the Source
        # table first by its indexed ID column and then by the timestamp.
        id_col, time_col = self._id_col, self._time_col  # save column names for scoping for the lambda
        if not by_band:
            self.source.map_partitions(lambda x: x.sort_values([id_col, time_col])).update_ensemble()
        else:
            band_col = self._band_col
            self.source.map_partitions(
                lambda x: x.sort_values([id_col, band_col, time_col])
            ).update_ensemble()

        return self

    def compute(self, table=None, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.compute()

        The compute operation performs the computations that had been lazily allocated
        and returns the results as an in-memory pandas data frame.

        Parameters
        ----------
        table: `str`, optional
            The table to materialize.

        Returns
        -------
        `pd.Dataframe`
            A single pandas data frame for the specified table or a tuple of
            (object, source) data frames.
        """
        if table:
            self._lazy_sync_tables(table)
            if table == "object":
                return self.object.compute(**kwargs)
            elif table == "source":
                return self.source.compute(**kwargs)
        else:
            self._lazy_sync_tables(table="all")
            return (self.object.compute(**kwargs), self.source.compute(**kwargs))

    def persist(self, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.persist()

        The compute operation performs the computations that had been lazily allocated,
        but does not bring the results into memory or return them. This is useful
        for preventing a Dask task graph from growing too large by performing part
        of the computation.
        """
        self._lazy_sync_tables("all")
        self.update_frame(self.object.persist(**kwargs))
        self.update_frame(self.source.persist(**kwargs))

    def sample(self, frac=None, replace=False, random_state=None):
        """Selects a random sample of objects (sampling each partition).

        This sampling will be lazily applied to the SourceFrame as well. A new
        Ensemble object is created, and no additional EnsembleFrames will be
        carried into the new Ensemble object. Most of docstring copied from
        https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.sample.html.

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
        ensemble: `tape.ensemble.Ensemble`
            A new ensemble with the subset of data selected

        """

        # first do an object sync, ensure object table is up to date
        self._lazy_sync_tables(table="object")

        # sample on the object table
        object_subset = self.object.sample(frac=frac, replace=replace, random_state=random_state)

        # make a new ensemble
        if self.client is not None:
            new_ens = Ensemble(client=self.client)

            # turn off cleanups -- in the case where multiple ensembles are
            # using a client, an individual ensemble should not close the
            # client during an __exit__ or __del__ event. This means that
            # the client will not be closed without an explicit client.close()
            # call, which is unfortunate... not sure of an alternative way
            # forward.
            self.cleanup_client = False
            new_ens.cleanup_client = False
        else:
            new_ens = Ensemble(client=False)

        new_ens.update_frame(object_subset)
        new_ens.update_frame(self.source.copy())

        # sync to source, removes all tied sources
        new_ens._lazy_sync_tables(table="source")

        return new_ens

    def columns(self, table="object"):
        """Retrieve columns from dask dataframe"""
        if table == "object":
            return self.object.columns
        elif table == "source":
            return self.source.columns
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def head(self, table="object", n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.head()"""
        self._lazy_sync_tables(table)

        if table == "object":
            return self.object.head(n=n, **kwargs)
        elif table == "source":
            return self.source.head(n=n, **kwargs)
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def tail(self, table="object", n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.tail()"""
        self._lazy_sync_tables(table)

        if table == "object":
            return self.object.tail(n=n, **kwargs)
        elif table == "source":
            return self.source.tail(n=n, **kwargs)
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def dropna(self, table="source", **kwargs):
        """Removes rows with a >=`threshold` nan values.

        Parameters
        ----------
        table: `str`, optional
            A string indicating which table to filter.
            Should be one of "object" or "source".
        **kwargs:
            keyword arguments passed along to
            `dask.dataframe.DataFrame.dropna`

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with nans removed according to the threshold
            scheme
        """
        if table == "object":
            self.update_frame(self.object.dropna(**kwargs))
        elif table == "source":
            self.update_frame(self.source.dropna(**kwargs))
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

        return self

    def select(self, columns, table="object"):
        """Select a subset of columns. Modifies the ensemble in-place by dropping
        the unselected columns.

        Parameters
        ----------
        columns: `list`
            A list of column labels to keep.
        table: `str`, optional
            A string indicating which table to filter.
            Should be one of "object" or "source".
        """
        self._lazy_sync_tables(table)
        if table == "object":
            cols_to_drop = [col for col in self.object.columns if col not in columns]
            self.update_frame(self.object.drop(cols_to_drop, axis=1))
        elif table == "source":
            cols_to_drop = [col for col in self.source.columns if col not in columns]
            self.update_frame(self.source.drop(cols_to_drop, axis=1))
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def query(self, expr, table="object"):
        """Keep only certain rows of a table based on an expression of
        what information to *keep*. Wraps Dask `query`.

        Parameters
        ----------
        expr: `str`
            A string specifying the expression of what to keep.
        table: `str`, optional
            A string indicating which table to filter.
            Should be one of "object" or "source".

        Examples
        --------
        Keep sources with flux above 100.0::

            ens.query("flux > 100", table="source")

        Keep sources in the green band::

            ens.query("band_col_name == 'g'", table="source")

        Filtering on the flux column without knowing its name::

            ens.query(f"{ens._flux_col} > 100", table="source")
        """
        self._lazy_sync_tables(table)
        if table == "object":
            self.update_frame(self.object.query(expr))
        elif table == "source":
            self.update_frame(self.source.query(expr))
        return self

    def filter_from_series(self, keep_series, table="object"):
        """Filter the tables based on a DaskSeries indicating which
        rows to keep.

        Parameters
        ----------
        keep_series: `dask.dataframe.Series`
            A series mapping the table's row to a Boolean indicating
            whether or not to keep the row.
        table: `str`, optional
            A string indicating which table to filter.
            Should be one of "object" or "source".
        """
        self._lazy_sync_tables(table)
        if table == "object":
            self.update_frame(self.object[keep_series])

        elif table == "source":
            self.update_frame(self.source[keep_series])
        return self

    def assign(self, table="object", temporary=False, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.assign()

        Parameters
        ----------
        table: `str`, optional
            A string indicating which table to filter.
            Should be one of "object" or "source".
        kwargs: dict of {str: callable or Series}
            Each argument is the name of a new column to add and its value specifies
            how to fill it. A callable is called for each row and a series is copied in.
        temporary: 'bool', optional
            Dictates whether the resulting columns are flagged as "temporary"
            columns within the Ensemble. Temporary columns are dropped when
            table syncs are performed, as their information is often made
            invalid by future operations. For example, the number of
            observations information is made invalid by a filter on the source
            table. Defaults to False.

        Returns
        -------
        self: `tape.ensemble.Ensemble`
            The ensemble object.

        Examples
        --------
        Direct assignment of my_series to a column named "new_column"::

            ens.assign(table="object", new_column=my_series)

        Subtract the value in "err" from the value in "flux"::

            ens.assign(table="source", lower_bnd=lambda x: x["flux"] - 2.0 * x["err"])
        """
        self._lazy_sync_tables(table)

        if table == "object":
            pre_cols = self.object.columns
            self.update_frame(self.object.assign(**kwargs))
            post_cols = self.object.columns

            if temporary:
                self._object_temp.extend(col for col in post_cols if col not in pre_cols)

        elif table == "source":
            pre_cols = self.source.columns
            self.update_frame(self.source.assign(**kwargs))
            post_cols = self.source.columns

            if temporary:
                self._source_temp.extend(col for col in post_cols if col not in pre_cols)

        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")
        return self

    def calc_nobs(self, by_band=False, label="nobs", temporary=True):
        """Calculates the number of observations per lightcurve.

        Parameters
        ----------
        by_band: `bool`, optional
            If True, also calculates the number of observations for each band
            in addition to providing the number of observations in total
        label: `str`, optional
            The label used to generate output columns. "_total" and the band
            labels (e.g. "_g") are appended.
        temporary: 'bool', optional
            Dictates whether the resulting columns are flagged as "temporary"
            columns within the Ensemble. Temporary columns are dropped when
            table syncs are performed, as their information is often made
            invalid by future operations. For example, the number of
            observations information is made invalid by a filter on the source
            table. Defaults to True.

        Returns
        -------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with nobs columns added to the object table.
        """

        # Perform sync if either table is dirty
        self._lazy_sync_tables("all")

        if by_band:
            # repartition the result to align with object
            if self.object.known_divisions:
                # Grab these up front to help out the task graph
                id_col = self._id_col
                band_col = self._band_col

                # Get the band metadata
                unq_bands = np.unique(self.source[band_col])
                meta = {band: float for band in unq_bands}

                # Map the groupby to each partition
                band_counts = self.source.map_partitions(
                    lambda x: x.groupby(id_col)[[band_col]]
                    .value_counts()
                    .to_frame()
                    .reset_index()
                    .pivot_table(values=band_col, index=id_col, columns=band_col, aggfunc="sum"),
                    meta=meta,
                ).repartition(divisions=self.object.divisions)
            else:
                band_counts = (
                    self.source.groupby([self._id_col])[self._band_col]  # group by each object
                    .value_counts()  # count occurence of each band
                    .to_frame()  # convert series to dataframe
                    .rename(columns={self._band_col: "counts"})  # rename column
                    .reset_index()  # break up the multiindex
                    .categorize(columns=[self._band_col])  # retype the band labels as categories
                    .pivot_table(
                        values=self._band_col, index=self._id_col, columns=self._band_col, aggfunc="sum"
                    )
                )  # the pivot_table call makes each band_count a column of the id_col row

                band_counts = band_counts.repartition(npartitions=self.object.npartitions)

            # short-hand for calculating nobs_total
            band_counts["total"] = band_counts[list(band_counts.columns)].sum(axis=1)

            bands = band_counts.columns.values
            self.object.assign(
                **{label + "_" + str(band): band_counts[band] for band in bands}
            ).update_ensemble()

            if temporary:
                self._object_temp.extend(label + "_" + str(band) for band in bands)

        else:
            if self.object.known_divisions and self.source.known_divisions:
                # Grab these up front to help out the task graph
                id_col = self._id_col
                band_col = self._band_col

                # Map the groupby to each partition
                counts = self.source.map_partitions(
                    lambda x: x.groupby([id_col])[[band_col]].aggregate("count")
                ).repartition(divisions=self.object.divisions)
            else:
                # Just do a groupby on all source
                counts = (
                    self.source.groupby([self._id_col])[[self._band_col]]
                    .aggregate("count")
                    .repartition(npartitions=self.object.npartitions)
                )

            self.object.assign(**{label + "_total": counts[self._band_col]}).update_ensemble()

            if temporary:
                self._object_temp.extend([label + "_total"])

        return self

    def prune(self, threshold=50, col_name=None):
        """remove objects with less observations than a given threshold

        Parameters
        ----------
        threshold: `int`, optional
            The minimum number of observations needed to retain an object.
            Default is 50.
        col_name: `str`, optional
            The name of the column to assess the threshold if available in
            the object table. If not specified, the ensemble will calculate
            the number of observations and filter on the total (sum across
            bands).

        Returns
        -------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with pruned rows removed
        """

        # Sync Required if source is dirty
        self._lazy_sync_tables(table="object")

        if not col_name:
            self.calc_nobs(label="nobs")
            col_name = "nobs_total"

        # Mask on object table
        self = self.query(f"{col_name} >= {threshold}", table="object")

        self.object.set_dirty(True)  # Object table is now dirty

        return self

    def find_day_gap_offset(self):
        """Finds an approximation of the MJD offset for noon at the
        observatory.

        This function looks for the longest strecth of hours of the day
        with zero observations. This gap is treated as the daylight hours
        and the function returns the middle hour of the gap. This is used
        for automatically finding offsets for binning.

        Returns
        -------
        empty_hours: `list`
            The estimated middle of the day as a floating point day. Returns
            -1.0 if no such time is found.

        Note
        ----
        Calls a compute on the source table.
        """
        self._lazy_sync_tables(table="source")

        # Compute a histogram of observations by hour of the day.
        hours = self.source[self._time_col].apply(
            lambda x: np.floor(x * 24.0).astype(int) % 24, meta=pd.Series(dtype=int)
        )
        hour_counts = hours.value_counts().compute()

        # Find the longest run of hours with no observations.
        start_hr = 0
        best_length = 0
        best_mid_pt = -1
        while start_hr < 24:
            # Find the end of the run of zero starting at `start_hr`.
            # Note that this might wrap around 24->0 hours.
            end_hr = start_hr
            while end_hr < 48 and (end_hr % 24) not in hour_counts.index:
                end_hr += 1

            # If we have found a new longest gap, record it.
            if end_hr - start_hr > best_length:
                best_length = end_hr - start_hr
                best_mid_pt = (start_hr + end_hr) / 2.0

            # Move to the next block.
            start_hr = end_hr + 1

        if best_length == 0:
            return -1
        return (best_mid_pt % 24.0) / 24.0

    def bin_sources(
        self, time_window=1.0, offset=0.0, custom_aggr=None, count_col=None, use_map=True, **kwargs
    ):
        """Bin sources on within a given time range to improve the estimates.

        Parameters
        ----------
        time_window : `float`, optional
            The time range (in days) over which to consider observations in the same bin.
            The default is 1.0 days.
        offset : `float`, optional
            The offset in days to use for binning. This should correspond to the middle
            of the daylight hours for the observatory. Default is 0.0.
            This value can also be computed with find_day_gap_offset.
        custom_aggr : `dict`, optional
            A dictionary mapping column name to aggregation method. This can be used to
            both include additional columns to aggregate OR overwrite the aggregation
            method for time, flux, or flux error by matching those column names.
            Example: {"my_value_1": "mean", "my_value_2": "max", "psFlux": "sum"}
        count_col : `str`, optional
            The name of the column in which to count the number of sources per bin.
            If None then it does not include this column.
        use_map : `boolean`, optional
            Determines whether `dask.dataframe.DataFrame.map_partitions` is
            used (True). Using map_partitions is generally more efficient, but
            requires the data from each lightcurve is housed in a single
            partition. If False, a groupby will be performed instead.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with pruned rows removed

        Notes
        -----
        * This should only be used for slowly varying sources where we can
          treat the source as constant within `time_window`.

        * As a default the function only aggregates and keeps the id, band,
          time, flux, and flux error columns. Additional columns can be preserved
          by providing the mapping of column name to aggregation function with the
          `additional_cols` parameter.
        """
        self._lazy_sync_tables(table="source")

        # Bin the time and add it as a column. We create a temporary column that
        # truncates the time into increments of `time_window`.
        tmp_time_col = "tmp_time_for_aggregation"
        if tmp_time_col in self.source.columns:
            raise KeyError(f"Column '{tmp_time_col}' already exists in source table.")
        self.source[tmp_time_col] = self.source[self._time_col].apply(
            lambda x: np.floor((x + offset) / time_window) * time_window, meta=pd.Series(dtype=float)
        )

        # Set up the aggregation functions for the time and flux columns.
        aggr_funs = {self._time_col: "mean", self._flux_col: "mean"}

        # If the source table has errors then add an aggregation function for it.
        if self._err_col in self.source.columns:
            aggr_funs[self._err_col] = dd.Aggregation(
                name="err_agg",
                chunk=lambda x: (x.count(), x.apply(lambda s: np.sum(np.power(s, 2)))),
                agg=lambda c, s: (c.sum(), s.sum()),
                finalize=lambda c, s: np.sqrt(s) / c,
            )

        # Handle the aggregation function for the bin count, including
        # adding an initial column of all ones if needed.
        if count_col is not None:
            self._bin_count_col = count_col
            if self._bin_count_col not in self.source.columns:
                self.source[self._bin_count_col] = self.source[self._time_col].apply(
                    lambda x: 1, meta=pd.Series(dtype=int)
                )
            aggr_funs[self._bin_count_col] = "sum"

        # Add any additional aggregation functions
        if custom_aggr is not None:
            for key in custom_aggr:
                # Warn the user if they are overwriting a predefined aggregation function.
                if key in aggr_funs:
                    warnings.warn(f"Warning: Overwriting aggregation function for column {key}.")
                aggr_funs[key] = custom_aggr[key]

        # Group the columns by id, band, and time bucket and aggregate.
        self.update_frame(
            self.source.groupby([self._id_col, self._band_col, tmp_time_col]).aggregate(aggr_funs)
        )

        # Fix the indices and remove the temporary column.
        self.update_frame(self.source.reset_index().set_index(self._id_col).drop(tmp_time_col, axis=1))

        # Mark the source table as dirty.
        self.source.set_dirty(True)
        return self

    def batch(
        self,
        func,
        *args,
        meta=None,
        by_band=False,
        use_map=True,
        on=None,
        label="",
        **kwargs,
    ):
        """Run a function from tape.TimeSeries on the available ids

        Parameters
        ----------
        func : `function`
            A function to apply to all objects in the ensemble. The function
            could be a TAPE function, an initialized feature extractor from
            `light-curve` package or a user-defined function. In the least
            case the function must have the following signature:
            `func(*cols, **kwargs)`, where the names of the `cols` are
            specified in `args`, `kwargs` are keyword arguments passed to the
            function, and the return value schema is described by `meta`.
            For TAPE and `light-curve` functions `args`, `meta` and `on` are
            populated automatically.
        *args:
            Denotes the ensemble columns to use as inputs for a function,
            order must be correct for function. If passing a TAPE
            or `light-curve` function, these are populated automatically.
        meta : `pd.Series`, `pd.DataFrame`, `dict`, or `tuple-like`
            Dask's meta parameter, which lays down the expected structure of
            the results. Overridden by TAPE for TAPE and `light-curve`
            functions. If none, attempts to coerce the result to a
            pandas.Series.
        by_band: `boolean`, optional
            If true, the lightcurves are split into separate inputs for each
            band and passed along to the function individually. If the band
            column is already specified in `on` then `batch` will ensure the
            band column is the final element in `on`. For all original columns
            outputted by `func`, by_band will generate a set of new columns per
            band (for example, a function with output column "result" will
            instead have "result_g" and "result_r" as columns if the data had g
            and r band data) If False (default), the full lightcurve is passed
            along to the function (assuming the band column in not already part
            of `on`)
        use_map : `boolean`
            Determines whether `dask.dataframe.DataFrame.map_partitions` is
            used (True). Using map_partitions is generally more efficient, but
            requires the data from each lightcurve is housed in a single
            partition. This can be checked using
            `Ensemble.check_lightcurve_cohesion`. If False, a groupby will be
            performed instead.
        on: 'str' or 'list', optional
            Designates which column(s) to groupby. Columns may be from the
            source or object tables. If not specified, then the id column is
            used by default. For TAPE and `light-curve` functions this is
            populated automatically.
        label: 'str', optional
            If provided the ensemble will use this label to track the result
            dataframe. If not provided, a label of the from "result_{x}" where x
            is a monotonically increasing integer is generated. If `None`,
            the result frame will not be tracked.
        **kwargs:
            Additional optional parameters passed for the selected function

        Returns
        -------
        result: `Dask.Series`
            Series of function results

        Examples
        --------
        Run a TAPE function on the ensemble::

            from tape.analysis.stetsonj import calc_stetson_J
            ens = Ensemble().from_dataset('rrlyr82')
            ensemble.batch(calc_stetson_J, band_to_calc='i')

        Run a light-curve function on the ensemble::

            from light_curve import EtaE
            ens.batch(EtaE(), band_to_calc='g')

        Run a custom function on the ensemble::

            def s2n_inter_quartile_range(flux, err):
            first, third = np.quantile(flux / err, [0.25, 0.75])
            return third - first

            ens.batch(s2n_inter_quartile_range, ens._flux_col, ens._err_col)

        Or even a numpy built-in function::

            amplitudes = ens.batch(np.ptp, ens._flux_col)
        """

        self._lazy_sync_tables(table="all")

        # Convert light-curve package feature into analysis function
        if isinstance(func, BaseLightCurveFeature):
            func = FeatureExtractor(func)
        # Extract function information if TAPE analysis function
        if isinstance(func, AnalysisFunction):
            args = func.cols(self)
            meta = func.meta(self)
            on = func.on(self)

        if meta is None:
            meta = (self._id_col, float)  # return a series of ids, default assume a float is returned

        # Translate the meta into an appropriate TapeFrame or TapeSeries. This ensures that the
        # batch result will be an EnsembleFrame or EnsembleSeries.
        meta = self._translate_meta(meta)

        if on is None:
            on = self._id_col  # Default grouping is by id_col
        if isinstance(on, str):
            on = [on]  # Convert to list if only one column is passed

        if by_band:
            if self._band_col not in on:
                on += [self._band_col]
            elif on[-1] != self._band_col:
                # Ensure band is the final column in the `on` list
                on[on.index(self._band_col)] = on[-1]
                on[-1] = self._band_col

        # Handle object columns to group on
        source_cols = list(self.source.columns)
        object_cols = list(self.object.columns)
        object_group_cols = [col for col in on if (col in object_cols) and (col not in source_cols)]

        if len(object_group_cols) > 0:
            object_col_dd = self.object[object_group_cols]
            source_to_batch = self.source.merge(object_col_dd, how="left")
        else:
            source_to_batch = self.source  # Can directly use the source table

        id_col = self._id_col  # pre-compute needed for dask in lambda function

        def _apply_func_to_lc(lc, func, *args, **kwargs):
            """
            Apply a batch function to a lightcurve
            """
            return func(
                *[lc[arg].to_numpy() if arg != id_col else lc.index.to_numpy() for arg in args],
                **kwargs,
            )

        if use_map:  # use map_partitions

            def _batch_apply(df, func, on, *args, **kwargs):
                """
                Apply a function to a partition of the dataframe
                """
                return df.groupby(on, group_keys=True, sort=False).apply(
                    _apply_func_to_lc, func, *args, **kwargs
                )

            id_col = self._id_col  # need to grab this before mapping

            batch = source_to_batch.map_partitions(_batch_apply, func, on, *args, **kwargs, meta=meta)

        else:  # use groupby
            # don't use _batch_apply as meta must be specified in the apply call
            batch = source_to_batch.groupby(on, group_keys=True, sort=False).apply(
                _apply_func_to_lc,
                func,
                *args,
                **kwargs,
                meta=meta,
            )

        # Output standardization
        batch = self._standardize_batch(batch, on, by_band)

        # Inherit divisions if known from source and the resulting index is the id
        # Groupby on index should always return a subset that adheres to the same divisions criteria
        if self.source.known_divisions and batch.index.name == self._id_col:
            batch.divisions = self.source.divisions

        if label is not None:
            if label == "":
                label = self._generate_frame_label()
                print(f"Using generated label, {label}, for a batch result.")
            # Track the result frame under the provided label
            self.add_frame(batch, label)

        return batch

    def _standardize_batch(self, batch, on, by_band):
        """standardizes the output of a batch result"""

        # Do some up front type checking
        if isinstance(batch, EnsembleSeries):
            # make sure the output is separated from the id column
            if batch.name == self._id_col:
                batch = batch.rename("result")
            res_cols = [batch.name]  # grab the series name to use as a column label

            # convert the series to an EnsembleFrame object
            batch = EnsembleFrame.from_dask_dataframe(batch.to_frame())

        elif isinstance(batch, EnsembleFrame):
            # collect output columns
            res_cols = list(batch._meta.columns)

        else:
            # unclear if there's really a pathway to trigger this, but added for completeness
            raise TypeError(
                f"The output type of batch ({type(batch)}) does not match any of the expected types: (EnsembleFrame, EnsembleSeries)"
            )

        # Handle formatting for multi-index results
        if len(on) > 1:
            batch = batch.reset_index()

            # Need to overwrite the meta manually as the multiindex will be
            # interpretted by dask as a single "index" column
            batch._meta = TapeFrame(columns=on + res_cols)

            # Further reformatting for per-band results
            # Pivots on the band column to generate a result column for each
            # photometric band.
            if by_band:
                batch = batch.categorize(self._band_col)
                batch = batch.pivot_table(index=on[0], columns=self._band_col, aggfunc="sum")

                # Need to once again reestablish meta for the pivot
                band_labels = batch.columns.values
                out_cols = []
                # To align with pandas pivot_table results, the columns should be generated in reverse order
                for col in res_cols[::-1]:
                    for band in band_labels:
                        out_cols += [(str(col), str(band))]
                batch._meta = TapeFrame(columns=out_cols)  # apply new meta

                # Flatten the columns to a new column per band
                batch.columns = ["_".join(col) for col in batch.columns.values]

                # The pivot returns a dask dataframe, need to convert back
                batch = EnsembleFrame.from_dask_dataframe(batch)
            else:
                batch = batch.set_index(on[0], sort=False)

        return batch

    def save_ensemble(self, path=".", dirname="ensemble", additional_frames=True, **kwargs):
        """Save the current ensemble frames to disk.

        Parameters
        ----------
        path: 'str' or path-like, optional
            A path to the desired location of the top-level save directory, by
            default this is the current working directory.
        dirname: 'str', optional
            The name of the saved ensemble directory, "ensemble" by default.
        additional_frames: bool, or list, optional
            Controls whether EnsembleFrames beyond the Object and Source Frames
            are saved to disk. If True or False, this specifies whether all or
            none of the additional frames are saved. Alternatively, a list of
            EnsembleFrame names may be provided to specify which frames should
            be saved. Object and Source will always be added and do not need to
            be specified in the list. By default, all frames will be saved.
        **kwargs:
            Additional kwargs passed along to EnsembleFrame.to_parquet()

        Returns
        ----------
        None

        Note
        ----
        If the object frame has no columns, which is often the case when an
        Ensemble is constructed using only source files/dictionaries, then an
        object subdirectory will not be created. `Ensemble.from_ensemble` will
        know how to work with the directory whether or not the object
        subdirectory is present.

        Be careful about repeated saves to the same directory name. This will
        not be a perfect overwrite, as any products produced by a previous save
        may not be deleted by successive saves if they are removed from the
        ensemble. For best results, delete the directory between saves or
        verify that the contents are what you would expect.
        """

        self._lazy_sync_tables("all")

        # Determine the path
        ens_path = os.path.join(path, dirname)

        # First look for an existing metadata file in the path
        try:
            with open(os.path.join(ens_path, METADATA_FILENAME), "r") as oldfile:
                # Reading from json file
                old_metadata = json.load(oldfile)
                old_subdirs = old_metadata["subdirs"]
                # Delete any old subdirectories
                for subdir in old_subdirs:
                    shutil.rmtree(os.path.join(ens_path, subdir))
        except FileNotFoundError:
            pass

        # Compile frame list
        if additional_frames is True:
            frames_to_save = list(self.frames.keys())  # save all frames
        elif additional_frames is False:
            frames_to_save = [OBJECT_FRAME_LABEL, SOURCE_FRAME_LABEL]  # save just object and source
        elif isinstance(additional_frames, Iterable):
            frames_to_save = set(additional_frames)
            invalid_frames = frames_to_save.difference(set(self.frames.keys()))
            # Raise an error if any frames were not found in the frame list
            if len(invalid_frames) != 0:
                raise ValueError(
                    f"The frame(s): {invalid_frames} specified in `additional_frames` were not found in the frame list."
                )
            frames_to_save = list(frames_to_save)

            # Make sure object and source are in the frame list
            if OBJECT_FRAME_LABEL not in frames_to_save:
                frames_to_save.append(OBJECT_FRAME_LABEL)
            if SOURCE_FRAME_LABEL not in frames_to_save:
                frames_to_save.append(SOURCE_FRAME_LABEL)
        else:
            raise ValueError("Invalid input to `additional_frames`, must be boolean or list-like")

        # Generate the metadata first
        created_subdirs = []  # track the list of created subdirectories
        divisions_known = []  # log whether divisions were known for each frame
        for frame_label in frames_to_save:
            # grab the dataframe from the frame label
            frame = self.frames[frame_label]

            # When the frame has no columns, avoid the save as parquet doesn't handle it
            # Most commonly this applies to the object table when it's built from source
            if len(frame.columns) == 0:
                print(f"Frame: {frame_label} will not be saved as no columns are present.")
                continue

            created_subdirs.append(frame_label)
            divisions_known.append(frame.known_divisions)

        # Save a metadata file
        col_map = self.make_column_map()  # grab the current column_mapper
        metadata = {
            "subdirs": created_subdirs,
            "known_divisions": divisions_known,
            "column_mapper": col_map.map,
        }
        json_metadata = json.dumps(metadata, indent=4)

        # Make the directory if it doesn't already exist
        os.makedirs(ens_path, exist_ok=True)
        with open(os.path.join(ens_path, METADATA_FILENAME), "w") as outfile:
            outfile.write(json_metadata)

        # Now write out the frames to subdirectories
        for subdir in created_subdirs:
            self.frames[subdir].to_parquet(os.path.join(ens_path, subdir), write_metadata_file=True, **kwargs)

        print(f"Saved to {os.path.join(path, dirname)}")

        return

    def from_ensemble(
        self,
        dirpath,
        additional_frames=True,
        column_mapper=None,
        **kwargs,
    ):
        """Load an ensemble from an on-disk ensemble.

        Parameters
        ----------
        dirpath: 'str' or path-like, optional
            A path to the top-level ensemble directory to load from.
        additional_frames: bool, or list, optional
            Controls whether EnsembleFrames beyond the Object and Source Frames
            are loaded from disk. If True or False, this specifies whether all
            or none of the additional frames are loaded. Alternatively, a list
            of EnsembleFrame names may be provided to specify which frames
            should be loaded. Object and Source will always be added and do not
            need to be specified in the list. By default, all frames will be
            loaded.
        column_mapper: Tape.ColumnMapper object, or None, optional
            Supplies a ColumnMapper to the Ensemble, if None (default) searches
            for a column_mapper.npy file in the directory, which should be
            created when the ensemble is saved.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object.
        """

        # Read in the metadata file
        with open(os.path.join(dirpath, METADATA_FILENAME), "r") as metadatafile:
            # Reading from json file
            metadata = json.load(metadatafile)

            # Load in the metadata
            subdirs = metadata["subdirs"]
            frame_known_divisions = metadata["known_divisions"]
            if column_mapper is None:
                column_mapper = ColumnMapper()
                column_mapper.map = metadata["column_mapper"]

        # Load Object and Source

        # Check for whether or not object is present, it's not saved when no columns are present
        if OBJECT_FRAME_LABEL in subdirs:
            # divisions should be known for both tables to use the sorted kwarg
            use_sorted = (
                frame_known_divisions[subdirs.index(OBJECT_FRAME_LABEL)]
                and frame_known_divisions[subdirs.index(SOURCE_FRAME_LABEL)]
            )

            self.from_parquet(
                os.path.join(dirpath, SOURCE_FRAME_LABEL),
                os.path.join(dirpath, OBJECT_FRAME_LABEL),
                column_mapper=column_mapper,
                sorted=use_sorted,
                sort=False,
                sync_tables=False,  # a sync should always be performed just before saving
                **kwargs,
            )
        else:
            use_sorted = frame_known_divisions[subdirs.index(SOURCE_FRAME_LABEL)]
            self.from_parquet(
                os.path.join(dirpath, SOURCE_FRAME_LABEL),
                column_mapper=column_mapper,
                sorted=use_sorted,
                sort=False,
                sync_tables=False,  # a sync should always be performed just before saving
                **kwargs,
            )

        # Load all remaining frames
        if additional_frames is False:
            return self  # we are all done
        else:
            if additional_frames is True:
                #  Grab all subdirectory paths in the top-level folder, filter out any files
                frames_to_load = [os.path.join(dirpath, f) for f in subdirs]
            elif isinstance(additional_frames, Iterable):
                frames_to_load = [os.path.join(dirpath, frame) for frame in additional_frames]
            else:
                raise ValueError("Invalid input to `additional_frames`, must be boolean or list-like")

            # Filter out object and source from additional frames
            frames_to_load = [
                frame
                for frame in frames_to_load
                if os.path.split(frame)[1] not in [OBJECT_FRAME_LABEL, SOURCE_FRAME_LABEL]
            ]
            if len(frames_to_load) > 0:
                for frame in frames_to_load:
                    label = os.path.split(frame)[1]
                    use_divisions = frame_known_divisions[subdirs.index(label)]
                    ddf = EnsembleFrame.from_parquet(
                        frame, label=label, calculate_divisions=use_divisions, **kwargs
                    )
                    self.add_frame(ddf, label)

            return self

    def from_pandas(
        self,
        source_frame,
        object_frame=None,
        column_mapper=None,
        sync_tables=True,
        npartitions=None,
        partition_size=None,
        **kwargs,
    ):
        """Read in Pandas dataframe(s) into an ensemble object

        Parameters
        ----------
        source_frame: 'pandas.Dataframe'
            A Dask dataframe that contains source information to be read into the ensemble
        object_frame: 'pandas.Dataframe', optional
            If not specified, the object frame is generated from the source frame
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        sync_tables: 'bool', optional
            In the case where an `object_frame`is provided, determines whether an
            initial sync is performed between the object and source tables. If
            not performed, dynamic information like the number of observations
            may be out of date until a sync is performed internally.
        npartitions: `int`, optional
            If specified, attempts to repartition the ensemble to the specified
            number of partitions
        partition_size: `int`, optional
            If specified, attempts to repartition the ensemble to partitions
            of size `partition_size`.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with the Dask dataframe data loaded.
        """
        # Construct Dask DataFrames of the source and object tables
        source = dd.from_pandas(source_frame, npartitions=npartitions)
        object = None if object_frame is None else dd.from_pandas(object_frame, npartitions=npartitions)
        return self.from_dask_dataframe(
            source,
            object_frame=object,
            column_mapper=column_mapper,
            sync_tables=sync_tables,
            npartitions=npartitions,
            partition_size=partition_size,
            **kwargs,
        )

    def from_dask_dataframe(
        self,
        source_frame,
        object_frame=None,
        column_mapper=None,
        sync_tables=True,
        npartitions=None,
        partition_size=None,
        sorted=False,
        sort=False,
        **kwargs,
    ):
        """Read in Dask dataframe(s) into an ensemble object

        Parameters
        ----------
        source_frame: 'dask.Dataframe'
            A Dask dataframe that contains source information to be read into the ensemble
        object_frame: 'dask.Dataframe', optional
            If not specified, the object frame is generated from the source frame
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        sync_tables: 'bool', optional
            In the case where an `object_frame`is provided, determines whether an
            initial sync is performed between the object and source tables.
        npartitions: `int`, optional
            If specified, attempts to repartition the ensemble to the specified
            number of partitions
        partition_size: `int`, optional
            If specified, attempts to repartition the ensemble to partitions
            of size `partition_size`.
        sorted: bool, optional
            If the index column is already sorted in increasing order.
            Defaults to False
        sort: `bool`, optional
            If True, sorts the DataFrame by the id column. Otherwise set the
            index on the individual existing partitions. Defaults to False.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with the Dask dataframe data loaded.
        """
        self._load_column_mapper(column_mapper, **kwargs)
        source_frame = SourceFrame.from_dask_dataframe(source_frame, self)

        # Repartition before any sorting
        if npartitions and npartitions > 1:
            source_frame = source_frame.repartition(npartitions=npartitions)
        elif partition_size:
            source_frame = source_frame.repartition(partition_size=partition_size)

        # Set the index of the source frame and save the resulting table
        if source_frame.index.name != self._id_col:  # prevents a potential no-op
            self.update_frame(source_frame.set_index(self._id_col, drop=True, sorted=sorted, sort=sort))
        else:
            self.update_frame(source_frame)  # the index is already set

        if object_frame is None:  # generate an indexed object table from source
            self.update_frame(self._generate_object_table())

        else:
            self.update_frame(ObjectFrame.from_dask_dataframe(object_frame, ensemble=self))
            if object_frame.index.name != self._id_col:  # prevents a potential no-op
                self.update_frame(self.object.set_index(self._id_col, sorted=sorted, sort=sort))

            # Optionally sync the tables, recalculates nobs columns
            if sync_tables:
                self.source.set_dirty(True)
                self.object.set_dirty(True)
                self._sync_tables()

        # Check that Divisions are established, warn if not.
        for name, table in [("object", self.object), ("source", self.source)]:
            if not table.known_divisions:
                warnings.warn(
                    f"Divisions for {name} are not set, certain downstream dask operations may fail as a result. We recommend setting the `sort` or `sorted` flags when loading data to establish division information."
                )
        return self

    def from_lsdb(
        self,
        source_catalog,
        object_catalog=None,
        column_mapper=None,
        sync_tables=False,
        sorted=True,
        sort=False,
    ):
        """Read in from LSDB catalog objects.

        Parameters
        ----------
        source_catalog: 'dask.Dataframe'
            An LSDB catalog that contains source information to be read into
            the ensemble.
        object_catalog: 'dask.Dataframe', optional
            An LSDB catalog containing object information. If not specified,
            a minimal ObjectFrame is generated from the source catalog.
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        sync_tables: 'bool', optional
            In the case where an `object_catalog`is provided, determines
            whether an initial sync is performed between the object and source
            tables. Defaults to False.
        sorted: bool, optional
            If the index column is already sorted in increasing order.
            Defaults to True.
        sort: `bool`, optional
            If True, sorts the DataFrame by the id column. Otherwise set the
            index on the individual existing partitions. Defaults to False.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with the LSDB catalog data loaded.
        """

        # Support for just source catalog is somewhat involved
        # The code below mainly tries to catch a few common pitfalls
        if object_catalog is None:
            # This is tricky, so just raise an error
            if column_mapper.map["id_col"] == "_hipscat_index":
                raise ValueError(
                    "Using the _hipscat_index as the id column is not advised without a specified object catalog, as the _hipscat_index is unique per source in this case. Use an object-level id.",
                )
            # And if they didn't choose _hipscat_index, it's almost certainly not sorted
            # Let's try to catch a bad sorted set, and reroute to sort for better user experience
            else:
                if sorted is True:
                    warnings.warn(
                        f" The sorted flag was set true with a non _hipscat_index id column ({column_mapper.map['id_col']}). This dataset is sorted by _hipscat_index, so the sorted flag has been turned off and sort has been turned on."
                    )
                    sorted = False
                    sort = True

            self.from_dask_dataframe(
                source_catalog._ddf,
                None,
                column_mapper=column_mapper,
                sync_tables=sync_tables,
                sorted=sorted,
                sort=sort,
                npartitions=None,
                partition_size=None,
            )

        # When we have both object and source, it's much simpler
        else:
            # We are still vulnerable to users choosing a non-_hipscat_index
            # Just warn them, though it's likely the function call will fail
            if column_mapper.map["id_col"] != "_hipscat_index":
                warnings.warn(
                    f"With hipscat data, it's advised to use the _hipscat_index as the id_col (instead of {column_mapper.map['id_col']}), as the data is sorted using this column. If you'd like to use your chosen id column, make sure it's in both catalogs and use sort=True and sorted=False (these have been auto-set for this call)",
                    UserWarning,
                )
                sorted = False
                sort = True

            self.from_dask_dataframe(
                source_catalog._ddf,
                object_catalog._ddf,
                column_mapper=column_mapper,
                sync_tables=sync_tables,
                sorted=sorted,
                sort=sort,
                npartitions=None,
                partition_size=None,
            )

        return self

    def from_hipscat(
        self,
        source_path,
        object_path=None,
        column_mapper=None,
        source_index=None,
        object_index=None,
        sorted=True,
        sort=False,
    ):
        """Use LSDB to read from a hipscat directory.

        This function utilizes LSDB for reading a hipscat directory into TAPE.
        In cases where a user would like to do operations on the LSDB catalog
        objects, it's best to use LSDB itself first, and then load the result
        into TAPE using `tape.Ensemble.from_lsdb`. A join is performed between
        the two tables to modify the source table to use the object index,
        using `object_index` and `source_index`.

        Parameters
        ----------
        source_path: str or Path
            A hipscat directory that contains source information to be read
            into the ensemble.
        object_path: str or Path, optional
            A hipscat directory containing object information. If not
            specified, a minimal ObjectFrame is generated from the sources.
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        object_index: 'str', optional
            The join index of the object table, should be the label for the
            object ID contained in the object table.
        source_index: 'str', optional
            The join index of the source table, should be the label for the
            object ID contained in the source table.
        sorted: bool, optional
            If the index column is already sorted in increasing order.
            Defaults to True.
        sort: `bool`, optional
            If True, sorts the DataFrame by the id column. Otherwise set the
            index on the individual existing partitions. Defaults to False.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with the hipscat data loaded.
        """

        # After margin/associated caches are implemented in LSDB, we should use them here
        source_catalog = lsdb.read_hipscat(source_path)

        if object_path is not None:
            object_catalog = lsdb.read_hipscat(object_path)

            # We do this to get the source catalog indexed by the objects hipscat index
            # Very specifically need object.join(source)
            joined_source_catalog = object_catalog.join(
                source_catalog,
                left_on=object_index,
                right_on=source_index,
                suffixes=("_drop_these_cols", ""),
            )

        else:
            object_catalog = None
            joined_source_catalog = source_catalog

        # We should also set index column to be object's _hipscat_index
        self.from_lsdb(
            joined_source_catalog,
            object_catalog,
            column_mapper=column_mapper,
            sync_tables=False,  # should never need to sync, the join does it for us
            sorted=sorted,
            sort=sort,
        )

        # drop the extra object columns from source
        if object_path is not None:
            cols_to_drop = [col for col in self.source.columns if col.endswith("_drop_these_cols")]
            self.source.drop(columns=cols_to_drop).update_ensemble()
        return self

    def make_column_map(self):
        """Returns the current column mapping.

        Returns
        -------
        result: `tape.utils.ColumnMapper`
            A new column mapper representing the Ensemble's current mappings.
        """
        result = ColumnMapper(
            id_col=self._id_col,
            time_col=self._time_col,
            flux_col=self._flux_col,
            err_col=self._err_col,
            band_col=self._band_col,
        )
        return result

    def update_column_mapping(self, column_mapper=None, **kwargs):
        """Update the mapping of column names.

        Parameters
        ----------
        column_mapper: `tape.utils.ColumnMapper`, optional
            An entirely new mapping of column names. If `None` then modifies the
            current mapping using kwargs.
        kwargs:
            Individual column to name settings.

        Returns
        -------
        self: `Ensemble`
        """
        if column_mapper is not None:
            self._load_column_mapper(column_mapper, **kwargs)
        else:
            column_mapper = self.make_column_map()
            column_mapper.assign(**kwargs)
            self._load_column_mapper(column_mapper, **kwargs)
        return self

    def _load_column_mapper(self, column_mapper, **kwargs):
        """Load a column mapper object.

        Parameters
        ----------
        column_mapper: `tape.utils.ColumnMapper` or None
            The `ColumnMapper` to use. If `None` then the function
            creates a new one from kwargs.
        kwargs: optional
            Individual column to name settings.

        Returns
        -------
        self: `Ensemble`

        Raises
        ------
        ValueError if a required column is missing.
        """
        if column_mapper is None:
            column_mapper = ColumnMapper(**kwargs)

        ready, needed = column_mapper.is_ready(show_needed=True)

        if ready:
            # Assign Critical Columns
            self._id_col = column_mapper.map["id_col"]
            self._time_col = column_mapper.map["time_col"]
            self._flux_col = column_mapper.map["flux_col"]
            self._err_col = column_mapper.map["err_col"]
            self._band_col = column_mapper.map["band_col"]
        else:
            raise ValueError(f"Missing required column mapping information: {needed}")

        return self

    def from_parquet(
        self,
        source_file,
        object_file=None,
        column_mapper=None,
        sync_tables=True,
        additional_cols=True,
        npartitions=None,
        partition_size=None,
        sorted=False,
        sort=False,
        **kwargs,
    ):
        """Read in parquet file(s) into an ensemble object

        Parameters
        ----------
        source_file: 'str'
            Path to a parquet file, or multiple parquet files that contain
            source information to be read into the ensemble
        object_file: 'str', optional
            Path to a parquet file, or multiple parquet files that contain
            object information. If not specified, it is generated from the
            source table
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        sync_tables: 'bool', optional
            In the case where object files are loaded in, determines whether an
            initial sync is performed between the object and source tables. If
            not performed, dynamic information like the number of observations
            may be out of date until a sync is performed internally.
        additional_cols: 'bool', optional
            Boolean to indicate whether to carry in columns beyond the
            critical columns, true will, while false will only load the columns
            containing the critical quantities (id,time,flux,err,band)
        npartitions: `int`, optional
            If specified, attempts to repartition the ensemble to the specified
            number of partitions
        partition_size: `int`, optional
            If specified, attempts to repartition the ensemble to partitions
            of size `partition_size`.
        sorted: bool, optional
            If the index column is already sorted in increasing order.
            Defaults to False
        sort: `bool`, optional
            If True, sorts the DataFrame by the id column. Otherwise set the
            index on the individual existing partitions. Defaults to False.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with parquet data loaded
        """

        # load column mappings
        self._load_column_mapper(column_mapper, **kwargs)

        # Handle additional columns
        if additional_cols:
            columns = None  # None will prompt read_parquet to read in all cols
        else:
            columns = [self._time_col, self._flux_col, self._err_col, self._band_col]

        # Read in the source parquet file(s)
        # Index is set False so that we can set it with a future set_index call
        # This has the advantage of letting Dask set partition boundaries based
        # on the divisions between the sources of different objects.
        source = SourceFrame.from_parquet(source_file, index=False, columns=columns, ensemble=self)

        object = None
        if object_file:
            # Read in the object file(s)
            # Index is False so that we can set it with a future set_index call
            # More meaningful for source than object but parity seems good here
            object = ObjectFrame.from_parquet(object_file, index=False, ensemble=self)
        return self.from_dask_dataframe(
            source_frame=source,
            object_frame=object,
            column_mapper=column_mapper,
            sync_tables=sync_tables,
            npartitions=npartitions,
            partition_size=partition_size,
            sorted=sorted,
            sort=sort,
            **kwargs,
        )

    def from_dataset(self, dataset, **kwargs):
        """Load the ensemble from a TAPE dataset.

        Parameters
        ----------
        dataset: 'str'
            The name of the dataset to import

        Returns
        -------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with the dataset loaded
        """

        req = requests.get(
            "https://github.com/lincc-frameworks/tape_benchmarking/blob/main/data/datasets.json?raw=True"
        )
        datasets_file = req.json()
        dataset_info = datasets_file[dataset]

        # Make column map from dataset
        dataset_map = dataset_info["column_map"]
        col_map = ColumnMapper(
            id_col=dataset_map["id"],
            time_col=dataset_map["time"],
            flux_col=dataset_map["flux"],
            err_col=dataset_map["error"],
            band_col=dataset_map["band"],
        )

        return self.from_parquet(
            source_file=dataset_info["source_file"],
            object_file=dataset_info["object_file"],
            column_mapper=col_map,
            **kwargs,
        )

    def available_datasets(self):
        """Retrieve descriptions of available TAPE datasets.

        Returns
        -------
        `dict`
            A dictionary of datasets with description information.
        """

        req = requests.get(
            "https://github.com/lincc-frameworks/tape_benchmarking/blob/main/data/datasets.json?raw=True"
        )
        datasets_file = req.json()

        return {key: datasets_file[key]["description"] for key in datasets_file.keys()}

    def from_source_dict(
        self, source_dict, column_mapper=None, npartitions=1, sorted=False, sort=False, **kwargs
    ):
        """Load the sources into an ensemble from a dictionary.

        Parameters
        ----------
        source_dict: 'dict'
            The dictionary containing the source information.
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        npartitions: `int`, optional
            If specified, attempts to repartition the ensemble to the specified
            number of partitions
        sorted: bool, optional
            If the index column is already sorted in increasing order.
            Defaults to False
        sort: `bool`, optional
            If True, sorts the DataFrame by the id column. Otherwise set the
            index on the individual existing partitions. Defaults to False.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with dictionary data loaded
        """

        # Load the source data into a dataframe.
        source_frame = SourceFrame.from_dict(source_dict, npartitions=npartitions)

        return self.from_dask_dataframe(
            source_frame,
            object_frame=None,
            column_mapper=column_mapper,
            sync_tables=True,
            npartitions=npartitions,
            sorted=sorted,
            sort=sort,
            **kwargs,
        )

    def convert_flux_to_mag(self, zero_point, zp_form="mag", out_col_name=None, flux_col=None, err_col=None):
        """Converts a flux column into a magnitude column.

        Parameters
        ----------
        zero_point: 'str' or 'float'
            The name of the ensemble column containing the zero point
            information for column transformation. Alternatively, a single
            float number to apply for all fluxes.
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
        flux_col: 'str', optional
            The name of the ensemble flux column to convert into magnitudes.
            Uses the Ensemble mapped flux column if not specified.
        err_col: 'str', optional
            The name of the ensemble column containing the errors to propagate.
            Errors are propagated using the following approximation:
            Err= (2.5/log(10))*(flux_error/flux), which holds mainly when the
            error in flux is much smaller than the flux. Uses the Ensemble
            mapped error column if not specified.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with a new magnitude (and error) column.

        """

        # Assign Ensemble cols if not provided
        if flux_col is None:
            flux_col = self._flux_col
        if err_col is None:
            err_col = self._err_col

        if out_col_name is None:
            out_col_name = flux_col + "_mag"

        if zp_form == "flux":  # mag = -2.5*np.log10(flux/zp)
            if isinstance(zero_point, str):
                self.update_frame(
                    self.source.assign(
                        **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col] / x[zero_point])}
                    )
                )
            else:
                self.update_frame(
                    self.source.assign(**{out_col_name: lambda x: -2.5 * np.log10(x[flux_col] / zero_point)})
                )

        elif zp_form == "magnitude" or zp_form == "mag":  # mag = -2.5*np.log10(flux) + zp
            if isinstance(zero_point, str):
                self.update_frame(
                    self.source.assign(
                        **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col]) + x[zero_point]}
                    )
                )
            else:
                self.update_frame(
                    self.source.assign(**{out_col_name: lambda x: -2.5 * np.log10(x[flux_col]) + zero_point})
                )
        else:
            raise ValueError(f"{zp_form} is not a valid zero_point format.")

        # Calculate Errors
        if err_col is not None:
            self.update_frame(
                self.source.assign(
                    **{out_col_name + "_err": lambda x: (2.5 / np.log(10)) * (x[err_col] / x[flux_col])}
                )
            )

        return self

    def _generate_object_table(self):
        """Generate an empty object table from the source table."""
        res = self.source.map_partitions(lambda x: TapeObjectFrame(index=x.index.unique()))

        return res

    def _lazy_sync_tables_from_frame(self, frame):
        """Call the sync operation for the frame only if the
        table being modified (`frame`) needs to be synced.
        Does nothing in the case that only the table to be modified
        is dirty or if it is not the object or source frame for this
        `Ensemble`.

        Parameters
        ----------
        frame: `tape.ensemble_frame.EnsembleFrame`
            The frame being modified. Only an `ObjectFrame` or
            `SourceFrame tracked by this `Ensemble` may trigger
            a sync.
        """
        if frame is self.object or frame is self.source:
            # See if we should sync the Object or Source tables.
            self._lazy_sync_tables(frame.label)
        return self

    def _lazy_sync_tables(self, table="object"):
        """Call the sync operation for the table only if the
        the table being modified (`table`) needs to be synced.
        Does nothing in the case that only the table to be modified
        is dirty.

        Parameters
        ----------
        table: `str`, optional
            The table being modified. Should be one of "object",
            "source", or "all"
        """
        if table == "object" and self.source.is_dirty():  # object table should be updated
            self._sync_tables()
        elif table == "source" and self.object.is_dirty():  # source table should be updated
            self._sync_tables()
        elif table == "all" and (self.source.is_dirty() or self.object.is_dirty()):
            self._sync_tables()
        return self

    def _sync_tables(self):
        """Sync operation to align both tables.

        Filtered objects are always removed from the source. But filtered
        sources may be kept in the object table is the Ensemble's
        keep_empty_objects attribute is set to True.
        """

        if self.object.is_dirty():
            # Sync Object to Source; remove any missing objects from source

            if self.object.known_divisions and self.source.known_divisions:
                # Lazily Create an empty object table (just index) for joining
                empty_obj = self.object.map_partitions(lambda x: TapeObjectFrame(index=x.index))
                if type(empty_obj) != type(self.object):
                    raise ValueError("Bad type for empty_obj: " + str(type(empty_obj)))

                # Join source onto the empty object table to align
                self.update_frame(self.source.join(empty_obj, how="inner"))
            else:
                warnings.warn("Divisions are not known, syncing using a non-lazy method.")
                obj_idx = list(self.object.index.compute())
                self.update_frame(self.source.map_partitions(lambda x: x[x.index.isin(obj_idx)]))
                self.update_frame(self.source.persist())  # persist the source frame

            # Drop Temporary Source Columns on Sync
            if len(self._source_temp):
                self.update_frame(self.source.drop(columns=self._source_temp))
                print(f"Temporary columns dropped from Source Table: {self._source_temp}")
                self._source_temp = []

        if self.source.is_dirty():  # not elif
            if not self.keep_empty_objects:
                if self.object.known_divisions and self.source.known_divisions:
                    # Lazily Create an empty source table (just unique indexes) for joining
                    empty_src = self.source.map_partitions(lambda x: TapeSourceFrame(index=x.index.unique()))
                    if type(empty_src) != type(self.source):
                        raise ValueError("Bad type for empty_src: " + str(type(empty_src)))

                    # Join object onto the empty unique source table to align
                    self.update_frame(self.object.join(empty_src, how="inner"))
                else:
                    warnings.warn("Divisions are not known, syncing using a non-lazy method.")
                    # Sync Source to Object; remove any objects that do not have sources
                    sor_idx = list(self.source.index.unique().compute())
                    self.update_frame(self.object.map_partitions(lambda x: x[x.index.isin(sor_idx)]))
                    self.update_frame(self.object.persist())  # persist the object frame

            # Drop Temporary Object Columns on Sync
            if len(self._object_temp):
                self.update_frame(self.object.drop(columns=self._object_temp))
                print(f"Temporary columns dropped from Object Table: {self._object_temp}")
                self._object_temp = []

        # Now synced and clean
        self.source.set_dirty(False)
        self.object.set_dirty(False)
        return self

    def select_random_timeseries(self, seed=None):
        """Selects a random lightcurve from a random partition of the Ensemble.

        Parameters
        ----------
        seed: int, or None
            Sets a seed to return the same object id on successive runs. `None`
            by default, in which case a seed is not set for the operation.

        Returns
        -------
        ts: `TimeSeries`
            Timeseries for a single object

        Note
        ----
        This is not uniformly sampled. As a random partition is chosen first to
        avoid a search in full index space, and partitions may vary in the
        number of objects they contain. In other words, objects in smaller
        partitions will have a higher probability of being chosen than objects
        in larger partitions.

        """

        rng = np.random.default_rng(seed)

        # We will select one partition at random to select an object from
        partitions = np.array(range(self.object.npartitions))
        rng.shuffle(partitions)  # shuffle for empty checking

        object_selected = False
        i = 0

        # Scan through the shuffled partition list until a partition with data is found
        while not object_selected:
            partition_index = self.object.partitions[partitions[i]].index
            # Check for empty partitions
            if len(partition_index) > 0:
                lcid = rng.choice(partition_index.values)  # randomly select lightcurve
                print(f"Selected Object {lcid} from Partition {partitions[i]}")
                object_selected = True
            else:
                i += 1
                if i >= len(partitions):
                    raise IndexError("Found no object IDs in the Object Table.")

        return self.to_timeseries(lcid)

    def to_timeseries(
        self,
        target,
        id_col=None,
        time_col=None,
        flux_col=None,
        err_col=None,
        band_col=None,
    ):
        """Construct a timeseries object from one target object_id, assumes
        that the result is a collection of lightcurves (output from query_ids)

        Parameters
        ----------
        target: `int`
            Id of a source to be extracted
        id_col: 'str', optional
            Identifies which column contains the Object IDs
        time_col: 'str', optional
            Identifies which column contains the time information
        flux_col: 'str', optional
            Identifies which column contains the flux/magnitude information
        err_col: 'str', optional
            Identifies which column contains the error information
        band_col: 'str', optional
            Identifies which column contains the band information

        Returns
        ----------
        ts: `TimeSeries`
            Timeseries for a single object

        Note
        ----
        All _col parameters when not specified will use the appropriate columns
        determined on data ingest as critical columns.
        """

        # Without a specified column, use defaults
        if id_col is None:
            id_col = self._id_col
        if time_col is None:
            time_col = self._time_col
        if flux_col is None:
            flux_col = self._flux_col
        if err_col is None:
            err_col = self._err_col
        if band_col is None:
            band_col = self._band_col

        df = self.source.loc[target].compute()
        ts = TimeSeries().from_dataframe(
            data=df,
            object_id=target,
            time_label=time_col,
            flux_label=flux_col,
            err_label=err_col,
            band_label=band_col,
        )
        return ts

    def _build_index(self, obj_id, band):
        """Build pandas multiindex from object_ids and bands

        Parameters
        ----------
        obj_id : `np.array` or `list`
            A list of object id for each row in the data.
        band : `np.array` or `list`
            A list of the band for each row in the data.

        Returns
        -------
        index : `pd.MultiIndex`
        """
        count_dict = {}
        idx = []
        for o, b in zip(obj_id, band):
            count = count_dict.get((o, b), 0)
            idx.append(count)

            # Increment count for obs_id + band or insert 1 there wasn't an ongoing count.
            count_dict[(o, b)] = count + 1
        tuples = zip(obj_id, band, idx)
        index = pd.MultiIndex.from_tuples(tuples, names=["object_id", "band", "index"])
        return index

    def sf2(self, sf_method="basic", argument_container=None, use_map=True):
        """Wrapper interface for calling structurefunction2 on the ensemble

        Parameters
        ----------
        sf_method : 'str'
            The structure function calculation method to be used, by default "basic".
        argument_container : StructureFunctionArgumentContainer, optional
            Container object for additional configuration options, by default None.
        use_map : `boolean`
            Determines whether `dask.dataframe.DataFrame.map_partitions` is
            used (True). Using map_partitions is generally more efficient, but
            requires the data from each lightcurve is housed in a single
            partition. If False, a groupby will be performed instead.

        Returns
        ----------
        result : `pandas.DataFrame`
            Structure function squared for each of input bands.

        Note
        ----------
        In case that no value for `band_to_calc` is passed, the function is
        executed on all available bands in `band`.
        """

        # Create a default argument container of the correct type if one was not
        # provided. This is only necessary here because we need to know about
        # `combine` in the next conditional.
        if argument_container is None:
            argument_container_type = SF_METHODS[sf_method].expected_argument_container()
            argument_container = argument_container_type()

        if argument_container.combine:
            result = calc_sf2(
                self.source[self._time_col],
                self.source[self._flux_col],
                self.source[self._err_col],
                self.source[self._band_col],
                self.source.index,
                argument_container=argument_container,
            )

        else:
            result = self.batch(calc_sf2, use_map=use_map, argument_container=argument_container)

        # Inherit divisions information if known
        if self.source.known_divisions and self.object.known_divisions:
            result.divisions = self.source.divisions

        return result

    def _translate_meta(self, meta):
        """Translates Dask-style meta into a TapeFrame or TapeSeries object.

        Parameters
        ----------
        meta : `dict`, `tuple`, `list`, `pd.Series`, `pd.DataFrame`, `pd.Index`, `dtype`, `scalar`

        Returns
        ----------
        result : `ensemble.TapeFrame` or `ensemble.TapeSeries`
            The appropriate meta for Dask producing an `tape.ensemble_frame.EnsembleFrame` or
            `Ensemble.EnsembleSeries` respectively
        """
        if isinstance(meta, TapeFrame) or isinstance(meta, TapeSeries):
            return meta

        # If the meta is not a DataFrame or Series, have Dask attempt translate the meta into an
        # appropriate Pandas object.
        meta_object = meta
        if not (isinstance(meta_object, pd.DataFrame) or isinstance(meta_object, pd.Series)):
            meta_object = dd.backends.make_meta_object(meta_object)

        # Convert meta_object into the appropriate TAPE extension.
        if isinstance(meta_object, pd.DataFrame):
            return TapeFrame(meta_object)
        elif isinstance(meta_object, pd.Series):
            return TapeSeries(meta_object)
        else:
            raise ValueError("Unsupported Meta: " + str(meta) + "\nTry a Pandas DataFrame or Series instead.")
