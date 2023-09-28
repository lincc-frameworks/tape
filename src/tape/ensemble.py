import glob
import os
import warnings
import requests

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client

from .analysis.base import AnalysisFunction
from .analysis.feature_extractor import BaseLightCurveFeature, FeatureExtractor
from .analysis.structure_function import SF_METHODS
from .analysis.structurefunction2 import calc_sf2
from .ensemble_frame import ObjectFrame, SourceFrame, TapeObjectFrame
from .timeseries import TimeSeries
from .utils import ColumnMapper

# TODO import from EnsembleFrame...?
SOURCE_FRAME_LABEL = "source"
OBJECT_FRAME_LABEL = "object"

class Ensemble:
    """Ensemble object is a collection of light curve ids"""

    def __init__(self, client=True, **kwargs):
        """Constructor of an Ensemble instance.

        Parameters
        ----------
        client: `dask.distributed.client` or `bool`, optional
            Accepts an existing `dask.distributed.Client`, or creates one if
            `client=True`, passing any additional kwargs to a
             dask.distributed.Client constructor call. If `client=False`, the
             Ensemble is created without a distributed client.

        """
        self.result = None  # holds the latest query

        self._source = None  # Source Table
        self._object = None  # Object Table

        self.frames = {} # Frames managed by this Ensemble, keyed by label

        # TODO(wbeebe@uw.edu) Replace self._source and self._object with these 
        self.source = None # Source Table EnsembleFrame
        self.object = None # Object Table EnsembleFrame

        # Default to removing empty objects.
        self.keep_empty_objects = kwargs.get("keep_empty_objects", False)

        # Initialize critical column quantities
        # Source
        self._id_col = None
        self._time_col = None
        self._flux_col = None
        self._err_col = None
        self._band_col = None
        self._provenance_col = None

        # Object, _id_col is shared
        self._nobs_tot_col = None
        self._nobs_band_cols = []

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
        frame: `tape.ensemble.EnsembleFrame`
            The frame object for the Ensemble to track.
        label: `str`
        |   The label for the Ensemble to use to track the frame.    

        Returns
        -------
        self: `Ensemble`

        Raises
        ------
        ValueError if the label is "source", "object", or already tracked by the Ensemble.
        """
        if label == SOURCE_FRAME_LABEL or label == OBJECT_FRAME_LABEL:
            raise ValueError(
                f"Unable to add frame with reserved label " f"'{label}'"
                )
        if label in self.frames:
            raise ValueError(
                f"Unable to add frame: a frame with label " f"'{label}'" f"is in the Ensemble."
                )
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
        self: `Ensemble`

        Raises
        ------
        ValueError if the `frame.label` is unpopulated, "source", or "object".
        """
        if frame.label is None:
            raise ValueError(
                f"Unable to update frame with no populated `EnsembleFrame.label`."
                )
        if isinstance(frame, SourceFrame) or isinstance(frame, ObjectFrame):
            expected_label = SOURCE_FRAME_LABEL if isinstance(frame, SourceFrame) else OBJECT_FRAME_LABEL
            if frame.label != expected_label:
                raise ValueError(f"Unable to update frame with reserved label " f"'{frame.label}'"
                )
            if isinstance(frame, SourceFrame):
                self._source = frame
                self.source = frame
            elif isinstance(frame, ObjectFrame):
                self._object = frame
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
        |   The label of the frame to be dropped by the Ensemble.

        Returns
        -------
        self: `Ensemble`

        Raises
        ------
        ValueError if the label is "source", or "object".
        KeyError if the label is not tracked by the Ensemble.
        """
        if label == SOURCE_FRAME_LABEL or label == OBJECT_FRAME_LABEL:
            raise ValueError(
                f"Unable to drop frame with reserved label " f"'{label}'"
                )
        if label not in self.frames:
            raise KeyError(
                f"Unable to drop frame: no frame with label " f"'{label}'" f"is in the Ensemble."
                )
        del self.frames[label]
        return self

    def select_frame(self, label):
        """Selects and returns frame tracked by the Ensemble.

        Parameters
        ----------
        label: `str`
        |   The label of a frame tracked by the Ensemble to be selected.

        Returns
        -------
        result: `tape.ensemeble.EnsembleFrame`

        Raises
        ------
        KeyError if the label is not tracked by the Ensemble.
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
        KeyError if a label in labels is not tracked by the Ensemble.
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

    def insert_sources(
        self,
        obj_ids,
        bands,
        timestamps,
        fluxes,
        flux_errs=None,
        provenance_label="custom",
        force_repartition=False,
        **kwargs,
    ):
        """Manually insert sources into the ensemble.

        Requires, at a minimum, the object’s ID and the band, timestamp,
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
        provenance_label: `str`, optional
            A label that denotes the provenance of the new observations.
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

        # Construct provenance array
        provenance = [provenance_label] * len(obj_ids)

        # Create a dictionary with the new information.
        rows = {
            self._id_col: obj_ids,
            self._band_col: bands,
            self._time_col: timestamps,
            self._flux_col: fluxes,
            self._provenance_col: provenance,
        }
        if flux_errs is not None:
            rows[self._err_col] = flux_errs

        # Add any other supplied columns to the dictionary.
        for key, value in kwargs.items():
            if key in self._data.columns:
                rows[key] = value

        # Create the new row and set the paritioning to match the original dataframe.
        df2 = dd.DataFrame.from_dict(rows, npartitions=1)
        df2 = df2.set_index(self._id_col, drop=True)

        # Save the divisions and number of partitions.
        prev_div = self._source.divisions
        prev_num = self._source.npartitions

        # Append the new rows to the correct divisions.
        self.update_frame(dd.concat([self._source, df2], axis=0, interleave_partitions=True))
        self._source.set_dirty(True)

        # Do the repartitioning if requested. If the divisions were set, reuse them.
        # Otherwise, use the same number of partitions.
        if force_repartition:
            if all(prev_div):
                self.update_frame(self._source.repartition(divisions=prev_div))
            elif self._source.npartitions != prev_num:
                self.update_frame(self._source.repartition(npartitions=prev_num))

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
        self._object.info(verbose=verbose, memory_usage=memory_usage, **kwargs)
        print("Source Table")
        self._source.info(verbose=verbose, memory_usage=memory_usage, **kwargs)

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
        A single pandas data frame for the specified table or a tuple of (object, source)
        data frames.
        """
        if table:
            self._lazy_sync_tables(table)
            if table == "object":
                return self._object.compute(**kwargs)
            elif table == "source":
                return self._source.compute(**kwargs)
        else:
            self._lazy_sync_tables(table="all")
            return (self._object.compute(**kwargs), self._source.compute(**kwargs))

    def persist(self, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.persist()

        The compute operation performs the computations that had been lazily allocated,
        but does not bring the results into memory or return them. This is useful
        for preventing a Dask task graph from growing too large by performing part
        of the computation.
        """
        self._lazy_sync_tables("all")
        self.update_frame(self._object.persist(**kwargs))
        self.update_frame(self._source.persist(**kwargs))

    def columns(self, table="object"):
        """Retrieve columns from dask dataframe"""
        if table == "object":
            return self._object.columns
        elif table == "source":
            return self._source.columns
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def head(self, table="object", n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.head()"""
        self._lazy_sync_tables(table)

        if table == "object":
            return self._object.head(n=n, **kwargs)
        elif table == "source":
            return self._source.head(n=n, **kwargs)
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def tail(self, table="object", n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.tail()"""
        self._lazy_sync_tables(table)

        if table == "object":
            return self._object.tail(n=n, **kwargs)
        elif table == "source":
            return self._source.tail(n=n, **kwargs)
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
            self.update_frame(self._object.dropna(**kwargs))
            self._object.set_dirty(True)  # This operation modifies the object table
        elif table == "source":
            self.update_frame(self._source.dropna(**kwargs))
            self._source.set_dirty(True)  # This operation modifies the source table
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
            cols_to_drop = [col for col in self._object.columns if col not in columns]
            self.update_frame(self._object.drop(cols_to_drop, axis=1))
            self._object.set_dirty(True)
        elif table == "source":
            cols_to_drop = [col for col in self._source.columns if col not in columns]
            self.update_frame(self._source.drop(cols_to_drop, axis=1))
            self._source.set_dirty(True)
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
        # Keep sources with flux above 100.0:
        ens.query("flux > 100", table="source")

        # Keep sources in the green band:
        ens.query("band_col_name == 'g'", table="source")

        # Filtering on the flux column without knowing its name:
        ens.query(f"{ens._flux_col} > 100", table="source")
        """
        self._lazy_sync_tables(table)
        if table == "object":
            self.update_frame(self._object.query(expr))
            self._object.set_dirty(True)
        elif table == "source":
            self.update_frame(self._source.query(expr))
            self._source.set_dirty(True)
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
            self.update_frame(self._object[keep_series])
            self._object.set_dirty(True)
        elif table == "source":
            self.update_frame(self._source[keep_series])
            self._source.set_dirty(True)
        return self

    def assign(self, table="object", **kwargs):
        """Wrapper for dask.dataframe.DataFrame.assign()

        Parameters
        ----------
        table: `str`, optional
            A string indicating which table to filter.
            Should be one of "object" or "source".
        kwargs: dict of {str: callable or Series}
            Each argument is the name of a new column to add and its value specifies
            how to fill it. A callable is called for each row and a series is copied in.

        Returns
        -------
        self: `tape.ensemble.Ensemble`
            The ensemble object.

        Examples
        --------
        # Direct assignment of my_series to a column named "new_column".
        ens.assign(table="object", new_column=my_series)

        # Subtract the value in "err" from the value in "flux".
        ens.assign(table="source", lower_bnd=lambda x: x["flux"] - 2.0 * x["err"])
        """
        self._lazy_sync_tables(table)

        if table == "object":
            self.update_frame(self._object.assign(**kwargs))
            self._object.set_dirty(True)
        elif table == "source":
            self.update_frame(self._source.assign(**kwargs))
            self._source.set_dirty(True)
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")
        return self

    def coalesce(self, input_cols, output_col, table="object", drop_inputs=False):
        """Combines multiple input columns into a single output column, with
        values equal to the first non-nan value encountered in the input cols.

        Parameters
        ----------
        input_cols: `list`
            The list of column names to coalesce into a single column.
        output_col: `str`, optional
            The name of the coalesced output column.
        table: `str`, optional
            "source" or "object", the table in which the input columns are
            located.
        drop_inputs: `bool`, optional
            Determines whether the input columns are dropped or preserved. If
            a mapped column is an input and dropped, the output column is
            automatically assigned to replace that column mapping internally.

        Returns
        -------
        ensemble: `tape.ensemble.Ensemble`
            An ensemble object.

        """
        # we shouldn't need to sync for this
        if table == "object":
            table_ddf = self._object
        elif table == "source":
            table_ddf = self._source
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

        # Create a subset dataframe with the coalesced columns
        # Drop index for dask series operations - unfortunate
        coal_ddf = table_ddf[input_cols].reset_index()

        # Coalesce each column iteratively
        i = 0
        coalesce_col = coal_ddf[input_cols[0]]
        while i < len(input_cols) - 1:
            coalesce_col = coalesce_col.combine_first(coal_ddf[input_cols[i + 1]])
            i += 1
        print("am I using this code")
        # Assign the new column to the subset df, and reintroduce index
        coal_ddf = coal_ddf.assign(**{output_col: coalesce_col}).set_index(self._id_col)

        # assign the result to the desired column name
        table_ddf = table_ddf.assign(**{output_col: coal_ddf[output_col]})

        # Drop the input columns if wanted
        if drop_inputs:
            # First check to see if any dropped columns were critical columns
            current_map = self.make_column_map().map
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

                new_colmap = self.make_column_map()
                new_colmap.map = new_map

                # Update the mapping
                self.update_column_mapping(new_colmap)

            table_ddf = table_ddf.drop(columns=input_cols)

        if table == "object":
            self.update_frame(table_ddf)
        elif table == "source":
            self.update_frame(table_ddf)

        return self

    def prune(self, threshold=50, col_name=None):
        """remove objects with less observations than a given threshold

        Parameters
        ----------
        threshold: `int`, optional
            The minimum number of observations needed to retain an object.
            Default is 50.
        col_name: `str`, optional
            The name of the column to assess the threshold

        Returns
        -------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with pruned rows removed
        """
        if not col_name:
            col_name = self._nobs_tot_col

        # Sync Required if source is dirty
        self._lazy_sync_tables(table="object")

        # Mask on object table
        mask = self._object[col_name] >= threshold
        self.update_frame(self._object[mask])

        self._object.set_dirty(True)  # Object Table is now dirty

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
        hours = self._source[self._time_col].apply(
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
        if tmp_time_col in self._source.columns:
            raise KeyError(f"Column '{tmp_time_col}' already exists in source table.")
        self._source[tmp_time_col] = self._source[self._time_col].apply(
            lambda x: np.floor((x + offset) / time_window) * time_window, meta=pd.Series(dtype=float)
        )

        # Set up the aggregation functions for the time and flux columns.
        aggr_funs = {self._time_col: "mean", self._flux_col: "mean"}

        # If the source table has errors then add an aggregation function for it.
        if self._err_col in self._source.columns:
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
            if self._bin_count_col not in self._source.columns:
                self._source[self._bin_count_col] = self._source[self._time_col].apply(
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
        self.update_frame(self._source.groupby([self._id_col, self._band_col, tmp_time_col]).aggregate(aggr_funs))

        # Fix the indices and remove the temporary column.
        self.update_frame(self._source.reset_index().set_index(self._id_col).drop(tmp_time_col, axis=1))

        # Mark the source table as dirty.
        self._source.set_dirty(True)
        return self

    def batch(self, func, *args, meta=None, use_map=True, compute=True, on=None, **kwargs):
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
        use_map : `boolean`
            Determines whether `dask.dataframe.DataFrame.map_partitions` is
            used (True). Using map_partitions is generally more efficient, but
            requires the data from each lightcurve is housed in a single
            partition. If False, a groupby will be performed instead.
        compute: `boolean`
            Determines whether to compute the result immediately or hold for a
            later compute call.
        on: 'str' or 'list'
            Designates which column(s) to groupby. Columns may be from the
            source or object tables. For TAPE and `light-curve` functions
            this is populated automatically.
        **kwargs:
            Additional optional parameters passed for the selected function

        Returns
        -------
        result: `Dask.Series`
            Series of function results

        Examples
        --------
        Run a TAPE function on the ensemble:
        ```
        from tape.analysis.stetsonj import calc_stetson_J
        ens = Ensemble().from_dataset('rrlyr82')
        ensemble.batch(calc_stetson_J, band_to_calc='i')
        ```

        Run a light-curve function on the ensemble:
        ```
        from light_curve import EtaE
        ens.batch(EtaE(), band_to_calc='g')
        ```

        Run a custom function on the ensemble:
        ```
        def s2n_inter_quartile_range(flux, err):
             first, third = np.quantile(flux / err, [0.25, 0.75])
             return third - first

        ens.batch(s2n_inter_quartile_range, ens._flux_col, ens._err_col)
        ```
        Or even a numpy built-in function:
        ```
        amplitudes = ens.batch(np.ptp, ens._flux_col)
        ```
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

        if on is None:
            on = self._id_col  # Default grouping is by id_col
        if isinstance(on, str):
            on = [on]  # Convert to list if only one column is passed

        # Handle object columns to group on
        source_cols = list(self._source.columns)
        object_cols = list(self._object.columns)
        object_group_cols = [col for col in on if (col in object_cols) and (col not in source_cols)]

        if len(object_group_cols) > 0:
            object_col_dd = self._object[object_group_cols]
            source_to_batch = self._source.merge(object_col_dd, how="left")
        else:
            source_to_batch = self._source  # Can directly use the source table

        id_col = self._id_col  # pre-compute needed for dask in lambda function

        if use_map:  # use map_partitions
            id_col = self._id_col  # need to grab this before mapping
            batch = source_to_batch.map_partitions(
                lambda x: x.groupby(on, group_keys=False).apply(
                    lambda y: func(
                        *[y[arg].to_numpy() if arg != id_col else y.index.to_numpy() for arg in args],
                        **kwargs,
                    )
                ),
                meta=meta,
            )
        else:  # use groupby
            batch = source_to_batch.groupby(on, group_keys=False).apply(
                lambda x: func(
                    *[x[arg].to_numpy() if arg != id_col else x.index.to_numpy() for arg in args], **kwargs
                ),
                meta=meta,
            )

        if compute:
            return batch.compute()
        else:
            return batch

    def from_hipscat(self, dir, source_subdir="source", object_subdir="object", column_mapper=None, **kwargs):
        """Read in parquet files from a hipscat-formatted directory structure
        Parameters
        ----------
        dir: 'str'
            Path to the directory structure
        source_subdir: 'str'
            Path to the subdirectory which contains source files
        object_subdir: 'str'
            Path to the subdirectory which contains object files, if None then
            files will only be read from the source_subdir
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        **kwargs:
            keyword arguments passed along to
            `tape.ensemble.Ensemble.from_parquet`

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with parquet data loaded
        """

        source_path = os.path.join(dir, source_subdir)
        source_files = glob.glob(os.path.join(source_path, "**", "*.parquet"), recursive=True)

        if object_subdir is not None:
            object_path = os.path.join(dir, object_subdir)
            object_files = glob.glob(os.path.join(object_path, "**", "*.parquet"), recursive=True)
        else:
            object_files = None

        return self.from_parquet(
            source_files,
            object_files,
            column_mapper=column_mapper,
            **kwargs,
        )

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
            provenance_col=self._provenance_col,
            nobs_total_col=self._nobs_tot_col,
            nobs_band_cols=self._nobs_band_cols,
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

            # Assign optional columns if provided
            if column_mapper.map["provenance_col"] is not None:
                self._provenance_col = column_mapper.map["provenance_col"]
            if column_mapper.map["nobs_total_col"] is not None:
                self._nobs_tot_col = column_mapper.map["nobs_total_col"]
            if column_mapper.map["nobs_band_cols"] is not None:
                self._nobs_band_cols = column_mapper.map["nobs_band_cols"]

        else:
            raise ValueError(f"Missing required column mapping information: {needed}")

        return self

    def from_parquet(
        self,
        source_file,
        object_file=None,
        column_mapper=None,
        provenance_label="survey_1",
        sync_tables=True,
        additional_cols=True,
        npartitions=None,
        partition_size=None,
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
        provenance_label: 'str', optional
            Determines the label to use if a provenance column is generated
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
            if self._provenance_col is not None:
                columns.append(self._provenance_col)
            if self._nobs_tot_col is not None:
                columns.append(self._nobs_tot_col)
            if self._nobs_band_cols is not None:
                for col in self._nobs_band_cols:
                    columns.append(col)

        # Read in the source parquet file(s)
        self.update_frame(SourceFrame.from_parquet(
            source_file, index=self._id_col, columns=columns, ensemble=self,
        ))

        if object_file:  # read from parquet files
            # Read in the object file(s)
            self.update_frame(ObjectFrame.from_parquet(object_file, index=self._id_col, ensemble=self))
            if self._nobs_band_cols is None:
                # sets empty nobs cols in object
                unq_filters = np.unique(self._source[self._band_col])
                self._nobs_band_cols = [f"nobs_{filt}" for filt in unq_filters]
                for col in self._nobs_band_cols:
                    self._object[col] = np.nan

            # Handle nobs_total column
            if self._nobs_tot_col is None:
                self._object["nobs_total"] = np.nan
                self._nobs_tot_col = "nobs_total"

            # Optionally sync the tables, recalculates nobs columns
            if sync_tables:
                self._source.set_dirty(True)
                self._object.set_dirty(True)
                self._sync_tables()

        else:  # generate object table from source
            self.update_frame(self._generate_object_table())
            self._nobs_bands = [col for col in list(self._object.columns) if col != self._nobs_tot_col]

        # Generate a provenance column if not provided
        if self._provenance_col is None:
            self._source["provenance"] = self._source.apply(
                lambda x: provenance_label, axis=1, meta=pd.Series(name="provenance", dtype=str)
            )
            self._provenance_col = "provenance"

        if npartitions and npartitions > 1:
            self.update_frame(self._source.repartition(npartitions=npartitions))
        elif partition_size:
            self.update_frame(self._source.repartition(partition_size=partition_size))

        return self
    
    def objsor_from_parquet(
        self,
        source_file,
        object_file,
        column_mapper=None,
        provenance_label="survey_1",
        sync_tables=True,
        additional_cols=True,
        npartitions=None,
        partition_size=None,
        **kwargs,
    ):
        """Read in parquet file(s) for the object and source tables into an Ensemble object.

        Parameters
        ----------
        source_file: 'str'
            Path to a parquet file, or multiple parquet files that contain
            source information to be read into the ensemble
        object_file: 'str'
            Path to a parquet file, or multiple parquet files that contain
            object information.
        column_mapper: 'ColumnMapper' object
            If provided, the ColumnMapper is used to populate relevant column
            information mapped from the input dataset.
        provenance_label: 'str', optional
            Determines the label to use if a provenance column is generated
        sync_tables: 'bool', optional
            In the case where object files are loaded in, determines whether an
            initial sync is performed between the object and source tables. If
            not performed, dynamic information like the number of observations
            may be out of date until a sync is performed internally.
        additional_cols: 'bool', optional
            Boolean to indicate whether to carry in columns beyond the
            critical columns, True will, while False will only load the columns
            containing the critical quantities (id,time,flux,err,band)
        npartitions: `int`, optional
            If specified, attempts to repartition the ensemble to the specified
            number of partitions
        partition_size: `int`, optional
            If specified, attempts to repartition the ensemble to partitions
            of size `partition_size`, the maximum number of bytes for partition
            as computed by `pandas.Dataframe.memory_usage`.

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with parquet data loaded
        """

        # load column mappings
        self._load_column_mapper(column_mapper, **kwargs)

        # Handle additional columnss
        if additional_cols:
            columns = None  # None will prompt read_parquet to read in all cols
        else:
            columns = [self._time_col, self._flux_col, self._err_col, self._band_col]
            if self._provenance_col is not None:
                columns.append(self._provenance_col)
            if self._nobs_tot_col is not None:
                columns.append(self._nobs_tot_col)
            if self._nobs_band_cols is not None:
                for col in self._nobs_band_cols:
                    columns.append(col)

        # Read in the source parquet file(s)
        self.update_frame(SourceFrame.from_parquet(source_file, index=self._id_col, columns=columns, 
                                               ensemble=self))

        # Read in the object file(s)
        self.update_frame(ObjectFrame.from_parquet(object_file, index=self._id_col, ensemble=self))

        if self._nobs_band_cols is None:
            # sets empty nobs cols in object
            unq_filters = np.unique(self.source[self._band_col])
            self._nobs_band_cols = [f"nobs_{filt}" for filt in unq_filters]
            for col in self._nobs_band_cols:
                self.object[col] = np.nan

        # Handle nobs_total column
        if self._nobs_tot_col is None:
            self.object["nobs_total"] = np.nan
            self._nobs_tot_col = "nobs_total"

        # TODO(wbeebe@uw.edu) Add in table syncing logic as part of milestone 4
            
        # Generate a provenance column if not provided
        if self._provenance_col is None:
            self.source["provenance"] = provenance_label
            self._provenance_col = "provenance"

        if npartitions and npartitions > 1:
            self.update_frame(self.source.repartition(npartitions=npartitions))
        elif partition_size:
            self.update_frame(self.source.repartition(partition_size=partition_size))

        return self

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
            provenance_label=dataset,
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

    def from_source_dict(self, source_dict, column_mapper=None, npartitions=1, **kwargs):
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

        Returns
        ----------
        ensemble: `tape.ensemble.Ensemble`
            The ensemble object with dictionary data loaded
        """
        # load column mappings
        self._load_column_mapper(column_mapper, **kwargs)

        # Load in the source data.
        self.update_frame(SourceFrame.from_dict(source_dict, npartitions=npartitions))
        self.update_frame(self._source.set_index(self._id_col, drop=True))

        # Generate the object table from the source.
        # TODO this is not the object Table oh no....
        self.update_frame(self._generate_object_table())

        # Now synced and clean
        self._source.set_dirty(False)
        self._object.set_dirty(False)
        return self
 
    def _generate_object_table(self):
        """Generate the object table from the source table."""
        counts = self._source.groupby([self._id_col, self._band_col])[self._time_col].aggregate("count")
        res = (
            counts.to_frame()
            .reset_index()
            .categorize(columns=[self._band_col])
            .pivot_table(values=self._time_col, index=self._id_col, columns=self._band_col, aggfunc="sum")
        )

        # Convert the resulting dataframe into an ObjectFrame
        # TODO(wbeebe@uw.edu): Inveestigate if we can correctly infer that `res` is an ObjectFrame instead
        res = ObjectFrame.from_dask_dataframe(res, ensemble=self)

        # If the ensemble's keep_empty_objects attribute is True and there are previous
        # objects, then copy them into the res table with counts of zero.
        if self.keep_empty_objects and self._object is not None:
            prev_partitions = self._object.npartitions

            # Check that there are existing object ids.
            object_inds = self._object.index.unique().values.compute()
            if len(object_inds) > 0:
                # Determine which object IDs are missing from the source table.
                source_inds = self._source.index.unique().values.compute()
                missing_inds = np.setdiff1d(object_inds, source_inds).tolist()

                # Create a dataframe of the missing IDs with zeros for all bands and counts.
                rows = {self._id_col: missing_inds}
                for i in res.columns.values:
                    rows[i] = [0] * len(missing_inds)

                zero_pdf = pd.DataFrame(rows, dtype=int).set_index(self._id_col)
                zero_ddf = dd.from_pandas(zero_pdf, sort=True, npartitions=1)

                # Concatenate the zero dataframe onto the results.
                res = dd.concat([res, zero_ddf], interleave_partitions=True).astype(int)
                res = res.repartition(npartitions=prev_partitions)

        # Rename bands to nobs_[band]
        band_cols = {col: f"nobs_{col}" for col in list(res.columns)}
        res = res.rename(columns=band_cols)

        # Add total nobs by summing across each band.
        if self._nobs_tot_col is None:
            self._nobs_tot_col = "nobs_total"
        res[self._nobs_tot_col] = res.sum(axis=1)

        return res

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
        if table == "object" and self._source.is_dirty():  # object table should be updated
            self._sync_tables()
        elif table == "source" and self._object.is_dirty():  # source table should be updated
            self._sync_tables()
        elif table == "all" and (self._source.is_dirty() or self._object.is_dirty()):
            self._sync_tables()
        return self

    def _sync_tables(self):
        """Sync operation to align both tables.

        Filtered objects are always removed from the source. But filtered
        sources may be kept in the object table is the Ensemble's
        keep_empty_objects attribute is set to True.
        """

        if self._object.is_dirty():
            # Sync Object to Source; remove any missing objects from source
            s_cols = self._source.columns
            self.update_frame(self._source.merge(
                self._object, how="right", on=[self._id_col], suffixes=(None, "_obj")
            ))
            cols_to_drop = [col for col in self._source.columns if col not in s_cols]
            self.update_frame(self._source.drop(cols_to_drop, axis=1))
            self.update_frame(self._source.persist())  # persist source

        if self._source._is_dirty:  # not elif
            # Generate a new object table; updates n_obs, removes missing ids
            new_obj = self._generate_object_table()

            # Join old obj to new obj; pulls in other existing obj columns
            self.update_frame(new_obj.join(self._object, on=self._id_col, how="left", lsuffix="", rsuffix="_old"))
            old_cols = [col for col in list(self._object.columns) if "_old" in col]
            self.update_frame(self._object.drop(old_cols, axis=1))
            self.update_frame(self._object.persist())  # persist object

        # Now synced and clean
        self._source.set_dirty(False)
        self._object.set_dirty(False)
        return self

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

        df = self._source.loc[target].compute()
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

        Notes
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
                self._source[self._time_col],
                self._source[self._flux_col],
                self._source[self._err_col],
                self._source[self._band_col],
                self._source.index,
                argument_container=argument_container,
            )
            return result
        else:
            result = self.batch(calc_sf2, use_map=use_map, argument_container=argument_container)

            return result
