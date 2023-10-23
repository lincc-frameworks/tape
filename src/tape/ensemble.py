import glob
import os
import warnings
import requests
import dask.dataframe as dd
import numpy as np
import pandas as pd

from dask.distributed import Client
from collections import Counter

from .analysis.base import AnalysisFunction
from .analysis.feature_extractor import BaseLightCurveFeature, FeatureExtractor
from .analysis.structure_function import SF_METHODS
from .analysis.structurefunction2 import calc_sf2
from .timeseries import TimeSeries
from .utils import ColumnMapper


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

        self._source_dirty = False  # Source Dirty Flag
        self._object_dirty = False  # Object Dirty Flag

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
        self._provenance_col = None

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
        df2 = df2.set_index(self._id_col, drop=True, sort=True)

        # Save the divisions and number of partitions.
        prev_div = self._source.divisions
        prev_num = self._source.npartitions

        # Append the new rows to the correct divisions.
        self._source = dd.concat([self._source, df2], axis=0, interleave_partitions=True)
        self._source_dirty = True

        # Do the repartitioning if requested. If the divisions were set, reuse them.
        # Otherwise, use the same number of partitions.
        if force_repartition:
            if all(prev_div):
                self._source = self._source.repartition(divisions=prev_div)
            elif self._source.npartitions != prev_num:
                self._source = self._source.repartition(npartitions=prev_num)

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
        """Wrapper for dask.dataframe.DataFrame.info()

        Parameters
        ----------
        verbose: `bool`, optional
            Whether to print the whole summary
        memory_usage: `bool`, optional
            Specifies whether total memory usage of the DataFrame elements
            (including the index) should be displayed.
        Returns
        ----------
        counts: `pandas.series`
            A series of counts by object
        """
        # Sync tables if user wants to retrieve their information
        self._lazy_sync_tables(table="all")

        print("Object Table")
        self._object.info(verbose=verbose, memory_usage=memory_usage, **kwargs)
        print("Source Table")
        self._source.info(verbose=verbose, memory_usage=memory_usage, **kwargs)

    def check_sorted(self, table="object"):
        """Checks to see if an Ensemble Dataframe is sorted on the index.

        Parameters
        ----------
        table: `str`, optional
            The table to check.

        Returns
        -------
        A boolean value indicating whether the index is sorted (True)
        or not (False)
        """
        if table == "object":
            idx = self._object.index
        elif table == "source":
            idx = self._source.index
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")
        return idx.map_partitions(lambda a: np.all(a[:-1] <= a[1:])).compute().all()

    def check_lightcurve_cohesion(ens):
        """Checks to see if lightcurves are split across multiple partitions.

        With partitioned data, and source information represented by rows, it
        is possible that when loading data or manipulating it in some way (most
        likely a repartition) that the sources for a given object will be split
        among multiple partitions. This function will check to see if all
        lightcurves are "cohesive", meaning the sources for that object only
        live in a single partition of the dataset.

        Returns
        -------
        A boolean value indicating whether the sources tied to a given object
        are only found in a single partition (True), or if they are split
        across multiple partitions (False)

        """
        idx = ens._source.index
        counts = idx.map_partitions(lambda a: Counter(a.unique())).compute()

        unq_counter = counts[0]
        for i in range(len(counts) - 1):
            unq_counter += counts[i + 1]
            if any(c >= 2 for c in unq_counter.values()):
                return False
        return True

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
        self._object = self._object.persist(**kwargs)
        self._source = self._source.persist(**kwargs)

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
            self._object = self._object.dropna(**kwargs)
            self._object_dirty = True  # This operation modifies the object table
        elif table == "source":
            self._source = self._source.dropna(**kwargs)
            self._source_dirty = True  # This operation modifies the source table
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
            self._object = self._object.drop(cols_to_drop, axis=1)
            self._object_dirty = True
        elif table == "source":
            cols_to_drop = [col for col in self._source.columns if col not in columns]
            self._source = self._source.drop(cols_to_drop, axis=1)
            self._source_dirty = True
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
            self._object = self._object.query(expr)
            self._object_dirty = True
        elif table == "source":
            self._source = self._source.query(expr)
            self._source_dirty = True
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
            self._object = self._object[keep_series]
            self._object_dirty = True
        elif table == "source":
            self._source = self._source[keep_series]
            self._source_dirty = True
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
        # Direct assignment of my_series to a column named "new_column".
        ens.assign(table="object", new_column=my_series)

        # Subtract the value in "err" from the value in "flux".
        ens.assign(table="source", lower_bnd=lambda x: x["flux"] - 2.0 * x["err"])
        """
        self._lazy_sync_tables(table)

        if table == "object":
            pre_cols = self._object.columns
            self._object = self._object.assign(**kwargs)
            self._object_dirty = True
            post_cols = self._object.columns

            if temporary:
                self._object_temp.extend(col for col in post_cols if col not in pre_cols)

        elif table == "source":
            pre_cols = self._source.columns
            self._source = self._source.assign(**kwargs)
            self._source_dirty = True
            post_cols = self._source.columns

            if temporary:
                self._source_temp.extend(col for col in post_cols if col not in pre_cols)

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

        table_ddf = table_ddf.map_partitions(lambda x: coalesce_partition(x, input_cols, output_col))

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
            self._object = table_ddf
        elif table == "source":
            self._source = table_ddf

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

        if by_band:
            band_counts = (
                self._source.groupby([self._id_col])[self._band_col]  # group by each object
                .value_counts()  # count occurence of each band
                .to_frame()  # convert series to dataframe
                .reset_index()  # break up the multiindex
                .categorize(columns=[self._band_col])  # retype the band labels as categories
                .pivot_table(values=self._band_col, index=self._id_col, columns=self._band_col, aggfunc="sum")
            )  # the pivot_table call makes each band_count a column of the id_col row

            # repartition the result to align with object
            if self._object.known_divisions:
                self._object.divisions = tuple([None for i in range(self._object.npartitions + 1)])
                band_counts = band_counts.repartition(npartitions=self._object.npartitions)
            else:
                band_counts = band_counts.repartition(npartitions=self._object.npartitions)

            # short-hand for calculating nobs_total
            band_counts["total"] = band_counts[list(band_counts.columns)].sum(axis=1)

            bands = band_counts.columns.values
            self._object = self._object.assign(**{label + "_" + band: band_counts[band] for band in bands})

            if temporary:
                self._object_temp.extend(label + "_" + band for band in bands)

        else:
            counts = self._source.groupby([self._id_col])[[self._band_col]].aggregate("count")

            # repartition the result to align with object
            if self._object.known_divisions:
                self._object.divisions = tuple([None for i in range(self._object.npartitions + 1)])
                counts = counts.repartition(npartitions=self._object.npartitions)
            else:
                counts = counts.repartition(npartitions=self._object.npartitions)

            self._object = self._object.assign(**{label + "_total": counts[self._band_col]})

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
        mask = self._object[col_name] >= threshold
        self._object = self._object[mask]

        self._object_dirty = True  # Object Table is now dirty

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
        self._source = self._source.groupby([self._id_col, self._band_col, tmp_time_col]).aggregate(aggr_funs)

        # Fix the indices and remove the temporary column.
        self._source = self._source.reset_index().set_index(self._id_col).drop(tmp_time_col, axis=1)

        # Mark the source table as dirty.
        self._source_dirty = True
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
            partition. This can be checked using
            `Ensemble.check_lightcurve_cohesion`. If False, a groupby will be
            performed instead.
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
            initial sync is performed between the object and source tables. If
            not performed, dynamic information like the number of observations
            may be out of date until a sync is performed internally.
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

        # Set the index of the source frame and save the resulting table
        self._source = source_frame.set_index(self._id_col, drop=True, sorted=sorted, sort=sort)

        if object_frame is None:  # generate an indexed object table from source
            self._object = self._generate_object_table()

        else:
            self._object = object_frame
            self._object = self._object.set_index(self._id_col, sorted=sorted, sort=sort)

            # Optionally sync the tables, recalculates nobs columns
            if sync_tables:
                self._source_dirty = True
                self._object_dirty = True
                self._sync_tables()

        if npartitions and npartitions > 1:
            self._source = self._source.repartition(npartitions=npartitions)
        elif partition_size:
            self._source = self._source.repartition(partition_size=partition_size)

        return self

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
        object_file: 'str'
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
            if self._provenance_col is not None:
                columns.append(self._provenance_col)

        # Read in the source parquet file(s)
        source = dd.read_parquet(source_file, index=self._id_col, columns=columns, split_row_groups=True)

        # Generate a provenance column if not provided
        if self._provenance_col is None:
            source["provenance"] = provenance_label
            self._provenance_col = "provenance"

        object = None
        if object_file:
            # Read in the object file(s)
            object = dd.read_parquet(object_file, index=self._id_col, split_row_groups=True)
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

    def from_source_dict(self, source_dict, column_mapper=None, npartitions=1, sort=False, **kwargs):
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
        source_frame = dd.DataFrame.from_dict(source_dict, npartitions=npartitions)

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
                self._source = self._source.assign(
                    **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col] / x[zero_point])}
                )
            else:
                self._source = self._source.assign(
                    **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col] / zero_point)}
                )

        elif zp_form == "magnitude" or zp_form == "mag":  # mag = -2.5*np.log10(flux) + zp
            if isinstance(zero_point, str):
                self._source = self._source.assign(
                    **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col]) + x[zero_point]}
                )
            else:
                self._source = self._source.assign(
                    **{out_col_name: lambda x: -2.5 * np.log10(x[flux_col]) + zero_point}
                )
        else:
            raise ValueError(f"{zp_form} is not a valid zero_point format.")

        # Calculate Errors
        if err_col is not None:
            self._source = self._source.assign(
                **{out_col_name + "_err": lambda x: (2.5 / np.log(10)) * (x[err_col] / x[flux_col])}
            )

        return self

    def _generate_object_table(self):
        """Generate an empty object table from the source table."""
        sor_idx = self._source.index.unique()
        obj_df = pd.DataFrame(index=sor_idx)
        res = dd.from_pandas(obj_df, npartitions=int(np.ceil(self._source.npartitions / 100)))

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
        if table == "object" and self._source_dirty:  # object table should be updated
            self._sync_tables()
        elif table == "source" and self._object_dirty:  # source table should be updated
            self._sync_tables()
        elif table == "all" and (self._source_dirty or self._object_dirty):
            self._sync_tables()
        return self

    def _sync_tables(self):
        """Sync operation to align both tables.

        Filtered objects are always removed from the source. But filtered
        sources may be kept in the object table is the Ensemble's
        keep_empty_objects attribute is set to True.
        """

        if self._object_dirty:
            # Sync Object to Source; remove any missing objects from source
            obj_idx = list(self._object.index.compute())
            self._source = self._source.map_partitions(lambda x: x[x.index.isin(obj_idx)])
            self._source = self._source.persist()  # persist the source frame

            # Drop Temporary Source Columns on Sync
            if len(self._source_temp):
                self._source = self._source.drop(columns=self._source_temp)
                print(f"Temporary columns dropped from Source Table: {self._source_temp}")
                self._source_temp = []

        if self._source_dirty:  # not elif
            if not self.keep_empty_objects:
                # Sync Source to Object; remove any objects that do not have sources
                sor_idx = list(self._source.index.unique().compute())
                self._object = self._object.map_partitions(lambda x: x[x.index.isin(sor_idx)])
                self._object = self._object.persist()  # persist the object frame

            # Drop Temporary Object Columns on Sync
            if len(self._object_temp):
                self._object = self._object.drop(columns=self._object_temp)
                print(f"Temporary columns dropped from Object Table: {self._object_temp}")
                self._object_temp = []

        # Now synced and clean
        self._source_dirty = False
        self._object_dirty = False
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
