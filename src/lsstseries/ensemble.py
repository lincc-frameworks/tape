import time

import dask.dataframe as dd
import pandas as pd
import pyvo as vo
from dask.distributed import Client

from .analysis.structurefunction2 import calc_sf2
from .timeseries import TimeSeries


class Ensemble:
    """Ensemble object is a collection of light curve ids"""

    def __init__(self, token=None, client=None, **kwargs):
        self.result = None  # holds the latest query
        self.token = token

        self._source = None  # Source Table
        self._object = None  # Object Table

        self._sor_dirty = False  # Source Dirty Flag
        self._obj_dirty = False  # Object Dirty Flag

        # Assign Default Values for critical column quantities
        # Source
        self._id_col = "object_id"
        self._time_col = "midPointTai"
        self._flux_col = "psFlux"
        self._err_col = "psFluxErr"
        self._band_col = "band"

        # Object, _id_col is shared
        self._nobs_col = "nobs_total"
        self._nobs_bands = []

        self.client = None
        self.cleanup_client = False
        # Setup Dask Distributed Client
        if client:
            self.client = client
        else:
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
        if self._sor_dirty or self._obj_dirty:
            self = self._sync_tables()

        print("Object Table")
        self._object.info(verbose=verbose, memory_usage=memory_usage, **kwargs)
        print("Source Table")
        self._source.info(verbose=verbose, memory_usage=memory_usage, **kwargs)

    def compute(self, table=None, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.compute()"""

        if table:
            if table == "object":
                if self._sor_dirty:  # object table should be updated
                    self = self._sync_tables()
                return self._object.compute(**kwargs)
            elif table == "source":
                if self._obj_dirty:  # source table should be updated
                    self._sync_tables()
                return self._source.compute(**kwargs)
        else:
            if self._sor_dirty or self._obj_dirty:
                self = self._sync_tables()

            return (self._object.compute(**kwargs), self._source.compute(**kwargs))

    def columns(self, table="object"):
        """Retrieve columns from dask dataframe"""
        if table == "object":
            if self._sor_dirty:  # object table should be updated
                self = self._sync_tables()
            return self._object.columns
        elif table == "source":
            if self._obj_dirty:  # source table should be updated
                self = self._sync_tables()
            return self._source.columns
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def head(self, table="object", n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.head()"""

        if table == "object":
            if self._sor_dirty:  # object table should be updated
                self = self._sync_tables()
            return self._object.head(n=n, **kwargs)
        elif table == "source":
            if self._obj_dirty:  # source table should be updated
                self = self._sync_tables()
            return self._source.head(n=n, **kwargs)
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def tail(self, table="object", n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.tail()"""

        if table == "object":
            if self._sor_dirty:  # object table should be updated
                self = self._sync_tables()
            return self._object.tail(n=n, **kwargs)
        elif table == "source":
            if self._obj_dirty:  # source table should be updated
                self = self._sync_tables()
            return self._source.tail(n=n, **kwargs)
        else:
            raise ValueError(f"{table} is not one of 'object' or 'source'")

    def dropna(self, threshold=1):
        """Removes rows with a >=`threshold` nan values.

        Parameters
        ----------
        threshold: `int`, optional
            The minimum number of nans present in a row needed to drop the row.
            Default is 1.

        Returns
        ----------
        ensemble: `lsstseries.ensemble.Ensemble`
            The ensemble object with nans removed according to the threshold
            scheme
        """
        self._source = self._source[self._source.isnull().sum(axis=1) < threshold]
        self._sor_dirty = True  # This operation modifies the source table
        return self

    def filter(self, on, criteria, table="object"):
        pass

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
        ----------
        ensemble: `lsstseries.ensemble.Ensemble`
            The ensemble object with pruned rows removed
        """
        """
        if col_name not in self._data.columns:
            counts = self._data.groupby(self._id_col).count()
            counts = counts.rename(columns={self._time_col: col_name})[[col_name]]
            self._data = self._data.join(counts, how="left")

        self._data = self._data[self._data[col_name] >= threshold]
        return self
        """
        if not col_name:
            col_name = self._nobs_col

        # Sync Required if source is dirty
        if self._sor_dirty:
            self = self._sync_tables()

        # Mask on object table
        mask = self._object[col_name] >= threshold
        self._object = self._object[mask]

        #  Join object to source; joins may not be ideal here, have to drop cols
        #  self._source = self._source.join(self._object, on=self._id_col,
        #                                 how='right', lsuffix="obj", rsuffix="sor")
        #  self._source = self._source.drop(list(self._object.columns), axis=1)

        self._obj_dirty = True  # Object Table is now dirty

        return self

    def batch(self, func, *args, meta=None, use_map=True, compute=True, **kwargs):
        """Run a function from lsstseries.TimeSeries on the available ids

        Parameters
        ----------
        func : `function`
            A function to apply to all objects in the ensemble
        *args:
            Denotes the ensemble columns to use as inputs for a function,
            order must be correct for function. If passing a lsstseries
            function, these are populated automatically.
        meta : `pd.Series`, `pd.DataFrame`, `dict`, or `tuple-like`
            Dask's meta parameter, which lays down the expected structure of
            the results. Overridden by lsstseries for lsstseries
            functions. If none, attempts to coerce the result to a
            pandas.series.
        use_map : `boolean`
            Determines whether `dask.dataframe.DataFrame.map_partitions` is
            used (True). Using map_partitions is generally more efficient, but
            requires the data from each lightcurve is housed in a single
            partition. If False, a groupby will be performed instead.
        compute: `boolean`
            Determines whether to compute the result immediately or hold for a
            later compute call.
        **kwargs:
            Additional optional parameters passed for the selected function

        Returns
        ----------
        result: `Dask.Series`
            Series of function results

        Example
        ----------
        `
        from lsstseries.analysis.stetsonj import calc_stetson_J
        ensemble.batch(calc_stetson_J, band_to_calc='i')
        `
        """

        # Needs tables to be in sync
        if self._sor_dirty or self._obj_dirty:
            self = self._sync_tables()

        known_cols = {
            "calc_stetson_J": [self._flux_col, self._err_col, self._band_col],
            "calc_sf2": [
                self._id_col,
                self._time_col,
                self._flux_col,
                self._err_col,
                self._band_col,
            ],
        }

        known_meta = {"calc_sf2": {"lc_id": "int", "band": "str", "dt": "float", "sf2": "float"}}
        if func.__name__ in known_cols:
            args = known_cols[func.__name__]
        if func.__name__ in known_meta:
            meta = known_meta[func.__name__]

        if meta is None:
            meta = (self._id_col, type(self._id_col))  # return a series of ids

        id_col = self._id_col  # pre-compute needed for dask in lambda function

        if use_map:  # use map_partitions
            id_col = self._id_col  # need to grab this before mapping
            batch = self._source.map_partitions(
                lambda x: x.groupby(id_col, group_keys=False).apply(
                    lambda y: func(
                        *[y[arg].to_numpy() if arg != id_col else y.index.to_numpy() for arg in args],
                        **kwargs,
                    )
                ),
                meta=meta,
            )
        else:  # use groupby
            batch = self._source.groupby(self._id_col, group_keys=False).apply(
                lambda x: func(
                    *[x[arg].to_numpy() if arg != id_col else x.index.to_numpy() for arg in args], **kwargs
                ),
                meta=meta,
            )

        if compute:
            return batch.compute()
        else:
            return batch

    def from_parquet(
        self,
        source_file,
        object_file=None,
        id_col=None,
        time_col=None,
        flux_col=None,
        err_col=None,
        band_col=None,
        additional_cols=True,
        npartitions=None,
        partition_size=None,
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
        id_col: 'str', optional
            Identifies which column contains the Object IDs
        time_col: 'str', optional
            Identifies which column contains the time information
        flux_col: 'str', optional
            Identifies which column contains the flux/magnitude information
        err_col: 'str', optional
            Identifies which column contains the flux/mag error information
        band_col: 'str', optional
            Identifies which column contains the band information
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
        ensemble: `lsstseries.ensemble.Ensemble`
            The ensemble object with parquet data loaded
        """

        # Track critical column changes
        if id_col is not None:
            self._id_col = id_col
        if time_col is not None:
            self._time_col = time_col
        if flux_col is not None:
            self._flux_col = flux_col
        if err_col is not None:
            self._err_col = err_col
        if band_col is not None:
            self._band_col = band_col

        if additional_cols:
            columns = None  # None will prompt read_parquet to read in all cols
        else:
            columns = [self._time_col, self._flux_col, self._err_col, self._band_col]

        # Read in the source parquet file(s)
        self._source = dd.read_parquet(
            source_file, index=self._id_col, columns=columns, split_row_groups=True
        )

        if npartitions and npartitions > 1:
            self._source = self._source.repartition(npartitions=npartitions)
        elif partition_size:
            self._source = self._source.repartition(partition_size=partition_size)

        if object_file:  # read from parquet files
            # may need to generate the object table and join in the info from parquet? To guarantee nobs cols
            self._object = dd.read_parquet(object_file, index=self._id_col, split_row_groups=True)
        else:  # generate object table from source
            self._object = self._generate_object_table()
            self._nobs_bands = [col for col in list(self._object.columns) if col != self._nobs_col]

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

        # Rename bands to nobs_[band]
        band_cols = {col: f"nobs_{col}" for col in list(res.columns)}
        res = res.rename(columns=band_cols)

        # Add total nobs
        res[self._nobs_col] = counts.groupby(self._id_col).agg("sum")

        return res

    def _sync_tables(self):
        """Sync operation to align both tables"""

        if self._obj_dirty:
            # Sync Object to Source; remove any missing objects from source
            self._source = self._source.join(self._object, on=self._id_col,
                                             how='right', lsuffix="obj", rsuffix="sor")
            self._source = self._source.drop(list(self._object.columns), axis=1)

        if self._sor_dirty:  # not elif
            # Generate a new object table; updates n_obs, removes missing ids
            new_obj = self._generate_object_table()

            # Join old obj to new obj; pulls in other existing obj columns
            self._object = new_obj.join(self._object, on=self._id_col,
                                        how="left", lsuffix="", rsuffix="_old")
            old_cols = [col for col in list(self._object.columns) if "_old" in col]
            self._object = self._object.drop(old_cols, axis=1)

        # Now synced and clean
        self._sor_dirty = False
        self._obj_dirty = False
        return self

    def tap_token(self, token):
        """Add/update a TAP token to the class, enables querying
        Read here for information on TAP access:
        https://data.lsst.cloud/api-aspect

        Parameters
        ----------
        token : `str`
            Token string
        """
        self.token = token

    def query_tap(self, query, maxrec=None, debug=False):
        """Query the TAP service

        Parameters
        ----------
        query : `str`
            Query is an ADQL formatted string
        maxrec: `int`, optional
            Max number of results returned

        Returns
        ----------
        result: `pd.df`
            Result of the query, as pandas dataframe
        """
        cred = vo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", self.token)
        service = vo.dal.TAPService("https://data.lsst.cloud/api/tap", cred.get("ivo://ivoa.net/sso#BasicAA"))
        time0 = time.time()
        results = service.search(query, maxrec=maxrec)
        time1 = time.time()
        if debug:
            print(f"Query Time: {time1-time0} (s)")
        result = results.to_table().to_pandas()
        self.result = result
        return result

    def query_ids(
        self,
        ids,
        time_col="midPointTai",
        flux_col="psFlux",
        err_col="psFluxErr",
        add_cols=None,
        id_field="diaObjectId",
        catalog="dp02_dc2_catalogs",
        table="DiaSource",
        to_mag=True,
        maxrec=None,
    ):
        """Query based on a list of object ids; applicable for DP0.2

        Parameters
        ----------
        ids: `int`
            Ids of object
        time_col: `str`
            Column to retrieve and use for time
        flux_col: `str`
            Column to retrieve and use for flux (or magnitude or any "signal")
        err_col: `str`
            Column to retrieve and use for errors
        add_cols: `list` of `str`
            Additional columns to retreive
        id_field: `str`
            Which Id is being queried
        catalog: `str`
            Source catalog
        table: `str`
            Source table

        Returns
        ----------
        result: `pd.df`
            Result of the query, as pandas dataframe
        """
        cols = [time_col, flux_col, err_col]

        if to_mag:
            flux_query, flux_label = self.flux_to_mag([flux_col])
            flux_col = flux_label[0]
            if err_col is not None:
                err_query, err_label = self.flux_to_mag([err_col])
                err_col = err_label[0]

            query_cols = [time_col] + flux_query + err_query
            cols = [time_col, flux_col, err_col]

        else:
            query_cols = cols

        if add_cols is not None:
            cols = cols + add_cols
            query_cols = query_cols + add_cols

        idx_cols = ["diaObjectId", "filterName"]

        result = pd.DataFrame(columns=idx_cols + cols)
        select_cols = ",".join(idx_cols) + "," + ",".join(query_cols)
        str_ids = [str(obj_id) for obj_id in ids]
        id_list = "(" + ",".join(str_ids) + ")"

        result = self.query_tap(
            f"SELECT {select_cols} " f"FROM {catalog}.{table} " f"WHERE {id_field} IN {id_list}",
            maxrec=maxrec,
        )
        index = self._build_index(result["diaObjectId"], result["filterName"])
        result.index = index
        result = result[cols].sort_index()
        self.result = result

        self._time_col = time_col
        self._flux_col = flux_col
        self._err_col = err_col

        return result

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

        df = self._data.loc[target].compute()
        ts = TimeSeries()._from_ensemble(
            data=df,
            object_id=target,
            time_label=time_col,
            flux_label=flux_col,
            err_label=err_col,
            band_label=band_col,
        )
        return ts

    def flux_to_mag(self, cols):
        """Transforms TAP query from fluxes to magnitudes

         Parameters
        ----------
        cols: `list` of `str`
            List of columns to be queried, containing Flux in the name

        Returns:
        ----------
        cols_mag `list` of `str`
            List of columns to be queried, replaced with magnitudes
        cols_label 'list' of 'str'
            List of column labels for the returned quantities
        """

        cols_mag = []
        cols_label = []
        for col in cols:
            pos_flux = col.find("Flux")
            if pos_flux == -1:
                cols_mag.append(col)
                cols_label.append(col)
            else:
                i = pos_flux + len("Flux")
                pre_var, post_var = col[:pos_flux], col[i:]
                flux_str = pre_var + "Flux"
                mag_str = pre_var + "AbMag"
                if col.find("Err") != -1:
                    flux_str_err = pre_var + "Flux" + post_var
                    mag_str_err = pre_var + "AbMag" + post_var
                    cols_mag.append(
                        "scisql_nanojanskyToAbMagSigma("
                        + flux_str
                        + ","
                        + flux_str_err
                        + ") AS "
                        + mag_str_err
                    )
                    cols_label.append(mag_str_err)
                else:
                    cols_mag.append("scisql_nanojanskyToAbMag(" + flux_str + ") AS " + mag_str)
                    cols_label.append(mag_str)
        return cols_mag, cols_label

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

    def sf2(self, bins=None, band_to_calc=None, combine=False, method="size", sthresh=100, use_map=True):
        """Wrapper interface for calling structurefunction2 on the ensemble

        Parameters
        ----------
        bins : `np.array` or `list`
        Manually provided bins, if not provided then bins are computed using
        the `method` kwarg
        band_to_calc : `str` or `list` of `str`
            Bands to calculate structure function on. Single band descriptor,
            or list of such descriptors.
        combine : 'bool'
            Boolean to determine whether structure function is computed for each
            light curve independently (combine=False), or computed for all light
            curves together (combine=True).
        method : 'str'
            The binning method to apply, choices of 'size'; which seeks an even
            distribution of samples per bin using quantiles, 'length'; which
            creates bins of equal length in time and 'loglength'; which creates
            bins of equal length in log time.
        sthresh : 'int'
            Target number of samples per bin.
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

        if combine:
            result = calc_sf2(
                self._data.index,
                self._data[self._time_col],
                self._data[self._flux_col],
                self._data[self._err_col],
                self._data[self._band_col],
                bins=bins,
                band_to_calc=band_to_calc,
                combine=combine,
                method=method,
                sthresh=sthresh,
            )
            return result
        else:
            result = self.batch(
                calc_sf2,
                bins=bins,
                band_to_calc=band_to_calc,
                combine=False,
                method=method,
                sthresh=sthresh,
                use_map=use_map,
            )

            return result
