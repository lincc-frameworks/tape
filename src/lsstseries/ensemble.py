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
        self._data = None

        # Assign Default Values for critical column quantities
        self._id_col = "object_id"
        self._time_col = "midPointTai"
        self._flux_col = "psFlux"
        self._err_col = "psFluxErr"
        self._band_col = "band"

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

    def insert(self, obj_ids, bands, timestamps, fluxes, flux_errs=None, **kwargs):
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
        df2 = df2.set_index(self._id_col, drop=True)

        # Save the divisions and number of partitions.
        prev_div = self._data.divisions
        prev_num = self._data.npartitions

        # Append the new rows to the correct divisions.
        self._data = dd.concat([self._data, df2], axis=0, interleave_partitions=True)

        # If the divisions were set, reuse them. Otherwise, use the same
        # number of partitions.
        if all(prev_div):
            self._data = self._data.repartition(divisions=prev_div)
        elif self._data.npartitions != prev_num:
            self._data = self._data.repartition(npartitions=prev_num)

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

    def info(self, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.info()"""
        return self._data.info(**kwargs)

    def compute(self, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.compute()"""
        return self._data.compute(**kwargs)

    def columns(self):
        """Retrieve columns from dask dataframe"""
        return self._data.columns

    def head(self, n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.head()"""

        return self._data.head(n=n, **kwargs)

    def tail(self, n=5, **kwargs):
        """Wrapper for dask.dataframe.DataFrame.head()"""

        return self._data.tail(n=n, **kwargs)

    def count(self, sort=True, ascending=False):
        """Return the number of available rows/measurements for each lightcurve

        Parameters
        ----------
        sort: `bool`, optional
            Indicates whether the resulting counts should be sorted on counts
        ascending: `bool`, optional
            When sorting, use ascending (lowest counts first) or descending (highest counts first)
            or descending

        Returns
        ----------
        counts: `pandas.series`
            A series of counts by object
        """
        counts = self._data.groupby(self._id_col)[self._time_col].count().compute()
        if sort:
            return counts.sort_values(ascending=ascending)
        else:
            return counts

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
        self._data = self._data[self._data.isnull().sum(axis=1) < threshold]
        return self

    def prune(self, threshold=50, col_name="num_obs"):
        """remove objects with less observations than a given threshold

        Parameters
        ----------
        threshold: `int`, optional
            The minimum number of observations needed to retain an object.
            Default is 50.
        col_name: `str`, optional
            The name of the output counts column. If already exists, directly
            uses the column to prune the ensemble.

        Returns
        ----------
        ensemble: `lsstseries.ensemble.Ensemble`
            The ensemble object with pruned rows removed
        """
        if col_name not in self._data.columns:
            counts = self._data.groupby(self._id_col).count()
            counts = counts.rename(columns={self._time_col: col_name})[[col_name]]
            self._data = self._data.join(counts, how="left")
        self._data = self._data[self._data[col_name] >= threshold]
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
            batch = self._data.map_partitions(
                lambda x: x.groupby(id_col, group_keys=False).apply(
                    lambda y: func(
                        *[y[arg].to_numpy() if arg != id_col else y.index.to_numpy() for arg in args],
                        **kwargs,
                    )
                ),
                meta=meta,
            )
        else:  # use groupby
            batch = self._data.groupby(self._id_col, group_keys=False).apply(
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
        file,
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
        file: 'str'
            Path to a parquet file, or multiple parquet files to be read into
            the ensemble
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

        # Read in a parquet file
        self._data = dd.read_parquet(file, index=self._id_col, columns=columns, split_row_groups=True)

        if npartitions and npartitions > 1:
            self._data = self._data.repartition(npartitions=npartitions)
        elif partition_size:
            self._data = self._data.repartition(partition_size=partition_size)

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
                pre_var, post_var = col[:pos_flux], col[pos_flux + len("Flux") :]
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
