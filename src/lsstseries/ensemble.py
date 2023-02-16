import time

import dask.dataframe as dd
import pandas as pd
import pyvo as vo
from dask.distributed import Client

from .analysis.structurefunction2 import calc_sf2
from .timeseries import timeseries


class ensemble:
    """ensemble object is a collection of light curve ids"""

    client = None

    def __init__(self, token=None, client=None, **kwargs):
        self.result = None  # holds the latest query
        self.token = token
        self.data = None

        # Assign Default Values for critical column quantities
        self._id_col = "object_id"
        self._time_col = "midPointTai"
        self._flux_col = "psFlux"
        self._err_col = "psFluxErr"
        self._band_col = "band"

        # Setup Dask Distributed Client
        if client:
            self.client = client
        elif not self.client:
            self.client = Client(**kwargs)  # arguments passed along to Client

    def __enter__(self):
        self.client = Client()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
        counts = self.data.groupby(self._id_col)[self._time_col].count().compute()
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
        ensemble: `lsstseries.ensemble.ensemble`
            The ensemble object with nans removed according to the threshold
            scheme
        """
        self.data = self.data[self.data.isnull().sum(axis=1) < threshold]
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
        ensemble: `lsstseries.ensemble.ensemble`
            The ensemble object with pruned rows removed
        """
        if col_name not in self.data.columns:
            counts = self.data.groupby(self._id_col).count()
            counts = counts.rename(columns={self._time_col: col_name})[[col_name]]
            self.data = self.data.join(counts, how="left")
        self.data = self.data[self.data[col_name] >= threshold]
        return self

    def batch(self, func, meta=None, *args, **kwargs):
        """Run a function from lsstseries.timeseries on the available ids

        Parameters
        ----------
        func : `function`
            A function to apply to all objects in the ensemble
        meta : `pd.Series`, `pd.DataFrame`, `dict`, or `tuple-like`
            Dask's meta parameter, which lays down the expected structure of
            the results. Overridden by lsstseries for lsstseries
            functions. If none, attempts to coerce the result to a
            pandas.series.
        *args:
            Denotes the ensemble columns to use as inputs for a function,
            order must be correct for function. If passing a lsstseries
            function, these are populated automatically.
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

        known_meta = {
            "calc_sf2": {"lc_id": "int", "band": "str", "dt": "float", "sf2": "float"}
        }
        if func.__name__ in known_cols:
            args = known_cols[func.__name__]
        if func.__name__ in known_meta:
            meta = known_meta[func.__name__]

        if meta is None:
            meta = (self._id_col, type(self._id_col))  # return a series of ids

        id_col = self._id_col  # pre-compute needed for dask in lambda function

        batch = self.data.groupby(self._id_col).apply(
            lambda x: func(
                *[x[arg] if arg != id_col else x.index for arg in args], **kwargs
            ),
            meta=meta,
        )

        result = batch.compute()
        return result

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
        ensemble: `lsstseries.ensemble.ensemble`
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
        self.data = dd.read_parquet(
            file, index=self._id_col, columns=columns, split_row_groups=True
        )

        if npartitions and npartitions > 1:
            self.data = self.data.repartition(npartitions=npartitions)
        elif partition_size:
            self.data = self.data.repartition(partition_size=partition_size)

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
        service = vo.dal.TAPService(
            "https://data.lsst.cloud/api/tap", cred.get("ivo://ivoa.net/sso#BasicAA")
        )
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
            f"SELECT {select_cols} "
            f"FROM {catalog}.{table} "
            f"WHERE {id_field} IN {id_list}",
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
        ts: `timeseries`
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

        df = self.data.loc[target].compute()
        ts = timeseries()._from_ensemble(
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
                    cols_mag.append(
                        "scisql_nanojanskyToAbMag(" + flux_str + ") AS " + mag_str
                    )
                    cols_label.append(mag_str)
        return cols_mag, cols_label

    def _build_index(self, obj_id, band):
        """Build pandas multiindex from object_ids and bands"""
        count_dict = {}
        idx = []
        for o, b in zip(obj_id, band):
            if f"{o},{b}" in count_dict:
                idx.append(count_dict[f"{o},{b}"])
                count_dict[f"{o},{b}"] += 1
            else:
                idx.append(0)
                count_dict[f"{o},{b}"] = 1
        tuples = zip(obj_id, band, idx)
        index = pd.MultiIndex.from_tuples(tuples, names=["object_id", "band", "index"])
        return index

    def sf2(
        self, bins=None, band_to_calc=None, combine=False, method="size", sthresh=100
    ):
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
                self.data.index,
                self.data[self._time_col],
                self.data[self._flux_col],
                self.data[self._err_col],
                self.data[self._band_col],
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
            )

            return result
