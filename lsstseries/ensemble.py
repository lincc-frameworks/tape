import pandas as pd
import pyvo as vo
from .timeseries import timeseries


class ensemble():
    """ensemble object is a collection of light curve ids"""
    def __init__(self, token=None):
        self.result = None  # holds the latest query
        self.token = token

        self._time_col = 'midPointTai'
        self._flux_col = 'psFlux'
        self._err_col = 'psFluxErr'

    def tap_token(self, token):
        """Add/update a TAP token to the class, enables querying
        Read here for information on TAP access: https://data.lsst.cloud/api-aspect

        Parameters
        ----------
        token : `str`
            Token string
        """
        self.token = token

    def query_tap(self, query, maxrec=None):
        """Query the TAP service

        Parameters
        ----------
        query : `str`
            Query is an ADQL formatted string
        maxrec: `int`
            Max number of results returned

        Returns
        ----------
        result: `pd.df`
            Result of the query, as pandas dataframe
        """
        cred = vo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", self.token)

        service = vo.dal.TAPService("https://data.lsst.cloud/api/tap", 
                                    cred.get("ivo://ivoa.net/sso#BasicAA"))
        results = service.search(query, maxrec=maxrec)
        result = results.to_table().to_pandas()
        self.result = result
        return result

    def query_ids(self, ids,
                  time_col='midPointTai',
                  flux_col='psFlux',
                  err_col='psFluxErr',
                  add_cols=[],
                  id_field='diaObjectId',
                  catalog='dp02_dc2_catalogs',
                  table='DiaSource',
                  maxrec=None):
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
        core_cols = [time_col,flux_col,err_col]
        cols = core_cols+add_cols
        idx_cols = ['diaObjectId', 'filterName']

        result = pd.DataFrame(columns=idx_cols+cols)
        select_cols = ",".join(idx_cols)+','+','.join(cols)
        str_ids = [str(obj_id) for obj_id in ids]
        id_list = "("+",".join(str_ids)+")"

        result = self.query_tap(f"SELECT {select_cols} "
                                f"FROM {catalog}.{table} "
                                f"WHERE {id_field} IN {id_list}")
        index = self._build_index(result['diaObjectId'], result['filterName'])
        result.index = index
        result = result[cols].sort_index()
        self.result = result

        self._time_col = time_col
        self._flux_col = flux_col
        self._err_col = err_col

        return result

    def to_timeseries(self, dataframe, target, time_col=None, 
                      flux_col=None, err_col=None):
        """Construct a timeseries object from one target object_id, assumes that the result
        is a collection of lightcurves (output from query_ids)

        Parameters
        ----------
        dataframe: `pd.df`
            Ensemble object
        target: `int`
            Id of a source to be extracted

        Returns
        ----------
        ts: `timeseries`
            Timeseries for a single object
        """

        # Without a specified column, use defaults (which are updated via query_id)
        if time_col is None:
            time_col = self._time_col
        if flux_col is None:
            flux_col = self._flux_col
        if err_col is None:
            err_col = self._err_col

        df = dataframe.xs(target)
        ts = timeseries()._from_ensemble(data=df, object_id=target, time_label=time_col, 
                                         flux_label=flux_col, err_label=err_col)
        return ts

    def _build_index(self, obj_id, band):
        """Build pandas multiindex from object_ids and bands"""
        # Create a multiindex

        idx = range(len(list(zip(obj_id, band))))
        tuples = zip(obj_id, band, idx)
        index = pd.MultiIndex.from_tuples(tuples, names=["object_id", "band", "index"])
        return index
