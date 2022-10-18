import pandas as pd
import numpy as np
import pyvo as vo
from .timeseries import timeseries

class ensemble():
    """ensemble object is a collection of light curve ids"""
    def __init__(self,token=None):
        self.result = None #holds the latest query
        self.token = token

    def tap_token(self, token):
        """Add/update a TAP token to the class, enables querying
        Read here for information on TAP access: https://data.lsst.cloud/api-aspect
        """
        self.token=token

    def query_tap(self,query,maxrec=None):
        """query the TAP service, query is an ADQL formatted string"""
        cred = vo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", self.token)

        service = vo.dal.TAPService("https://data.lsst.cloud/api/tap",cred.get("ivo://ivoa.net/sso#BasicAA"))
        results = service.search(query, maxrec=maxrec)
        result = results.to_table().to_pandas()
        self.result = result
        return result

    def query_ids(self, ids, 
                  core_cols=['midPointTai','psFlux','psFluxErr'],
                  add_cols=[],
                  id_field='diaObjectId',
                  catalog='dp02_dc2_catalogs',
                  table='DiaSource',
                  maxrec=None):
        """query based on a list of object ids"""
        cols = core_cols+add_cols
        idx_cols = ['diaObjectId','filterName']

        result = pd.DataFrame(columns=idx_cols+cols)
        select_cols = ",".join(idx_cols)+','+','.join(cols)
        str_ids = [str(obj_id) for obj_id in ids]
        id_list = "("+",".join(str_ids)+")"

        result = self.query_tap(f"SELECT {select_cols} "
                            f"FROM {catalog}.{table} "
                            f"WHERE {id_field} IN {id_list}")
        index = self._build_index(result['diaObjectId'],result['filterName'])
        result.index = index
        result = result[cols].sort_index()
        self.result = result
        return result

    def to_timeseries(self,dataframe,target,core_cols=['midPointTai','psFlux','psFluxErr'],add_cols=[]):
        """construct a timeseries object from one target object_id, assumes that the result
        is a collection of lightcurves (output from query_ids)"""
        cols = core_cols+add_cols
        df = dataframe.xs(target)
        ts = timeseries()._from_ensemble(data=df,object_id=target)
        return ts

    def _build_index(self,obj_id, band):
        """Build pandas multiindex from object_ids and bands"""
        #Create a multiindex

        idx = range(len(list(zip(obj_id,band))))
        tuples = zip(obj_id,band,idx)
        index = pd.MultiIndex.from_tuples(tuples, names=["object_id","band", "index"])
        return index

