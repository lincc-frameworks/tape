"""
    The following package-level methods can be used to create a new Ensemble object 
    by reading in the given data source.
"""

import requests

import dask.dataframe as dd

from tape import Ensemble
from tape.utils import ColumnMapper


def read_ensemble(
    dirpath,
    additional_frames=True,
    column_mapper=None,
    dask_client=True,
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
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.

    Returns
    ----------
    ensemble: `tape.ensemble.Ensemble`
        An ensemble object.
    """

    new_ens = Ensemble(dask_client)

    new_ens.from_ensemble(
        dirpath,
        additional_frames=additional_frames,
        column_mapper=column_mapper,
        **kwargs,
    )

    return new_ens


def read_pandas_dataframe(
    source_frame,
    object_frame=None,
    dask_client=True,
    column_mapper=None,
    sync_tables=True,
    npartitions=None,
    partition_size=None,
    **kwargs,
):
    """Read in Pandas dataframe(s) and return an ensemble object

    Parameters
    ----------
    source_frame: 'pandas.Dataframe'
        A Dask dataframe that contains source information to be read into the ensemble
    object_frame: 'pandas.Dataframe', optional
        If not specified, the object frame is generated from the source frame
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.
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

    return read_dask_dataframe(
        source_frame=source,
        object_frame=object,
        dask_client=dask_client,
        column_mapper=column_mapper,
        sync_tables=sync_tables,
        npartitions=npartitions,
        partition_size=partition_size,
        **kwargs,
    )


def read_dask_dataframe(
    source_frame,
    object_frame=None,
    dask_client=True,
    column_mapper=None,
    sync_tables=True,
    npartitions=None,
    partition_size=None,
    **kwargs,
):
    """Read in Dask dataframe(s) and return an ensemble object

    Parameters
    ----------
    source_frame: 'dask.Dataframe'
        A Dask dataframe that contains source information to be read into the ensemble
    object_frame: 'dask.Dataframe', optional
        If not specified, the object frame is generated from the source frame
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.
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
    new_ens = Ensemble(dask_client, **kwargs)

    new_ens.from_dask_dataframe(
        source_frame=source_frame,
        object_frame=object_frame,
        column_mapper=column_mapper,
        sync_tables=sync_tables,
        npartitions=npartitions,
        partition_size=partition_size,
        **kwargs,
    )

    return new_ens


def read_parquet(
    source_file,
    object_file=None,
    column_mapper=None,
    dask_client=True,
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
    object_file: 'str'
        Path to a parquet file, or multiple parquet files that contain
        object information. If not specified, it is generated from the
        source table
    column_mapper: 'ColumnMapper' object
        If provided, the ColumnMapper is used to populate relevant column
        information mapped from the input dataset.
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.
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

    new_ens = Ensemble(dask_client, **kwargs)

    new_ens.from_parquet(
        source_file=source_file,
        object_file=object_file,
        column_mapper=column_mapper,
        sync_tables=sync_tables,
        additional_cols=additional_cols,
        npartitions=npartitions,
        partition_size=partition_size,
        **kwargs,
    )

    return new_ens


def read_lsdb(
    source_catalog,
    object_catalog=None,
    column_mapper=None,
    sync_tables=False,
    sorted=True,
    sort=False,
    dask_client=True,
    **kwargs,
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
        tables.
    sorted: bool, optional
            If the index column is already sorted in increasing order.
            Defaults to True.
    sort: `bool`, optional
        If True, sorts the DataFrame by the id column. Otherwise set the
        index on the individual existing partitions. Defaults to False.
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.

    Returns
    ----------
    ensemble: `tape.ensemble.Ensemble`
        The ensemble object with the LSDB catalog data loaded.
    """

    new_ens = Ensemble(dask_client, **kwargs)

    new_ens.from_lsdb(
        source_catalog=source_catalog,
        object_catalog=object_catalog,
        column_mapper=column_mapper,
        sync_tables=sync_tables,
        sorted=sorted,
        sort=sort,
    )

    return new_ens


def read_hipscat(
    source_path,
    object_path=None,
    column_mapper=None,
    source_index=None,
    object_index=None,
    sorted=True,
    sort=False,
    dask_client=True,
    **kwargs,
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
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.

    Returns
    ----------
    ensemble: `tape.ensemble.Ensemble`
        The ensemble object with the hipscat data loaded.
    """

    new_ens = Ensemble(dask_client, **kwargs)

    new_ens.from_hipscat(
        source_path,
        object_path=object_path,
        column_mapper=column_mapper,
        source_index=source_index,
        object_index=object_index,
        sorted=sorted,
        sort=sort,
    )

    return new_ens


def read_source_dict(source_dict, column_mapper=None, npartitions=1, dask_client=True, **kwargs):
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
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.

    Returns
    ----------
    ensemble: `tape.ensemble.Ensemble`
        The ensemble object with dictionary data loaded
    """

    new_ens = Ensemble(dask_client, **kwargs)

    new_ens.from_source_dict(
        source_dict=source_dict, column_mapper=column_mapper, npartitions=npartitions, **kwargs
    )

    return new_ens


def read_dataset(dataset, dask_client=True, **kwargs):
    """Load the ensemble from a TAPE dataset.

    Parameters
    ----------
    dataset: 'str'
        The name of the dataset to import
    dask_client: `dask.distributed.client` or `bool`, optional
        Accepts an existing `dask.distributed.Client`, or creates one if
        `client=True`, passing any additional kwargs to a
        dask.distributed.Client constructor call. If `client=False`, the
        Ensemble is created without a distributed client.

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

    return read_parquet(
        source_file=dataset_info["source_file"],
        object_file=dataset_info["object_file"],
        column_mapper=col_map,
        dask_client=dask_client,
        **kwargs,
    )
