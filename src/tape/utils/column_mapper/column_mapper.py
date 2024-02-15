from collections import namedtuple


class ColumnMapper:
    """Maps columns from a given dataset into known ensemble column"""

    def __init__(
        self,
        id_col=None,
        time_col=None,
        flux_col=None,
        err_col=None,
        band_col=None,
    ):
        """

        Parameters
        ----------
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

        Returns
        -------
        ColumnMapper object
        """

        Column = namedtuple("Column", ["name", "is_required"])

        self.map = {
            "id_col": id_col,
            "time_col": time_col,
            "flux_col": flux_col,
            "err_col": err_col,
            "band_col": band_col,
        }

        self.required = [
            Column("id_col", True),
            Column("time_col", True),
            Column("flux_col", True),
            Column("err_col", True),
            Column("band_col", True),
        ]

        self.known_maps = {"ZTF": ZTFColumnMapper, "PS1": PS1ColumnMapper}

    def _set_known_map(self, hipscat=True):
        """Must be defined in a known map class"""
        raise NotImplementedError

    def use_known_map(self, map_id, hipscat=True):
        """Use a known mapping scheme

        Parameters
        ----------
        map_id: 'str'
            Identifies which mapping scheme to use
        hipscat: 'bool'
            Indicates whether the data is in hipscat format or not, which will
            affect the chosen ID column (_hipscat_index will be used when
            hipscat is true. True by default.

        Returns
        -------
        A ColumnMapper subclass object dependent on the map_id provided,
        ZTFColumnMapper in the case of "ZTF" for example

        """
        if map_id.upper() in self.known_maps:
            return self.known_maps[map_id.upper()]()._set_known_map(hipscat=hipscat)
        else:
            raise ValueError(f'Unknown Mapping: "{map_id}"')

    def is_ready(self, show_needed=False):
        """shows whether the ColumnMapper has all critical columns assigned

        Parameters
        ----------
        show_needed: 'bool', optional
            Indicates whether to also return a list of missing columns

        Returns
        -------
        `bool` or tuple of (bool, list) dependent on show_needed parameter

        """

        # Grab required column keys
        required_keys = [col.name for col in self.required if col.is_required]

        # Check the map for assigned keys
        ready = True
        needed = []
        for key in required_keys:
            if self.map[key] is None:
                needed.append(key)
                ready = False

        if show_needed:
            return (ready, needed)
        else:
            return ready

    def assign(
        self,
        id_col=None,
        time_col=None,
        flux_col=None,
        err_col=None,
        band_col=None,
    ):
        """Updates a given set of columns

        Parameters
        ----------
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
        nobs_col: list of 'str', optional
            Identifies which columns contain number of observations for each
            band, if available in the input object file
        nobs_tot_col: 'str', optional
            Identifies which column contains the total number of observations,
            if available in the input object file
        """
        assign_map = {
            "id_col": id_col,
            "time_col": time_col,
            "flux_col": flux_col,
            "err_col": err_col,
            "band_col": band_col,
        }

        for item in assign_map.items():
            if item[1] is not None:
                self.map[item[0]] = item[1]

        return self


class ZTFColumnMapper(ColumnMapper):
    """This class establishs a known mapping to Zwicky Transient Facility (ZTF)
    catalog data columns"""

    def _set_known_map(self, hipscat=True):
        """sets the map to a pre-defined ZTF mapping"""
        if hipscat:
            id_col = "_hipscat_index"
        else:
            id_col = "ps1_objid"
        self.map = {
            "id_col": id_col,
            "time_col": "mjd",
            "flux_col": "mag",
            "err_col": "magerr",
            "band_col": "band",
        }
        return self


class PS1ColumnMapper(ColumnMapper):
    """This class establishs a known mapping to Pan-STARRs (PS1)
    catalog data columns"""

    def _set_known_map(self, hipscat=True):
        """sets the map to a pre-defined PS1 mapping"""
        if hipscat:
            id_col = "_hipscat_index"
        else:
            id_col = "objID"
        self.map = {
            "id_col": id_col,
            "time_col": "obsTime",
            "flux_col": "apFlux",
            "err_col": "apFluxErr",
            "band_col": "filterID",
        }
        return self
