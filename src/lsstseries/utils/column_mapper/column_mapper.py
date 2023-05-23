class ColumnMapper:
    """Maps columns from a given dataset into known ensemble column"""

    def __init__(
        self,
        id_col=None,
        time_col=None,
        flux_col=None,
        err_col=None,
        band_col=None,
        provenance_col=None,
        nobs_total_col=None,
        nobs_band_cols=None,
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
        nobs_col: list of 'str', optional
            Identifies which columns contain number of observations for each
            band, if available in the input object file
        nobs_tot_col: 'str', optional
            Identifies which column contains the total number of observations,
            if available in the input object file
        provenance_col: 'str', optional
            Identifies which column contains the provenance information, if
            None the provenance column is generated.

        Returns
        -------
        ColumnMapper object
        """

        self.map = {
            "id_col": id_col,
            "time_col": time_col,
            "flux_col": flux_col,
            "err_col": err_col,
            "band_col": band_col,
            "provenance_col": provenance_col,
            "nobs_total_col": nobs_total_col,
            "nobs_band_cols": nobs_band_cols,
        }

        # Specifies which column mappings must be set for the Ensemble
        self.required = {
            "id_col": True,
            "flux_col": True,
            "time_col": True,
            "err_col": True,
            "band_col": True,
            "provenance_col": False,
            "nobs_total_col": False,
            "nobs_band_cols": False,
        }

        self.known_maps = {"ZTF": ZTFColumnMapper}

    def _set_known_map(self):
        """Must be defined in a known map class"""
        raise NotImplementedError

    @staticmethod
    def map_id() -> str:
        return None

    def use_known_map(self, map_id):
        """Use a known mapping scheme

        Parameters
        ----------
        map_id: 'str'
            Identifies which mapping scheme to use

        Returns
        -------
        A ColumnMapper subclass object dependent on the map_id provided, for example
        ZTFColumnMapper in the case of "ZTF"

        """
        if map_id in self.known_maps.keys():
            if map_id == "ZTF":
                self = ZTFColumnMapper()._set_known_map()
            return self
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
        required_keys = [item[0] for item in self.required.items() if item[1]]

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
        provenance_col=None,
        nobs_total_col=None,
        nobs_band_cols=None,
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
        provenance_col: 'str', optional
            Identifies which column contains the provenance information, if
            None the provenance column is generated.
        """
        assign_map = {
            "id_col": id_col,
            "time_col": time_col,
            "flux_col": flux_col,
            "err_col": err_col,
            "band_col": band_col,
            "provenance_col": provenance_col,
            "nobs_total_col": nobs_total_col,
            "nobs_band_cols": nobs_band_cols,
        }

        for item in assign_map.items():
            if item[1] is not None:
                self.map[item[0]] = item[1]

        return self


class ZTFColumnMapper(ColumnMapper):
    """This class establishs a known mapping to Zwicky Transient Facility (ZTF)
    catalog data columns"""

    def _set_known_map(self):
        self.map = {
            "id_col": "ps1_objid",
            "time_col": "midPointTai",
            "flux_col": "psFlux",
            "err_col": "psFluxErr",
            "band_col": "filterName",
            "provenance_col": None,
            "nobs_total_col": "nobs_total",
            "nobs_band_cols": None,
        }
        return self

    @staticmethod
    def map_id() -> str:
        return "ZTF"
