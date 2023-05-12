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
        """"""

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

    def is_ready(self, show_needed=False):
        """shows whether the ColumnMapper has all critical columns assigned"""

        # Grab required column keys
        required_keys = [item[0] for item in self.required.items() if item[1]]

        # Check the map for assigned keys
        ready = True
        needed = []
        for key in required_keys:
            if self.map[key] is None:
                needed.append(key)
                ready = False

        #
        if show_needed:
            return (ready, needed)
        else:
            return ready

    def known_map(self, dataset_label):
        """Applies a known dataset mapping"""
        pass

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
        """Updates a given set of columns"""
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
