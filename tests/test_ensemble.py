"""Test ensemble manipulations"""

from lsstseries import ensemble


def test_build_index():
    """
    Test that ensemble indexing returns expected behavior
    """

    obj_ids = [1, 1, 1, 2, 1, 2, 2]
    bands = ["u", "u", "u", "g", "g", "u", "u"]

    ens = ensemble()
    result = list(ens._build_index(obj_ids, bands).get_level_values(2))
    target = [0, 1, 2, 0, 0, 0, 1]
    assert result == target
