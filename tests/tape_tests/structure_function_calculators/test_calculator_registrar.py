import pytest

from tape.analysis.structure_function.calculator_registrar import (
    SF_METHODS,
    register_sf_subclasses,
    update_sf_subclasses,
)
from tape.analysis.structure_function.base_calculator import StructureFunctionCalculator
from tape.analysis.structure_function.basic.calculator import BasicStructureFunctionCalculator


def test_register_sf_subclasses():
    """Base test to ensure that we register the most basic subclass of
    StructureFunctionCalculator
    """
    output = register_sf_subclasses()
    assert output["basic"] == BasicStructureFunctionCalculator


def test_register_sf_subclasses_duplicate_name():
    """Create an child class of StructureFunctionCalculator with an intentionally
    duplicate name to check the assertion of `register_sf_subclasses`.
    """

    class DuplicateStructureFunction(StructureFunctionCalculator):
        def calculate(self):
            return 1

        @staticmethod
        def name_id():
            return "basic"

    with pytest.raises(ValueError) as excinfo:
        _ = register_sf_subclasses()

    assert "Attempted to add duplicate Structure" in str(excinfo.value)


def test_update_sf_subclasses():
    """Create a new child class of StructureFunctionCalculator and call
    `update_sf_sublclasses`. Ensure that SF_METHODS has been properly updated.
    """

    class SimpleStructureFunction(StructureFunctionCalculator):
        def calculate(self):
            return 1

        @staticmethod
        def name_id():
            return "simple_sf"

    pre_update_keys = SF_METHODS.keys()
    update_sf_subclasses()
    post_update_keys = SF_METHODS.keys()

    assert "simple_sf" in post_update_keys
    for pre_key in pre_update_keys:
        assert pre_key in post_update_keys
