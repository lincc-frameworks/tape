from .base_argument_container import StructureFunctionArgumentContainer
from .base_calculator import StructureFunctionCalculator
from .basic.calculator import BasicStructureFunctionCalculator

# This dynamically generates the dictionary of all available subclasses of the
# StructureFunctionCalculator. It will have the form `{"unique_name" : class}`.
SF_METHODS = {subcls.name_id(): subcls for subcls in StructureFunctionCalculator.__subclasses__()}
