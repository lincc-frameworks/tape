from .basic_structure_function_calculator import BasicStructureFunctionCalculator
from .structure_function_calculator import StructureFunctionCalculator

# This dynamically generates the dictionary of all available subclasses of the
# StructureFunctionCalculator. It will have the form `{"unique_name" : class}`.
SF_METHODS = {subcls.name_id(): subcls for subcls in StructureFunctionCalculator.__subclasses__()}
