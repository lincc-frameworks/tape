from .basic_structure_function_calculator import BasicStructureFunctionCalculator
from .structure_function_calculator import StructureFunctionCalculator

SF_METHODS = {subcls.name_id(): subcls for subcls in StructureFunctionCalculator.__subclasses__()}
