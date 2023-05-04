from .base_argument_container import StructureFunctionArgumentContainer
from .base_calculator import StructureFunctionCalculator
from .basic.calculator import BasicStructureFunctionCalculator
from .iqr.calculator import IqrStructureFunctionCalculator
from .macleod_2012.calculator import Macleod2012StructureFunctionCalculator

# This dynamically generates the dictionary of all available subclasses of the
# StructureFunctionCalculator. It will have the form `{"unique_name" : class}`.
SF_METHODS = {subcls.name_id(): subcls for subcls in StructureFunctionCalculator.__subclasses__()}
