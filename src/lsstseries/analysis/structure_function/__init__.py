from .base_argument_container import StructureFunctionArgumentContainer
from .base_calculator import StructureFunctionCalculator, register_sf_subclasses
from .basic.calculator import BasicStructureFunctionCalculator
from .bauer_2009a.calculator import Bauer2009AStructureFunctionCalculator
from .bauer_2009b.calculator import Bauer2009BStructureFunctionCalculator
from .macleod_2012.calculator import Macleod2012StructureFunctionCalculator
from .schmidt_2010.calculator import Schmidt2010StructureFunctionCalculator

# The dictionary of all available subclasses of the StructureFunctionCalculator.
# It will have the form `{"unique_name" : class}`.
SF_METHODS = register_sf_subclasses()
