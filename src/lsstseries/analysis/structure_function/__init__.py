from .base_argument_container import StructureFunctionArgumentContainer  # noqa
from .base_calculator import StructureFunctionCalculator  # noqa
from .basic.calculator import BasicStructureFunctionCalculator  # noqa
from .bauer_2009a.calculator import Bauer2009AStructureFunctionCalculator  # noqa
from .bauer_2009b.calculator import Bauer2009BStructureFunctionCalculator  # noqa
from .macleod_2012.calculator import Macleod2012StructureFunctionCalculator  # noqa
from .schmidt_2010.calculator import Schmidt2010StructureFunctionCalculator  # noqa

# This dynamically generates the dictionary of all available subclasses of the
# StructureFunctionCalculator. It will have the form `{"unique_name" : class}`.
SF_METHODS = {subcls.name_id(): subcls for subcls in StructureFunctionCalculator.__subclasses__()}
