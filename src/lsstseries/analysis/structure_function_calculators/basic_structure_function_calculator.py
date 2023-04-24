from lsstseries.analysis.structure_function_calculators.structure_function_calculator import StructureFunctionCalculator
from lsstseries.analysis.structure_function_argument_containers.structure_function_argument_container import StructureFunctionArgumentContainer

class BasicStructureFunctionCalculator(StructureFunctionCalculator):
    def __init__(self, time:list[float], flux:list[float], err: list[float], argument_container:StructureFunctionArgumentContainer):
        # Not sure if there's any addition data manipulation that will be 
        # needed for this calculator

        super().__init__(time, flux, err, argument_container)

    def calculate(self):
        # Do the actual calculation here
        print("Doin' the calculation")

    @staticmethod
    def name_id():
        return "basic"

    @staticmethod
    def expected_argument_container():
        return StructureFunctionArgumentContainer