from tape.analysis.structure_function.base_calculator import StructureFunctionCalculator


def register_sf_subclasses():
    """This method will identify all of the subclasses of `StructureFunctionCalculator`
    and build a dictionary that maps `name : subclass`.

    Returns
    -------
    dict
        A dictionary of all of subclasses of `StructureFunctionCalculator`. Where
        the str returned from `subclass.name_id()` is the key, and the class is
        the value.

    Raises
    ------
    ValueError
        If a duplicate key is found, a ValueError will be raised. This would
        likely occur if a user copy/pasted an existing subclass but failed to
        update the unique name_id string.
    """
    subclass_dict = {}
    for subcls in StructureFunctionCalculator.__subclasses__():
        if subcls.name_id() in subclass_dict:
            raise ValueError(
                "Attempted to add duplicate Structure Function calculator name to SF_METHODS: "
                + str(subcls.name_id())
            )

        subclass_dict[subcls.name_id()] = subcls

    return subclass_dict


def update_sf_subclasses():
    """This function is used to register newly created subclasses of the
    `StructureFunctionCalculator`.
    """
    for subcls in StructureFunctionCalculator.__subclasses__():
        if subcls.name_id() not in SF_METHODS.keys():
            SF_METHODS[subcls.name_id()] = subcls


# The dictionary of all available subclasses of the StructureFunctionCalculator.
SF_METHODS = register_sf_subclasses()
