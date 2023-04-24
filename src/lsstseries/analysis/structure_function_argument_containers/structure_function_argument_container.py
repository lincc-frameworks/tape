from dataclasses import dataclass, field

@dataclass
class StructureFunctionArgumentContainer():
    bands_to_calc: str | list[str] = field(default=None)
    combine: bool = field(default=False)
    bins: list[float] = field(default=None)
    method: str = field(default="size")
    sthresh: int = field(default=100)
    use_timestamps: bool = field(default=True)
    sf_calculator_method: str = field(default="basic")

    def __post_init__(self):
        # Nothing here yet
        return
