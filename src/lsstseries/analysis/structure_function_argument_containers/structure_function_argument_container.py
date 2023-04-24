from dataclasses import dataclass, field


@dataclass
class StructureFunctionArgumentContainer:
    band_to_calc: str | list[str] = field(default=None)
    combine: bool = field(default=False)
    sf_calculator: str = field(default="basic")
    bins: list[float] = field(default=None)
    method: str = field(default="size")
    sthresh: int = field(default=100)
    use_timestamps: bool = field(default=True)

    def __post_init__(self):
        # Nothing here yet
        return
