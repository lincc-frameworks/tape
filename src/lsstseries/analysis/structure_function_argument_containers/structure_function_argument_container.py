from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class StructureFunctionArgumentContainer:
    band: List[str] = field(default=None)
    lc_id: Union[int, List[int]] = field(default=None)
    sf_method: str = field(default="basic")
    band_to_calc: Union[str, List[str]] = field(default=None)
    combine: bool = field(default=False)
    bins: List[float] = field(default=None)
    bin_method: str = field(default="size")
    bin_count_target: int = field(default=100)
    use_timestamps: bool = field(default=True)

    def __post_init__(self):
        # Nothing here yet
        return
