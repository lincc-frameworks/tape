"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

#import example_benchmarks
import numpy as np
import pandas as pd
from tape.ensemble import Ensemble


#def time_computation():
#    """Time computations are prefixed with 'time'."""
#    example_benchmarks.runtime_computation()


#def mem_list():
#    """Memory computations are prefixed with 'mem' or 'peakmem'."""
#    return example_benchmarks.memory_computation()


def time_basic_workflow():
    np.random.seed(1)

    # Generate 10 astronomical objects
    n_obj = 10
    ids = 8000 + np.arange(n_obj)
    names = ids.astype(str)
    object_table = pd.DataFrame(
        {
            "id": ids,
            "name": names,
            "ddf_bool": np.random.randint(0, 2, n_obj), # 0 if from deep drilling field, 1 otherwise
            "libid_cadence": np.random.randint(1, 130, n_obj),
        }
    )

    # Create 1000 lightcurves with 100 measurements each
    lc_len = 100
    num_points = 1000
    all_bands = np.array(["r", "g", "b", "i"])
    source_table = pd.DataFrame(
        {
            "id": 8000 + (np.arange(num_points) % n_obj),
            "time": np.arange(num_points),
            "flux":  np.random.random_sample(size=num_points)*10,
            "band": np.repeat(all_bands, num_points / len(all_bands)),
            "error": np.random.random_sample(size=num_points),
            "count": np.arange(num_points),
        },
    )

    ens = Ensemble()  # initialize an ensemble object

    # Read in the generated lightcurve data
    ens.from_pandas(
        source_frame=source_table,
        object_frame=object_table,
        id_col="id",
        time_col="time",
        flux_col="flux",
        err_col="error",
        band_col="band",
        npartitions=1)
