import pytest
import time
from tape.analysis.stetsonj import calc_stetson_J
from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structurefunction2 import calc_sf2


@pytest.mark.benchmark(
    group="test",
    max_time=2.0,
    timer=time.time,
    disable_gc=True,
    warmup=False
)
def test_my_stuff(benchmark):
    @benchmark
    def result():
        # Code to be measured
        return time.sleep(0.000001)

    # Extra code, to verify that the run
    # completed correctly.
    # Note: this code is not measured.
    assert result is None


@pytest.mark.benchmark(
    group="stetson-j",
    timer=time.time,
    disable_gc=True,
    warmup=False
)
def test_stetson_j(benchmark, parquet_ensemble):
    ens = parquet_ensemble
    result = benchmark(ens.batch, calc_stetson_J)

    # Extra code, to verify that the run
    # completed correctly.
    # Note: this code is not measured.
    assert result is not None

@pytest.mark.benchmark(
    group="stetson-j-oneband",
    timer=time.time,
    disable_gc=True,
    warmup=False
)
def test_stetson_j_oneband(benchmark, parquet_ensemble):
    ens = parquet_ensemble
    result = benchmark(ens.batch, calc_stetson_J, band_to_calc="i")

    # Extra code, to verify that the run
    # completed correctly.
    # Note: this code is not measured.
    assert result is not None



@pytest.mark.parametrize("sf_method", ["basic", "macleod_2012", "bauer_2009a", "bauer_2009b", "schmidt_2010"])
@pytest.mark.benchmark()
def test_calc_sf2(benchmark, parquet_ensemble, sf_method, use_map=True):

    ens = parquet_ensemble

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "loglength"
    arg_container.combine = False
    arg_container.bin_count_target = 50
    arg_container.sf_method = sf_method

    benchmark.group = f'SF2 - {sf_method}'
    res_batch = benchmark(ens.batch, calc_sf2, use_map=use_map, argument_container=arg_container)

    assert res_batch is not None