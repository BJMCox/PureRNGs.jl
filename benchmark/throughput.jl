using PureRNGs
import KernelAbstractions
import MLDataDevices
using Printf

# Load the backend package and set `target_device` before including this
# file. For example:
#
# import CUDA
# target_device = MLDataDevices.CUDADevice()
# next_fill_function = randexp_next!
# result_type = Float32
# elements = 2^27
# include("benchmark/throughput.jl")
#
# AMDGPUDevice, MetalDevice, and CPUDevice work the same way.
@isdefined(target_device) || (target_device = MLDataDevices.CPUDevice())
@isdefined(family) || (family = Philox4x32)
@isdefined(result_type) || (result_type = UInt64)
@isdefined(elements) || (elements = 2^22)
@isdefined(seconds) || (seconds = 10.0)
@isdefined(threaded) || (threaded = true)
@isdefined(next_fill_function) || (next_fill_function = rand_next!)

elements isa Integer || error("elements must be an integer")
elements > 0 || error("elements must be positive")
seconds isa Real && isfinite(seconds) && seconds > 0 ||
    error("seconds must be finite and positive")

function measure_fill(next_fill_function, rng, values, seconds, threaded)
    expected_device = MLDataDevices.get_device_type(rng.device)
    actual_device = MLDataDevices.get_device_type(values)
    actual_device === expected_device ||
        error("expected $expected_device output, received $actual_device")
    backend = KernelAbstractions.get_backend(values)

    rng, _ = next_fill_function(rng, values; threaded)
    KernelAbstractions.synchronize(backend)
    GC.gc()

    runs = 0
    started = time_ns()
    elapsed = 0
    while elapsed < seconds * 1.0e9
        rng, _ = next_fill_function(rng, values; threaded)
        KernelAbstractions.synchronize(backend)
        runs += 1
        elapsed = time_ns() - started
    end

    elapsed_seconds = elapsed / 1.0e9
    values_per_second = runs * length(values) / elapsed_seconds
    output_gib_per_second = values_per_second * sizeof(eltype(values)) / 2.0^30
    milliseconds_per_fill = 1.0e3 * elapsed_seconds / runs
    return (;
        runs,
        elapsed_seconds,
        milliseconds_per_fill,
        values_per_second,
        output_gib_per_second,
    )
end

rng = family(12345) |> target_device
values = rand(rng, result_type, elements)
result = measure_fill(next_fill_function, rng, values, seconds, threaded)

@printf(
    "%s %s %s on %s: %.3f Gvalue/s, %.3f GiB/s output, %.3f ms/fill (%d runs)\n",
    next_fill_function,
    nameof(family),
    result_type,
    nameof(typeof(target_device)),
    result.values_per_second / 1.0e9,
    result.output_gib_per_second,
    result.milliseconds_per_fill,
    result.runs,
)

result
