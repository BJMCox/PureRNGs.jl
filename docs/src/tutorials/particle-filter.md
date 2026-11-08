# Particle filter

A bootstrap particle filter combines normal draws, likelihood evaluation, and weighted resampling.
This tutorial starts with array operations on either CPU or CUDA, then fuses an entire small filter into one CUDA kernel.

## Model and resampling law

Consider a scalar state with Gaussian process and observation noise:

```math
x_0 \sim \mathcal N(0,1),\qquad
x_t = 0.9x_{t-1} + 0.3\epsilon_t,\qquad
y_t = x_t + 0.5\eta_t.
```

The noises are independent standard normals.
The target at time ``t`` is the filtering mean ``\mathbb E[x_t \mid y_{1:t}]``.
Each step propagates equally weighted particles, weights them by the observation likelihood, and resamples with replacement.
This is multinomial resampling, not sampling without replacement.
Compute the mean before resampling to avoid extra resampling noise.

## Reuse particle buffers

```@example particle_filter
using PureRNGs

function bootstrap_filter(root, observations, count; threaded = false)
    initial_rng = subrng(root, 1)
    propagation_rng = subrng(root, 2)
    resampling_rng = subrng(root, 3)
    particles, _ = randn_next(initial_rng, Float32, count; threaded)
    noise = similar(particles)
    resampled = similar(particles)
    weights = similar(particles, Float64)
    weighted = similar(weights)
    estimates = Vector{Float64}(undef, length(observations))

    for (t, observation) in enumerate(observations)
        _, propagation_rng = randn_next!(propagation_rng, noise; threaded)
        @. particles = 0.9f0 * particles + 0.3f0 * noise
        y = Float32(observation)
        @. weights = -0.5 * ((y - particles) / 0.5f0)^2
        peak = maximum(weights)
        @. weights = exp(weights - peak)
        @. weighted = particles * weights
        estimates[t] = sum(weighted) / sum(weights)
        _, resampling_rng = randsample_next!(
            resampling_rng,
            particles,
            weights,
            resampled;
            replace = true,
            threaded,
        )
        particles, resampled = resampled, particles
    end

    return (; estimates, particles, propagation_rng, resampling_rng)
end

observations = Float32[0.1, 0.4, -0.2, 0.3, 0.0, 0.2]
result = bootstrap_filter(Philox4x32(123456), observations, 65_536)
result.estimates
```

Use positive particle counts and finite observations in this example.
Subtracting the maximum log weight keeps the largest weight equal to one.
The omitted Gaussian normalization constant cancels from relative weights.

Separate purpose keys isolate initialization, propagation, and resampling.
Continue the returned propagation and resampling states at each step, rather than recreating their keys.
Repeating the function with the same root repeats the filter.

The destination must not alias the population during resampling.
Two particle buffers alternate that role.
Buffer reuse avoids particle-array allocation, but weight preparation still uses sampling scratch space.
The weights change at each observation, so a cached weight table would be stale.

## Check against an exact filter

This linear Gaussian model also has an exact Kalman filter:

```@example particle_filter
function kalman_means(observations)
    mean, variance = 0.0, 1.0
    estimates = Float64[]
    for y in observations
        mean *= 0.9
        variance = 0.9^2 * variance + 0.3^2
        gain = variance / (variance + 0.5^2)
        mean += gain * (y - mean)
        variance *= 1 - gain
        push!(estimates, mean)
    end
    return estimates
end

exact = kalman_means(observations)
maximum(abs.(result.estimates - exact))
```

The particle estimate has Monte Carlo error. Increasing the particle count reduces that error, not floating-point differences between backends.

## Run the array version on CUDA

The remaining examples require CUDA, MLDataDevices, and KernelAbstractions in your environment, plus an NVIDIA GPU.
They do not run in the CPU documentation build.

```julia
using CUDA, MLDataDevices

CUDA.allowscalar(false)
gpu_result = bootstrap_filter(Philox4x32(123456) |> CUDADevice(), observations, 262_144)
@assert gpu_result.particles isa CuArray
gpu_result.estimates
```

The random fill and particle broadcasts run on the GPU.
The observations and scalar estimates stay on the host.
Reductions and weighted-sampling validation can synchronize, so this version is not a single asynchronous kernel.
Within each step, all particles form one batch.
Steps remain sequential because each step uses the previous resampled population.

## Fuse the whole observation loop

Normal draws and likelihood arithmetic can share a kernel for any particle count.
Resampling also needs a reduction and a cumulative weight table across the entire population.
A normal CUDA block barrier cannot synchronize different blocks.
For a large filter, keep the global reduction and resampling stages separate.
Fusing those stages requires a different synchronization design, not just moving the loop into a kernel.

For a small filter that fits in one block, all stages can share memory and barriers.
The following kernel assigns one **complete filter** to each block and one particle to each thread.
It scans observations sequentially inside the kernel, while different filters run independently.
It does not partition one large filter into separately resampled islands.

The launch uses 256 particles per filter.
The reduction requires a power-of-two particle count, with exactly that many threads per block.
The shared arrays hold particles, a cumulative weight table, and weighted moments.

```julia
function fused_particle_filters!(
    estimates,
    final_particles,
    observations,
    root,
    ::Val{N},
) where {N}
    lane = Int(CUDA.threadIdx().x)
    filter = Int(CUDA.blockIdx().x)
    rng = subrng(root, filter)
    initial_rng = subrng(rng, 1)
    propagation_rng = subrng(rng, 2)
    resampling_rng = subrng(rng, 3)
    particles = CUDA.CuStaticSharedArray(Float32, N)
    cumulative = CUDA.CuStaticSharedArray(Float64, N)
    moments = CUDA.CuStaticSharedArray(Float64, N)
    particles[lane] = randn_at(initial_rng, Float32, lane)
    CUDA.sync_threads()

    for t in axes(observations, 1)
        address = (t - 1) * N + lane
        x = 0.9f0 * particles[lane] + 0.3f0 * randn_at(propagation_rng, Float32, address)
        logweight = -0.5 * ((observations[t, filter] - x) / 0.5f0)^2
        particles[lane] = x
        cumulative[lane] = logweight
        CUDA.sync_threads()

        stride = N ÷ 2
        while stride > 0
            if lane <= stride
                cumulative[lane] = max(cumulative[lane], cumulative[lane+stride])
            end
            CUDA.sync_threads()
            stride ÷= 2
        end
        peak = cumulative[1]
        CUDA.sync_threads() # All threads read the maximum before reusing its storage.
        weight = exp(logweight - peak)
        cumulative[lane] = weight
        moments[lane] = Float64(x) * weight
        CUDA.sync_threads()

        offset = 1
        while offset < N
            left_weight = lane > offset ? cumulative[lane-offset] : 0.0
            left_moment = lane > offset ? moments[lane-offset] : 0.0
            CUDA.sync_threads() # Read the previous scan level before any thread writes.
            cumulative[lane] += left_weight
            moments[lane] += left_moment
            CUDA.sync_threads()
            offset *= 2
        end
        total = cumulative[N]
        if lane == 1
            estimates[t, filter] = moments[N] / total
        end

        target = min(rand_at(resampling_rng, Float64, address) * total, prevfloat(total))
        lo, hi = 1, N
        while lo < hi
            mid = (lo + hi) ÷ 2
            if cumulative[mid] <= target
                lo = mid + 1
            else
                hi = mid
            end
        end
        selected = particles[lo]
        CUDA.sync_threads() # Finish every ancestor read before replacing particles.
        particles[lane] = selected
        CUDA.sync_threads()
    end

    final_particles[lane, filter] = particles[lane]
    return nothing
end

particle_count, filter_count = 256, 256
device_observations = CuArray(repeat(observations, 1, filter_count))
estimates = CUDA.zeros(Float64, length(observations), filter_count)
final_particles = CUDA.zeros(Float32, particle_count, filter_count)
root = Threefry4x32(123456) |> CUDADevice()
CUDA.@cuda threads=particle_count blocks=filter_count fused_particle_filters!(
    estimates,
    final_particles,
    device_observations,
    root,
    Val(particle_count),
)
CUDA.synchronize()
```

This is one kernel launch for all filters and observations.
No normal array or global weight table exists.
Addressed draws use a distinct index for each particle and time, without mutable per-thread RNG state.
The supplied root stays unchanged. Derive a fresh purpose key before an independent repeat.

Here all filters see the same observations but use different purpose keys.
That lets us estimate Monte Carlo uncertainty across independent filter runs:

```julia
host_estimates = Array(estimates)
average = sum(host_estimates; dims = 2) / filter_count
mcse = sqrt.(
    sum((host_estimates .- average) .^ 2; dims = 2) / (filter_count * (filter_count - 1)),
)
(; mean = vec(average), mcse = vec(mcse), exact = kalman_means(observations))
```

The mean across small filters is not equivalent to one filter with their combined particle count.
Its Monte Carlo standard error does not measure finite-particle bias.
For different series, put each series in a column of `device_observations` instead.

This example owns its parallel cumulative scan and categorical lookup.
It uses the same multinomial resampling law, but does not promise the exact stream mapping of `randsample_next!`.
The tree scan, GPU arithmetic, and different particle counts also prevent bitwise comparison with the array version.
Use the Kalman solution to check the statistical target.

One block-sized filter does not occupy a whole GPU.
This design suits batches of small filters, not an arbitrary large population.
Benchmark the complete workload, including its final synchronization, before choosing either path.

## References

Doucet, Arnaud, and Adam M. Johansen (2011).
“A Tutorial on Particle Filtering and Smoothing: Fifteen Years Later.”
In *The Oxford Handbook of Nonlinear Filtering*, edited by Dan Crisan and Boris Rozovskii.
Oxford University Press.
[Author manuscript](https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf).

CUDA.jl contributors. “Kernel programming.” *CUDA.jl documentation*.
[Shared memory, synchronization, and kernel launches](https://cuda.juliagpu.org/stable/development/kernel/).
