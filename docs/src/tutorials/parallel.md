# Parallel jobs

Assign a stream to each logical job, then let Julia schedule jobs freely.
This avoids shared mutable state and dependence on thread IDs.

```@example parallel
using PureRNGs, Random
using Base.Threads

function job_mean(root, job)
    rng = subrng(root, job)
    buffer = Vector{Float64}(undef, 1024)
    _, rng = randn_next!(rng, buffer)
    return sum(buffer) / length(buffer)
end

root = Threefry4x32(123456)
results = zeros(8)

Threads.@threads for job in eachindex(results)
    results[job] = job_mean(root, job)
end

serial = [job_mean(root, job) for job in eachindex(results)]
@assert results == serial
results
```

The root uses a 128-bit key to leave room for many derived job keys.
Inner fills run serially by default, so the outer loop owns all the parallelism.

Job IDs must be integers in `0:typemax(UInt64)`, excluding `Bool`.
`subrng` rejects IDs outside that domain instead of wrapping them to another job's ID.

For nested work, `subrng(root, job, batch, purpose)` derives a child for the job,
then a child for the batch, then a child for the purpose. It equals three nested
single-ID calls. The IDs and their order define the path.

If a job needs several batches, continue its local generator with `randn_next!`.
Do not derive the same job key again for each batch unless repetition is intentional.

Changing the thread count preserves each job's input stream.
Changing a floating-point reduction order can still change the final computed statistic.

## Derive from the job index, not from a running generator

Deriving inside a loop that also advances the parent gives the same job stream
every iteration, because derivation ignores the parent's position:

```@example parallel
function repeated_jobs(rng, n)
    means = Float64[]
    for _ in 1:n
        child, _ = splitrng(rng)
        values, child = rand_next(child, Float64, 16)
        push!(means, sum(values) / 16)
        _, rng = rand_next(rng, Float64)
    end
    return means
end

repeated_jobs(Threefry4x32(123456), 3)
```

The loop index is the purpose ID, so derive from it directly:

```@example parallel
function indexed_jobs(root, n)
    return [sum(first(rand_next(subrng(root, i), Float64, 16))) / 16 for i in 1:n]
end

job_means = indexed_jobs(Threefry4x32(123456), 3)
@assert allunique(job_means)
job_means
```
