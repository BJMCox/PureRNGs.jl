# Parallel jobs

Assign a stream to each logical job, then let Julia schedule jobs freely.
This avoids shared mutable state and dependence on thread IDs.

```@example parallel
using PureRNGs, Random
using Base.Threads

function job_mean(root, job)
    rng = subrng(root, job)
    buffer = Vector{Float64}(undef, 1024)
    randn!(rng, buffer; threaded=false)
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
The example disables inner fill threading because the outer loop owns parallelism.

If a job needs several batches, continue its local generator with `randn_next!`.
Do not derive the same job key again for each batch unless repetition is intentional.

Changing the thread count preserves each job's input stream.
Changing a floating-point reduction order can still change the final computed statistic.
