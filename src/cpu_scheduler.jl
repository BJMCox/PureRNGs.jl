const _CPU_FILL_CHUNK_BITS = UInt64(4096 * 32)
const _CPU_FILL_MIN_WORKITEMS = 4

@inline function _dense_fill_chunk_elements(::Type{T}) where {T}
    raw = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_draw_bits(T)))
    group = _dense_fill_group(T)
    return raw - raw % group
end
@inline function _dense_fill_bounds(workitem::Int, count::Int, chunk_elements::Int)
    first = (workitem - 1) * chunk_elements + 1
    chunk_count = min(chunk_elements, count - first + 1)
    return first, first + chunk_count - 1
end

# Chunks are handed out dynamically so an unequal core mix does not idle the
# fast cores. Chunk boundaries fall on draw boundaries, so the stream is the
# same as the serial fill.
@inline function _run_chunks(body, count::Int, chunk_elements::Int)
    workitems = cld(count, chunk_elements)
    if workitems < _CPU_FILL_MIN_WORKITEMS
        body(1, count)
        return nothing
    end
    next = Threads.Atomic{Int}(1)
    tasks = Vector{Task}(undef, min(Threads.nthreads(), workitems))
    for t in eachindex(tasks)
        tasks[t] = Threads.@spawn begin
            while true
                workitem = Threads.atomic_add!(next, 1)
                workitem > workitems && break
                first, last = _dense_fill_bounds(workitem, count, chunk_elements)
                body(first, last)
            end
        end
    end
    foreach(wait, tasks)
    return nothing
end

const _SPLIT_CHUNK_CHILDREN = 4096
