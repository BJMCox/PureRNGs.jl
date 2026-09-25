@inline _cooperative_value(::Val{:uniform}, ::Type{T}, raw) where {T} = _from_bits(T, raw)
@inline _fill_width(::Val{:uniform}, ::Type{T}) where {T} = _draw_bits(T)

# The core never names a KernelAbstractions type, so a host destination
# resolves to the package's own token. The KernelAbstractions extension adds the
# device method, which returns the launchable backend.
@inline _fill_backend(::_CPUBackend, destination) = _CPU_BACKEND

# Indexing by an index vector the package built, so every index is in bounds.
# The KernelAbstractions extension gives GPU backends kernels for both, since
# GPUArrays' indexing reads the index extrema back to the host to check them.
_gather(backend, source, indices) = source[indices]
function _scatter!(backend, destination, indices, source)
    destination[indices] = source
    return destination
end

# Calls `f(destination, column, args...)` for every column of a matrix. The
# host runs the columns in order or on threads, and the KernelAbstractions
# extension gives GPU backends a workitem per column.
function _foreach_column!(::_CPUBackend, f, destination, threaded::Bool, args...)
    if threaded
        Threads.@threads for column in axes(destination, 2)
            f(destination, column, args...)
        end
    else
        for column in axes(destination, 2)
            f(destination, column, args...)
        end
    end
    return destination
end

# Every device fill runs a KernelAbstractions kernel, so the extension owns all
# methods of this launcher.
function _launch_device_fill! end
