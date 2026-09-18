@inline _cooperative_value(::Val{:uniform}, ::Type{T}, raw) where {T} = _from_bits(T, raw)
@inline _fill_width(::Val{:uniform}, ::Type{T}) where {T} = _draw_bits(T)

# [R37] the core never names a KernelAbstractions type, so a host destination
# resolves to the package's own token. The KernelAbstractions extension adds the
# device method, which returns the launchable backend.
@inline _fill_backend(::_CPUBackend, destination) = _CPU_BACKEND

# Every device fill runs a KernelAbstractions kernel, so the extension owns all
# methods of this launcher.
function _launch_device_fill! end
