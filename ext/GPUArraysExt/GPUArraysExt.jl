module GPUArraysExt

import GPUArrays: AbstractGPUArray, @allowscalar
import MadDiff: onehot!

function onehot!(x::AbstractGPUArray{T}, i) where {T}
    fill!(x, zero(T))
    @allowscalar x[i] = one(T)
end

end  # module