struct _SensitivitySolverShim{T, S<:AbstractMadNLPSolver{T}, K<:AbstractKKTSystem{T}} <: AbstractMadNLPSolver{T}
    inner::S
    kkt::K
end

function Base.getproperty(s:: _SensitivitySolverShim, name::Symbol)
    name === :inner && return getfield(s, :inner)
    name === :kkt && return getfield(s, :kkt)
    return getproperty(getfield(s, :inner), name)
end

function Base.setproperty!(s:: _SensitivitySolverShim, name::Symbol, value)
    name === :inner && return setfield!(s, :inner, value)
    name === :kkt && return setfield!(s, :kkt, value)
    return setproperty!(getfield(s, :inner), name, value)
end
