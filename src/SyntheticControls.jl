module SyntheticControls

using Distributions
using Optim
# using StatsBase
using ProgressMeter
using StatsPlots, RecipesBase
using LinearAlgebra

"""
    Weightings{M<:AbstractMatrix}

Wrapper for a matrix of type `M`. Rows correspond to candidate controls, columns correspond to synthetic controls (and therefore sum to one).
"""
struct Weightings{M<:AbstractMatrix}
    w::M
    function Weightings(w::M) where M
        w0 = sum(w, dims=1)
        all(>=(0), w) || error("∃ at least one negative weight in at least one synthetic control.")
        return new{M}(w./w0)
    end
end
Weightings(ws::AbstractVector{T}) where T<:AbstractVector = Weightings(hcat(ws...))

Base.getindex(scw::Weightings, i) = selectdim(scw.w, 2, i)
function Base.setindex!(scw::Weightings, v, i)
    selectdim(scw.w, 2, i) .= v
end
Base.firstindex(scw::Weightings) = 1
Base.lastindex(scw::Weightings) = size(scw.w, 2)
Base.iterate(scw::Weightings, state...) = iterate(eachcol(scw.w), state...)
Base.length(scw::Weightings) = size(scw.w, 2)
Base.IteratorEltype(::Type{T}) where T<:Weightings = Base.EltypeUnknown()


struct _Dirichlet{D <: Dirichlet}
    distribution::D
end
Distributions.rand(p::_Dirichlet, n::Integer) = Weightings(rand(p.distribution, n))
Distributions.rand(p::_Dirichlet) = rand(p.distribution)
Distributions.pdf(p::_Dirichlet, scw::Weightings) = pdf(p.distribution, scw.w)
Distributions.logpdf(p::_Dirichlet, scw::Weightings) = logpdf(p.distribution, scw.w)
Distributions.logpdf(p::_Dirichlet, w::AbstractVector) = logpdf(p.distribution, w)
Distributions.insupport(p::_Dirichlet, w) = insupport(p.distribution, w)
Distributions.insupport(p::_Dirichlet, scw::Weightings) = insupport(p, scw.w)
(p::_Dirichlet)(scw::Weightings) = pdf(p, scw)
number_controls(p::_Dirichlet) = length(p.distribution)


function Prior(α::AbstractVector)
    αhat = max_entropy_Dirichlet_parameter(α)
    distribution = Dirichlet(αhat)
    return _Dirichlet(distribution)
end
function Prior(dim::Integer)
    distribution = Dirichlet(dim, 1.0)
    return _Dirichlet(distribution)
end
function max_entropy_objective(α::AbstractVector)
    F = function (λ)
        return -entropy(Dirichlet(λ .* α))
    end
    return F
end
function max_entropy_Dirichlet_parameter(α::AbstractVector)
    lower = 1.0 / maximum(α)
    upper = 1.0 / minimum(α)
    res = optimize(max_entropy_objective(α), lower, upper)
    λ = Optim.minimizer(res)
    return λ .* α
end

struct LogLikelihood{M<:AbstractMatrix, V<:AbstractVector}
    X_control::M
    X_intervention::V
    function LogLikelihood(X_control::M, X_intervention::V) where {M, V}
        d1 = size(X_control, 2)
        d2 = length(X_intervention)
        d1 == d2 || error("Dimension mismatch: control has $(d1) and intervention has $(d2) covariates!")
        return new{M, V}(X_control, X_intervention)
    end
end
Base.length(ℓ::LogLikelihood) = length(ℓ.X_intervention)
number_controls(ℓ::LogLikelihood) = size(ℓ.X_control, 1)
number_covariates(ℓ::LogLikelihood) = size(ℓ.X_control, 2)


function (ℓ::LogLikelihood)(ws::AbstractMatrix, Σ)
    X_synthetic = permutedims(ℓ.X_control) * ws
    return logpdf(MvNormal(ℓ.X_intervention, Σ), X_synthetic)
end
(ℓ::LogLikelihood)(ws::Weightings, Σ) = ℓ(ws.w, Σ)



function Proposal(α::AbstractVector; scale=length(α), kwargs...)
    λ = scale / minimum(α)
    distribution = Dirichlet(λ .* α)
    return _Dirichlet(distribution)
end
Proposal(αs::AbstractVector{V}; kwargs...) where V<:AbstractVector = map(α -> Proposal(α; kwargs...), αs)
Proposal(ws::Weightings; kwargs...) = map(w -> Proposal(w; kwargs...), ws)


function _mcmc_propose(ws::Weightings; kwargs...)
    q_forwards = Proposal(ws; kwargs...)
    proposed_ws = Weightings(rand.(q_forwards))
    q_backwards = Proposal(proposed_ws; kwargs...)

    forward_step_logpdf = logpdf.(q_forwards, proposed_ws)
    backward_step_logpdf = logpdf.(q_backwards, ws)

    return proposed_ws, forward_step_logpdf, backward_step_logpdf
end

function _mcmc_sample_step!(ws, prior, ℓ_Σ; kwargs...)
    proposed_ws, forward_step_logpdf, backward_step_logpdf = _mcmc_propose(ws; kwargs...)
    α = exp.(
        logpdf(prior, proposed_ws)
        .- logpdf(prior, ws)
        .+ ℓ_Σ(proposed_ws)
        .- ℓ_Σ(ws)
        .+ backward_step_logpdf
        .- forward_step_logpdf
    )
    u = rand(size(α)...)
    accept_flags = u .< α
    for (w, proposed_w, accept_flag) in zip(ws, proposed_ws, accept_flags)
        accept_flag && (w .= proposed_w)
    end
    return mean(accept_flags)
end
function _mcmc_sample_step!(ws, prior, ℓ_Σ, numIter::Integer; progress_meter = Progress(numIter; dt=1), info=(), kwargs...)
    acceptance_rate = map(1:numIter) do _
        ProgressMeter.next!(progress_meter; showvalues=info)
        _mcmc_sample_step!(ws, prior, ℓ_Σ; kwargs...)
    end
    return acceptance_rate
end
_mcmc_sample_step!(ws, prior, ℓ::LogLikelihood, Σ::AbstractMatrix, numIter...; kwargs...) = _mcmc_sample_step!(ws, prior, (_ws) -> ℓ(_ws, Σ), numIter...; kwargs...)



mutable struct SMCProblem{W<:Weightings, V<:AbstractVector, D<:_Dirichlet, L<:LogLikelihood, M<:AbstractMatrix}
    ws::W
    smc_logws::V
    prior::D
    loglikelihood::L
    Σ₀::M
    dΣ::Float64
    t::Int64
    function SMCProblem(ws::W, smc_logws::V, prior::D, loglikelihood::L, Σ₀::M, dΣ::Float64; kwargs...) where {W<:Weightings, V<:AbstractVector, D<:_Dirichlet, L<:LogLikelihood, M<:AbstractMatrix}
        number_controls(prior) == number_controls(loglikelihood) == size(ws.w, 1) || error("number_controls mismatch!")
        size(Σ₀, 1) == number_covariates(loglikelihood) ||error("number_covariates mismatch!")
        size(ws.w, 2) == length(smc_logws) || error("number_synthetic_controls mismatch!")
        isposdef(Σ₀) || error("Σ₀ is not positive definite!")
        0 < dΣ < 1 || error("Needs dΣ in (0,1)!")
        return new{W, V, D, L, M}(ws, smc_logws, prior, loglikelihood, Σ₀, dΣ, 0)
    end
end
function SMCProblem(ws::W, smc_logws::V, prior::D, loglikelihood::L, Σ₀::M; halflife::Int64=5, kwargs...) where {W<:Weightings, V<:AbstractVector, D<:_Dirichlet, L<:LogLikelihood, M<:AbstractMatrix} 
    dΣ = Float64(2^(-1/halflife))
    return SMCProblem(ws, smc_logws, prior, loglikelihood, Σ₀, dΣ; kwargs...)
end
function SMCProblem(ws::W, smc_logws::V, prior::D, loglikelihood::L, dΣ::Float64...; kwargs...) where {W<:Weightings, V<:AbstractVector, D<:_Dirichlet, L<:LogLikelihood} 
    X_synthetic = permutedims(loglikelihood.X_control)*ws.w
    Σ₀ = Symmetric(cov(X_synthetic, dims=2))
    return SMCProblem(ws, smc_logws, prior, loglikelihood, Σ₀, dΣ...; kwargs...)
end
function SMCProblem(N::Integer, prior::D, args...; kwargs...) where {D<:_Dirichlet}
    ws = rand(prior, N)
    smc_logws = zeros(N)
    return SMCProblem(ws, smc_logws, prior, args...; kwargs...)
end
Base.length(prob::SMCProblem) = length(prob.smc_logws)


function _smc_rescale!(prob::SMCProblem)
    scalor = maximum(prob.smc_logws)
    prob.smc_logws .-= scalor
    return nothing
end
function _smc_reweighting!(prob::SMCProblem, ℓₜ, ℓₜ_prev)
    prob.smc_logws .+= ℓₜ(prob.ws) .- ℓₜ_prev(prob.ws)
    _smc_rescale!(prob)
    return nothing
end
ESS(smc_ws::AbstractVector) = sum(smc_ws)^2 / sum(smc_ws.^2)
function ESS(prob::SMCProblem)
    ess = sum(exp, prob.smc_logws)^2 / sum(w -> exp(2*w), prob.smc_logws)
    return ess
end

function _smc_resample!(prob::SMCProblem)
    numParticles = length(prob)
    _smc_rescale!(prob)

    W = exp.(prob.smc_logws)
    W ./= sum(W)
    idx = rand(Categorical(W), numParticles)

    copy!(prob.ws.w, prob.ws.w[:, idx])
    prob.smc_logws .= 0.0
    return nothing
end

function _smc_step!(prob::SMCProblem, numMCMCIter::Integer, ℓₜ_prev = (W) -> zeros(length(W)); force_resample::Bool=false, kwargs...)
    Σₜ = ((prob.dΣ)^(prob.t)) .* prob.Σ₀
    ℓₜ = (W) -> prob.loglikelihood(W, Σₜ)

    _smc_reweighting!(prob, ℓₜ, ℓₜ_prev)
    (force_resample || ((2.0 * ESS(prob)) < length(prob))) && _smc_resample!(prob)
    _ = _mcmc_sample_step!(prob.ws, prob.prior, ℓₜ, numMCMCIter; kwargs...)
    
    prob.t += 1
    return ℓₜ
end

function _reset!(prob::SMCProblem)
    prob.t=0
    N = length(prob)
    prob.ws = rand(prob.prior, N)
    prob.smc_logws .= 0.0
    return nothing
end

function sample!(prob::SMCProblem; numMCMCIter::Integer, numGenerations::Integer, ℓₜ = (ws)->zeros(length(ws)), kwargs...)
    iszero(prob.t) || _reset!(prob)
    p = Progress((numMCMCIter+1)*(numGenerations+1); dt=0.5, enabled=true)
    progress_status(prob) = () -> [(:generation, prob.t), (:ESS, ESS(prob))]
    
    while prob.t < numGenerations
        ℓₜ = _smc_step!(prob, numMCMCIter, ℓₜ; progress_meter=p, info = progress_status(prob))
        ProgressMeter.next!(p; showvalues=progress_status(prob))
    end

    _smc_resample!(prob)
    _mcmc_sample_step!(prob.ws, prob.prior, ℓₜ, numMCMCIter; progress_meter=p, info=progress_status(prob), kwargs...)
    
    return ℓₜ
end

include("recipes.jl")

end # module
