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


function (ℓ::LogLikelihood)(ws::AbstractMatrix, Ps::AbstractVector{<:AbstractMatrix})
    centered_X_synthetic = (permutedims(ℓ.X_control) * ws) .- ℓ.X_intervention
    d, c = size(centered_X_synthetic)
    length(Ps)==c || error("Dimension mismatch in number of synthetic controls")
    
    xs = eachcol(centered_X_synthetic)
    PPs = Symmetric.(Ps)
    logpdfs = map(zip(xs, PPs)) do (x, P)
        ((LinearAlgebra.logdet(P) - (d*log(2π))) - LinearAlgebra.dot(x,P,x))/2
    end
    return logpdfs
end
(ℓ::LogLikelihood)(ws::AbstractMatrix, P::AbstractMatrix) = ℓ(ws, fill(P, size(ws,2)))
(ℓ::LogLikelihood)(ws::Weightings, Ps_or_P) = ℓ(ws.w, Ps_or_P)


number_covariates(w::Wishart) = size(w,1)
struct BayesProblem{D<:_Dirichlet, W<:Wishart, L<:LogLikelihood}
    prior_ws::D
    prior_Ps::W
    loglikelihood::L
    function BayesProblem(prior_ws::D, prior_Ps::W, loglikelihood::L) where {D, W, L}
        number_controls(prior_ws) == number_controls(loglikelihood) || error("number_controls mismatch")
        number_covariates(prior_Ps) == number_covariates(loglikelihood) || error("number_covariates mismatch")
        return new{D, W, L}(prior_ws, prior_Ps, loglikelihood)
    end
end



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

function _mcmc_sample_step!(ws::Weightings, ℓ::LogLikelihood, prior::_Dirichlet; kwargs...)
    proposed_ws, forward_step_logpdf, backward_step_logpdf = _mcmc_propose(ws; kwargs...)
    α = exp.(
        logpdf(prior, proposed_ws)
        .- logpdf(prior, ws)
        .+ ℓ(proposed_ws)
        .- ℓ(ws)
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

function _mcmc_sample_step!(ws, ℓ, prior, numIter::Integer; progress_meter = Progress(numIter; dt=1), info=(), kwargs...)
    acceptance_rate = map(1:numIter) do _
        ProgressMeter.next!(progress_meter; showvalues=info)
        _mcmc_sample_step!(ws, ℓ, prior; kwargs...)
    end
    return acceptance_rate
end

function _update_smc_weighting!(smc_weights, synthetic_control_particles, ℓ_n, ℓ_n_minus_1)
    smc_weights .*= exp.(ℓ_n(synthetic_control_particles) .- ℓ_n_minus_1(synthetic_control_particles))
    smc_weights ./= sum(smc_weights)
    return nothing
end
ESS(smc_weights) = sum(smc_weights)^2 / sum(smc_weights.^2)

function _smc_resample!(synthetic_control_particles, smc_weights)
    numParticles = length(synthetic_control_particles)
    idx = rand(Categorical(smc_weights), numParticles)
    synthetic_control_particles.w .= synthetic_control_particles.w[:, idx]
    smc_weights .= 1.0/numParticles
    return nothing
end

function Distributions.sample(prob::BayesProblem, numParticles::Integer, numMCMCIter::Integer; kwargs...)
    numGenerations = length(prob.loglikelihood)
    
    synthetic_control_particles = rand(prob.prior, numParticles)
    smc_weights = ones(numParticles)
    smc_weights ./= numParticles

    ℓ_n = x -> zeros(length(x))

    p = Progress((numMCMCIter+1)*(numGenerations+1); dt=0.2, enabled=true)
    progress_status(generation, smc_weights, acceptance_rates) = () -> [(:generation, generation), (:ESS, ESS(smc_weights)), (:prev_gen_acceptance_rate, mean(acceptance_rates))]
    progress_status(generation, smc_weights, ::Nothing) = () -> [(:generation, generation), (:ESS, ESS(smc_weights))]
    acceptance_rates = nothing

    for generation in 1:numGenerations
        ℓ_n_minus_1 = ℓ_n
        ℓ_n = marginal(prob.loglikelihood, generation)
        _update_smc_weighting!(smc_weights, synthetic_control_particles, ℓ_n, ℓ_n_minus_1)
        (ESS(smc_weights) < numParticles/2) && (_smc_resample!(synthetic_control_particles, smc_weights))
        acceptance_rates = _mcmc_sample_step!(synthetic_control_particles, ℓ_n, prob.prior, numMCMCIter; progress_meter=p, info=progress_status(generation, smc_weights, acceptance_rates), kwargs...)
        ProgressMeter.next!(p; showvalues=progress_status(generation, smc_weights, acceptance_rates))
        # @info("Generation $(generation) has ESS of $(ESS(smc_weights)) and MCMC acceptance rate of $(mean(acceptance_rates))")
    end

    _smc_resample!(synthetic_control_particles, smc_weights)
    acceptance_rates = _mcmc_sample_step!(synthetic_control_particles, prob.loglikelihood, prob.prior, numMCMCIter; progress_meter=p, info=progress_status(numGenerations+1, smc_weights, acceptance_rates), kwargs...)
    @info("Final MCMC acceptance rate: $(mean(acceptance_rates))")
    return synthetic_control_particles
end

include("recipes.jl")

end # module
