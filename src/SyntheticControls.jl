module SyntheticControls

using Distributions
using Optim
# using StatsBase
using ProgressMeter
using StatsPlots, RecipesBase

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

@recipe function f(ws::Weightings)
    fontfamily --> "Helvetica"
    bar_position --> :stack
    linecolor --> :match
    legend --> :none
    xticks --> []
    showaxis --> :y
    color_palette --> palette(:seaborn_dark, rev=true)

    W = permutedims(ws.w)
    Wbar = mean(W, dims=1)
    m, n = size(W)
    sorted_candidate_idx = sortperm(vec(Wbar))
    W_sorted_candidate = view(W, :, sorted_candidate_idx)
    
    @series begin
        bar_width --> m/10
        primary := false
        seriescolor --> permutedims(collect(1:n))
        RecipesBase.recipetype(:groupedbar, [-m/10], Wbar[:, sorted_candidate_idx])
    end

    xticks := ([-m/10, m/2], ["Mean", "Ensemble ($m synthetic controls)"])
    title --> "Synthetic Control Weightings"
    titlefontsize --> 11
    seriescolor --> permutedims(collect(1:n))
    bar_width --> 1
    sorted_syn_con_idx = sortperm(selectdim(W_sorted_candidate, 2, n), rev=true)
    W_sorted = view(W_sorted_candidate, sorted_syn_con_idx, :)

    return RecipesBase.recipetype(:groupedbar, W_sorted)
end

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

struct LogLikelihood{M<:AbstractMatrix, D<:MvNormal}
    X_control::M
    distribution::D
    function LogLikelihood(X_control::M, distribution::D) where {M, D}
        d1 = size(X_control, 2)
        d2 = length(distribution)
        d1 == d2 || error("Dimension mismatch: control data is size $(size(X_control)) and intervention data is dimension $(length(distribution)))!")
        return new{M,D}(X_control, distribution)
    end
end
LogLikelihood(X_control, X_intervention, Σ) = LogLikelihood(X_control, MvNormal(X_intervention, Σ))
Base.length(ℓ::LogLikelihood) = length(ℓ.distribution)
Distributions.mean(ℓ::LogLikelihood) = mean(ℓ.distribution)
Distributions.cov(ℓ::LogLikelihood) = cov(ℓ.distribution)

function marginal(ℓ, idx)
    μ = mean(ℓ.distribution)
    Σ = cov(ℓ.distribution)
    marginal_distribution = MvNormal(μ[idx], Σ[idx, idx])
    return LogLikelihood(selectdim(ℓ.X_control,2,idx), marginal_distribution)
end
marginal(ℓ, n::Integer) = marginal(ℓ, 1:n)
number_controls(ℓ::LogLikelihood) = size(ℓ.X_control, 1)
number_covariates(ℓ::LogLikelihood) = size(ℓ.X_control, 2)


function (ℓ::LogLikelihood)(w::AbstractArray)
    X_synthetic = permutedims(ℓ.X_control) * w 
    return logpdf(ℓ.distribution, X_synthetic)
end
(ℓ::LogLikelihood)(ws::Weightings) = ℓ(ws.w)


@recipe function f(ws::Weightings, ℓ::LogLikelihood)
    fontfamily --> "Helvetica"
    dim = length(ℓ.distribution)
    color_palette --> palette(:seaborn_pastel)
    xminorticks --> 1
    xlabel --> "Covariate dimension"
    ylabel --> "Value"
    title --> "Synthetic control ensemble vs Intervention unit"
    titlefontsize --> 11

    @series begin
        seriestype --> :violin
        color --> permutedims(collect(1:dim))
        label --> ""
        X_synthetic_controls = permutedims(ws.w) * ℓ.X_control
        X_synthetic_controls
    end
    color --> :black
    label --> "Intervention unit"
    seriestype --> :scatter
    markersize --> 5
    μ = mean(ℓ.distribution)
    μ
end



struct BayesProblem{P<:_Dirichlet, L<:LogLikelihood}
    prior::P
    loglikelihood::L
    function BayesProblem(prior::P, loglikelihood::L) where {P, L}
        number_controls(prior) == number_controls(loglikelihood) || error("number_controls mismatch")
        return new{P, L}(prior, loglikelihood)
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

end # module
