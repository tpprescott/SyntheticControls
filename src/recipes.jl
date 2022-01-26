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

@recipe function f(ws::Weightings, v::AbstractVector)
    seriestype --> :density
    X_synthetic_controls = permutedims(ws.w) * v
    X_synthetic_controls
end
@recipe function f(ws::Weightings, m::AbstractMatrix)
    seriestype --> :violin
    X_synthetic_controls = permutedims(ws.w) * m
    X_synthetic_controls
end
@recipe function f(ws::Weightings, ℓ::LogLikelihood)
    fontfamily --> "Helvetica"
    dim = number_covariates(ℓ)
    color_palette --> palette(:seaborn_pastel)
    xminorticks --> 1
    xlabel --> "Covariate dimension"
    ylabel --> "Value"
    title --> "Synthetic control ensemble vs Intervention unit"
    titlefontsize --> 11

    @series begin
        color --> permutedims(collect(1:dim))
        label --> ""
        ws, ℓ.X_control
    end
    color --> :black
    label --> "Intervention unit"
    seriestype --> :scatter
    markersize --> 5
    ℓ.X_intervention
end

