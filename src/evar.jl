import Optim: optimize, Brent


"""
    evar(X, p::Distribution, α; β_max = 100.)

Compute the EVaR risk measure of the random variable `X` represented as a vector over
outcomes and distributed according to the measure `p` with risk level `α`. The risk
level `α` should be in [0,1].

When `α = 0`, the function computes the expected value, and when `α = 1`, then the 
function computes the essential infimum (a minimum value with positive probability).

The function solves 
``\\sup_{β > 0} \\operatorname{erm}_β (X) + β^{-1} \\log (1-α)``.
The optimization over the value of `β` is upper bounded by `β_max`. Large
values of `β_max` may cause the computation to overflow.

Returns the evar and the value β that achieves the supremum above.

See:
Ahmadi-Javid, A. “Entropic Value-at-Risk: A New Coherent Risk Measure.” Journal of Optimization Theory and Applications 155(3), 2012.
"""
function evar(X::AbstractVector{<:Real}, p::Distribution{<:Real}, α::Real; β_max = 100.)
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    length(X) > 0 || _bad_risk("X must have some elements")

    if iszero(α)
        (evar = sum(X .* p.p) |> float, β=0.)
    elseif isone(α)
        (evar = minimum(@inbounds X[i] for i ∈ 1:length(X) if !iszero(p.p[i])) |> float,
        β = β_max)
    elseif zero(α) ≤ α ≤ one(α)
        f(β) = -erm(X, p, β) - 1.0/β*log(1-α)
        sol = optimize(f, 0., β_max, Brent())
        sol.converged || _bad_risk("Failed to find optimal β (unknown reason).")
        (evar = -float(sol.minimum), β = float(sol.minimizer) )
    else
        _bad_risk("Parameter β must be in [0,1]")
    end
end

evar(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real; kwargs...) where {T<:Real} =
    evar(X, Distribution{T}(p), α; kwargs...)

evar(X::AbstractVector{<:Real}, α::Real; kwargs...) =
    evar(X, uniform(length(X)), α; kwargs...) 
