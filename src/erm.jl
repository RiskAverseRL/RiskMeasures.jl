"""
    erm(X, p::Distribution, α)

Compute the entropic risk measure of the random variable `X` represented as 
a vector over outcomes and distributed according to the measure `p` with risk level `α`. 

Assumes a maximization problem. Using `α=0` computes expectation
and `α=Inf` computes the essential infimum (smallest value with positive probability).

More details: https://en.wikipedia.org/wiki/Entropic_risk_measure
"""
function erm(X::AbstractVector{<:Real}, p::Distribution{<:Real}, α::Real)
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    length(X) > 0 || _bad_risk("X must have some elements")
    if iszero(α)
        sum(X .* p.p) |> float
    elseif isinf(α) && α > zero(α)
        minimum(@inbounds X[i] for i ∈ 1:length(X) if !iszero(p.p[i])) |> float
    elseif zero(α) ≤ α 
        # because entropic risk measure is translation equivariant, we can change X so that
        # it is positive. That change makes it less likely that it overflows
        Xmin = minimum(X) #  TODO: only needs to look at X with pos prob. `
        Xmin - one(α) / α * log(sum(p.p .* exp.(-α .* (X .- Xmin) ))) |> float
    else
        _bad_risk("Risk level α must be non-negative.")
    end
end

erm(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real) where {T<:Real} =
    erm(X, Distribution{T}(p), α)

erm(X::AbstractVector{<:Real}, α::Real) = erm(X, uniform(length(X)), α)
