"""
    erm(X, p::Distribution, α [, Xmin])

Compute the entropic risk measure of the random variable `X` represented as 
a vector over outcomes and distributed according to the measure `p` with risk level `α`. 

The optional `Xmin` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `X` is used instead. 

Assumes a maximization problem. Using `α=0` computes expectation
and `α=Inf` computes the essential infimum (smallest value with positive probability).

More details: https://en.wikipedia.org/wiki/Entropic_risk_measure
"""
function erm end

erm(X, p, α) = erm(X, p, α, minimum(X))

function erm(X::AbstractVector{<:Real}, p::Distribution{<:Real}, α::Real, Xmin::Real)
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    length(X) > 0 || _bad_risk("X must have some elements")
    if iszero(α)
        sum(X .* p.p) |> float
    elseif isinf(α) && α > zero(α)
        minimum(@inbounds X[i] for i ∈ 1:length(X) if !iszero(p.p[i])) |> float
    elseif zero(α) ≤ α 
        # because entropic risk measure is translation equivariant, we can change X so that
        # it is positive. That change makes it less likely that it overflows
        if isfinite(Xmin)
            #@fastmath Xmin-one(α) / α*log(sum(p.p .* exp.(-α .* (X .- Xmin) ))) |> float
            a = Iterators.map((x,p) -> (@fastmath p * exp(α * (x - Xmin))), X, p.p) |> sum
            @fastmath Xmin - one(α) / α * log(a) |> float
        else
            NaN
        end
    else
        _bad_risk("Risk level α must be non-negative.")
    end
end

erm(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real) where {T<:Real} =
    erm(X, Distribution{T}(p), α)

erm(X::AbstractVector{<:Real}, α::Real) = erm(X, uniform(length(X)), α)


"""
    softmin(X, p::Distribution, α [, Xmin])

Compute a weighted soft-min for random variable `X` represented as 
a vector over outcomes and distributed according to the measure `p` with risk level `α`. 

The optional `Xmin` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `X` is used instead. 

The risk level must satisfy α > 0
"""
function softmin end

softmin(X, p, α) = softmin(X, p, α, minimum(X))

function softmin(X::AbstractVector{<:Real}, p::Distribution{<:Real}, α::Real, Xmin::Real)
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    length(X) > 0 || _bad_risk("X must have some elements")

    if α > zero(α)
        #  TODO: only needs to look at X with pos prob.
        # because softmin is translation invariant add the subtract the smallest value
        # ensures that X ≥ 0
        if isfinite(Xmin)
            np = similar(p.p)
            np .= @fastmath p.p .* exp.(-α .* (X .- Xmin) )
            np .= @fastmath inv(sum(np)) .* np
        else
            error("Input must be finite.")
        end
    else
        _bad_risk("Risk level α must be positive.")
    end
end
