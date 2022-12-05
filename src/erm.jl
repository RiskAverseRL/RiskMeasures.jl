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

erm(X::AbstractVector{<:Real}, α::Real) = erm(X, uniform(length(X)), α)
erm(X::AbstractVector{<:Real}, p, α) = erm(X, p, α, minimum(X))
erm(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real, Xmin::Real) where {T<:Real} =
    erm(X, Distribution{T}(p), α, Xmin)

function erm(X::AbstractVector{<:Real}, p::Distribution{<:Real}, α::Real, Xmin::Real)
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    length(X) > 0 || _bad_risk("X must have some elements")
    if iszero(α)
        mean(X, p) |> float
    elseif isinf(α) && α > zero(α)
        minimum(@inbounds X[i] for i ∈ 1:length(X) if !iszero(p.p[i])) |> float
    elseif zero(α) ≤ α 
        # because entropic risk measure is translation equivariant, we can change X so that
        # it is positive. That change makes it less likely that it overflows
        if isfinite(Xmin)
            #@fastmath Xmin-one(α) / α*log(sum(p.p .* exp.(-α .* (X .- Xmin) ))) |> float
            a = Iterators.map((x,p) -> (@fastmath p * exp(-α * (x - Xmin))), X, p.p) |> sum
            @fastmath Xmin - one(α) / α * log(a) |> float
        else
            NaN |> float
        end
    else
        _bad_risk("Risk level α must be non-negative.")
    end
end



"""
    softmin(X, p::Distribution, α [, Xmin])

Compute a weighted softmin function for random variable `X` represented as 
a vector over outcomes and distributed according to the measure `p` with risk level `α`. 
This can be seen as an approximation of the arg min function and not the min function.

The operator computes a distribution p such that
`` p_i = \frac{e^{-α x_i}}{E[e^{-α X}]} ``

The optional `Xmin` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `X` is used instead. 

The value α must be positive
"""
function softmin end

softmin(X, p, α) = softmin(X, p, α, minimum(X))
softmin(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real) where {T<:Real} =
    softmin(X, Distribution{T}(p), α)
softmin(X::AbstractVector{<:Real}, α::Real) = softmin(X, uniform(length(X)), α)

function softmin(X::AbstractVector{<:Real}, p::Distribution{T}, α::Real, Xmin::Real) where
    {T <: Real}
    
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
            mapreduce(isfinite, &, np) || error("Overflow, reduce α.")
            np
        else
            error("Input must be finite.")
        end
    else
        _bad_risk("Risk level α must be positive.")
    end
end


"""
    softmintrue(X, p::Distribution, α [, Xmin])

Compute a weighted appriximation of the min function for random variable `X` represented as 
a vector over outcomes and distributed according to the measure `p` with risk level `α`. 

The operator computes a distribution p such that
`` \frac{E[-α X * e^{-α x_i}]}{E[e^{-α X}]} ``

The optional `Xmin` parameter is used as an offset in order to avoid overflows
when computing the exponential function. If not provided, the minimum value
of `X` is used instead. 

The value α must be positive
"""
function softmintrue end

softmintrue(X, p, α) = softmintrue(X, p, α, minimum(X))
softmintrue(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real) where {T<:Real} =
    softmintrue(X, Distribution{T}(p), α)
softminture(X::AbstractVector{<:Real}, α::Real) = softmintrue(X, uniform(length(X)), α)

function softmintrue(X::AbstractVector{<:Real}, p::Distribution{T}, α::Real, Xmin::Real) where
    {T <: Real}
    pn = softmin(X, p, α, Xmin)
    sum(pn .* (-α * X))
end
