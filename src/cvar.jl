
"""
    essinf(X,p)

Computes the essential infimum of the random variable
"""
function essinf(X::AbstractVector{<:Real}, p::Distribution{T}) where {T<:Real}
    minval = typemax(float(T)); minindex = -1
    @inbounds for i ∈ eachindex(X, p.p)
        (!iszero(p.p[i]) && X[i] < minval) &&
            (minval = float(X[i]); minindex = i)
    end
    pc = zeros(T, length(p)); pc[minindex] = one(T)
    (minval, Distribution{T}(pc))
end

"""
    cvar(X, p::Distribution, α) 

Compute the conditional value at risk at level `α` for the random variable `X` distributed 
according the measure `p` in ``n \\log(n)`` time where `n = length(X)`. 

Risk must satisfy ``α∈[0,1]`` and `α=0` computes expectation and `α=1` computes the 
essential infimum (smallest value with positive probability).

Assumes a reward maximization setting and solves the dual form
``\\min_{q ∈ Q} q^T X``
where
``Q = {q ∈ Δ^n : q_i ≤ p_i/(1-α)}``
and ``Δ^n`` is the probability simplex. 

Returns a distribution `pc` that solves the optimization above and satisfies 
that ``\\mathbb{E}_{X\\sim pc}{X}`` equals to the cvar value.

More details: https://en.wikipedia.org/wiki/Expected_shortfall
"""
function cvar(X::AbstractVector{T}, p::Distribution{T2}, α::Real) where
             {T<:Real,T2<:Real}
    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    zero(α) ≤ α ≤ one(α) || _bad_risk("Risk level α must be in [0,1].")

    # handle special cases
    if iszero(α)
        return (cvar = (mean(X, p.p) |> float), p = p)
    elseif isone(α)
        v,np = essinf(X,p)
        return (cvar = v, p = np)
    end

    # Now α ∈ (0,1)
    pc = zeros(T2, length(p))   # this is the new distribution
    cvarv = zero(float(T))   # CVaR value
    p_left = one(T2)         # probabilities left to allocate
    α̂ = one(α) - α

    # Efficiency note: sorting by X is O(n*log n); quickselect is O(n) and would suffice
    sortedi = sort(eachindex(X, p.p); by=(i->@inbounds X[i]))
    @inbounds for i ∈ sortedi
        # update index's probability and probability left to sum to 1.0
        increment = min(p.p[i] / α̂, p_left)
        # update  return values
        pc[i] = increment
        cvarv += increment * X[i]
        p_left -= increment
        p_left ≤ zero(T2) && break 
    end
    return (cvar=float(cvarv), p=Distribution{T2}(pc))
end


cvar(X::AbstractVector{<:Real}, p::AbstractVector{T}, α::Real) where {T<:Real} =
    cvar(X, Distribution{T}(p), α)

cvar(X::AbstractVector{<:Real}, α::Real) = cvar(X, uniform(length(X)), α)
