"""
    var(X, p::Distribution, α) 

Compute the value at risk at risk level `α` for the random variable `X` disctributed
according to the measure `p` in ``n \\log(n)`` time where `n = length(X)`. 

Risk must satisfy ``α∈[0,1]`` and `α=0.5` computes the median and `α=1` computes the 
essential infimum (smallest value with positive probability) and `α=0` computes
the essential supremum.

Assumes a reward maximization setting and solves for
``\\inf \\{x ∈ \\mathbb{R} : \\mathbb{P}[X ≤ x] ≥ 1-α \\}

In general, this function is neither convex nor concave.

Returns the value var and an index `i` such that `X[i] = x` in the minimization above.
"""
function var(X::AbstractVector{T}, p::Distribution{T2}, α::Real) where
             {T<:Real,T2<:Real}

    # TODO: should be solving for
    #``\\inf \\{x ∈ \\mathbb{R} : \\mathbb{P}[X ≤ x] > 1-α \\}
    # with a strict inequality
    

    length(X) == length(p) || _bad_distribution("Lengths of X and p must match.")
    zero(α) ≤ α ≤ one(α) || _bad_risk("Risk level α must be in [0,1].")

    # special cases
    if isone(α) # minimum
        minval = typemax(T); minindex = -1
        @inbounds for i ∈ eachindex(X, p.p)
            (!iszero(p.p[i]) && X[i] < minval) &&
                (minval = X[i]; minindex = i)
        end
        return (var = minval, p = minindex)
    elseif iszero(α) # maximum
        maxval = typemin(T); maxindex = -1
        @inbounds for i ∈ eachindex(X, p.p)
            (!iszero(p.p[i]) && X[i] > maxval) &&
                (maxval = X[i]; maxindex = i)
        end
        return (var = maxval, p = maxindex)
    end

    # Efficiency note: sorting by X is O(n*log n); quickselect is O(n) and would suffice
    sortedi = sort(eachindex(X, p.p); by=(i->@inbounds X[i]))

    pos = last(sortedi) # this value is used when the loop does not break
    p_accum = zero(T2) 
   
    α̂ = one(α) - α
    # find the index such that the sum of the probabilities is greater than alpha
    @inbounds for i ∈ sortedi
        p_accum += p.p[i]
        p_accum ≥ α̂ && (pos = i; break)
    end

    return (var = X[pos], index = pos)
end


var(X::AbstractVector{T1}, p::AbstractVector{T2}, α::Real) where {T1<:Real, T2<:Real} =
    var(X, Distribution{T2}(p), α)

var(X::AbstractVector{<:Real}, α::Real) = var(X, uniform(length(X)), α)
