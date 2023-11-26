"""
    VaR(x̃, p::Distribution, α) 

Compute the value at risk at risk level `α` for the random VaRiable `x̃` disctributed
according to the measure `p` in ``n \\log(n)`` time where `n = length(x̃)`. 

Risk must satisfy ``α∈[0,1]`` and `α=0.5` computes the median and `α=1` computes the 
essential infimum (smallest value with positive probability) and `α=0` computes
the essential supremum.

Solves for
``\\inf \\{x ∈ \\mathbb{R} : \\mathbb{P}[x̃ ≤ x] > 1-α \\}``

In general, this function is neither convex nor concave.

Returns the value VaR and an index `i` such that `x̃[i] = x` in the minimization above.
"""
function VaR(x̃::AbstractVector{T}, p::Distribution{T2}, α::Real) where
             {T<:Real,T2<:Real}

    # TODO: need to make sure that we are solving for
    #``\\inf \\{x ∈ \\mathbb{R} : \\mathbb{P}[x̃ ≤ x] > 1-α \\}
    # with a strict inequality

    
    
    # special cases
    if isone(α) # minimum
        minval = typemax(T); minindex = -1
        @inbounds for i ∈ eachindex(x̃, p.p)
            (!iszero(p.p[i]) && x̃[i] < minval) &&
                (minval = x̃[i]; minindex = i)
        end
        return (VaR = minval, p = minindex)
    elseif iszero(α) # maximum
        maxval = typemin(T); maxindex = -1
        @inbounds for i ∈ eachindex(x̃, p.p)
            (!iszero(p.p[i]) && x̃[i] > maxval) &&
                (maxval = x̃[i]; maxindex = i)
        end
        return (VaR = maxval, p = maxindex)
    end

    # Efficiency note: sorting by x̃ is O(n*log n); quickselect is O(n) and would suffice
    # descending sort (to make sure that VaR is optimistic as it should be)
    sortedi = sort(eachindex(x̃, p.p); by=(i->@inbounds -x̃[i]))

    pos = last(sortedi) # this value is used when the loop does not break
    p_accum = zero(T2) 

    α̂ = α - 1e-10  # numerical issues
    # find the index such that the sum of the probabilities is greater than alpha
    @inbounds for i ∈ sortedi
        p_accum += p.p[i]
        p_accum ≥ α̂ && (pos = i; break)
    end

    return (VaR = x̃[pos], index = pos)
end


VaR(x̃::AbstractVector{T1}, p::AbstractVector{T2}, α::Real) where {T1<:Real, T2<:Real} =
    VaR(x̃, Distribution{T2}(p), α)

VaR(x̃::AbstractVector{<:Real}, α::Real) = VaR(x̃, uniform(length(x̃)), α)
