"""
    VaR(x̃, α) 

Compute the value at risk at risk level `α` for the random variable `x̃`.

Risk must satisfy ``α ∈ [0,1]`` and `α=0.5` computes the median and `α=1` computes the 
essential infimum (smallest value with positive probability) and `α=0` returns infinity.

Solves for
``\\inf \\{x ∈ \\mathbb{R} : \\mathbb{P}[x̃ ≤ x] > 1-α \\}``

In general, this function is neither convex nor concave.
"""
function VaR end

# TODO: need to make sure that we are solving for
#``\\inf \\{x ∈ \\mathbb{R} : \\mathbb{P}[x̃ ≤ x] > 1-α \\}
# with a strict inequality; check on Bernoulli distribution

"""
    VaR_e(values, pmf, α; ...) 

Compute VaR for a discrete random variable with `values` and the probability mass
function `pmf`. See `VaR(x̃, α)` for more details. Also compute the index that achieves 

Runs in ``n \\log(n)`` time where `n = length(x̃)`.

Also returns the value VaR and an index `i` such that `values[i] = x` in the minimization above. If such an index does not exist, then returns -1.
"""
function VaR_e(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real;
               check_inputs = true) 

    _check_α(α)
    check_inputs && _check_pmf(values, pmf)
    
    T = eltype(pmf)
    # special cases
    if isone(α) # minimum
        return essinf_e(value, pmf, α; check_inputs = check_inputs)
    elseif iszero(α) # maximum (it is unbounded)
        return (value = typemax(Tval), index = -1)
    end

    # Efficiency note: sorting by values is O(n*log n); quickselect is O(n) and would suffice
    # descending sort (to make sure that VaR is optimistic as it should be)
    sortedi = sort(eachindex(x̃, pmf); by=(i->@inbounds -x̃[i]))

    pos = last(sortedi) # this value is used when the loop does not break
    p_accum = zero(T) 

    α̂ = α - 1e-10  # ...numerical issues
    # find the index such that the sum of the probabilities is greater than alpha
    @inbounds for i ∈ sortedi
        p_accum += pmf[i]
        p_accum ≥ α̂ && (pos = i; break)
    end

    return (value = values[pos], index = pos)
end


VaR(x̃::DiscreteNonParametric, α::Real; kwargs...) =
    VaR_e(support(x̃), probs(x̃), α; kwargs...).value
