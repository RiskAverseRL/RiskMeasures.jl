using Distributions

"""
    CVaR(x̃, α)

Compute the conditional value at risk at level `α` for the random variable `x̃`.

The risk level `α` must satisfy the ``α ∈ [0,1]``. Risk aversion increases with an increasing `α` and, `α = 0` represents the expectation,  `α = 1` computes the essential infimum (smallest value with positive probability).

Assumes a reward maximization setting and solves the dual form
``\\min_{q ∈ Q} q^T x̃``
where
``Q = {q ∈ Δ^n : q_i ≤ p_i/(1-α)}``
and ``Δ^n`` is the probability simplex. 

Returns a named tuple with `value` and the `distribution` that solves the robust
CVaR formulation such that ``\\mathbb{E}_{x̃\\sim pc}{x̃}`` equals to the CVaR value.

More details: https://en.wikipedia.org/wiki/Expected_shortfall
"""
function CVaR end

"""
    CVaR_e(x̃, α)

Compute the conditional value at risk at level `α` for the random variable `x̃`. Also
compute the equivalent random variable with the same support but a different distribution.
"""
function CVaR_e end



"""
    CVaR_e(values, pmf, α; ...) 

Compute CVaR for a discrete random variable with `values` and the probability mass
function `pmf`. See `CVaR(x̃, α)` for more details.
"""
function CVaR_e(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real;
                check_inputs = true) 
    _check_α(α)
    check_inputs && _check_pmf(values, pmf)
    
    T = eltype(pmf)

    # handle special cases
    if iszero(α)
        return (value = values'* pmf, pmf = Vector(pmf))
    elseif isone(α)
        minval = essinf_e(values, pmf; check_inputs = false)
        minpmf = zeros(T, length(pmf))
        minpmf[minval.index] = one(T)
        return (value = minval.value, pmf = minpmf)
    end
    
    # Here on: α ∈ (0,1)
    pc = zeros(T, length(pmf))  # this is the new distribution
    value = zero(T)                  # CVaR value
    p_left = one(T)           # probabilities left for allocation
    α̂ = one(α) - α                      # probabilities to allocate

    # Efficiency note: sorting by values is O(n*log n);
    # quickselect is O(n) and would suffice but would need be based on quantile
    sortedi = sort(eachindex(values, pmf); by=(i->@inbounds values[i]))
    @inbounds for i ∈ sortedi
        # update index's probability and probability left to sum to 1.0
        increment = min(pmf[i] / α̂, p_left)
        # update  return values
        pc[i] = increment
        value += increment * values[i]
        p_left -= increment
        p_left ≤ zero(p_left) && break 
    end
    return (value = value, pmf = pc)
end

# Definition for DiscreteNonparametric
CVaR(x̃, α::Real; kwargs...) =
    CVaR_e(support(x̃), probs(x̃), α; kwargs...).value

function CVaR_e(x̃, α::Real; kwargs...)
    v1 = CVaR_e(support(x̃), probs(x̃), α; kwargs...)
    ỹ = DiscreteNonParametric(support(x̃), v1.pmf)
    (value = v1.value, solution = ỹ)
end
