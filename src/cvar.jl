using Distributions

"""
    CVaR(x̃, α)

Compute the conditional value at risk at level `α` for the random variable `x̃`.

The risk level `α` must satisfy the ``α ∈ [0,1]``. Risk aversion descreses with an increasing `α` and, `α = 1` represents the expectation,
`α = 0` computes the essential infimum (smallest value with positive probability).

Assumes a reward maximization setting and solves the dual form
```math
\\min_{q ∈ \\mathcal{Q}} q^T x̃
```
where
```math
\\mathcal{Q} = \\left\\{q ∈ Δ^n : q_i ≤ \\frac{p_i}{α}\\right\\}
```
and ``Δ^n`` is the probability simplex, and ``p`` is the distribution of ``x̃``. 


More details: https://en.wikipedia.org/wiki/Expected_shortfall
"""
function CVaR end

"""
    CVaR(x̃, α)

Compute the conditional value at risk at level `α` for the random variable `x̃`. Also
compute the equivalent random variable with the same support but a different distribution.

Returns a named tuple with `value` and the `solution` random variable that solves the robust
CVaR formulation. That is if `ỹ` is the solution, then the support of `x̃` and `ỹ` are the same
and  ``\\mathbb{E}[ỹ] = \\operatorname{CVaR}_{α}[x̃]``.
"""
function CVaR end



"""
    CVaR(values, pmf, α; ...) 

Compute CVaR for a discrete random variable with `values` and the probability mass
function `pmf`. See `CVaR(x̃, α)` for more details.
"""
function CVaR(values::AbstractVector{<:Real}, pmf::AbstractVector{<:Real}, α::Real; check_inputs=true)
    check_inputs && _check_α(α)
    check_inputs && _check_pmf(values, pmf)

    T = eltype(pmf)

    # handle special cases
    if isone(α)
        return (value=values' * pmf, pmf=Vector(pmf))
    elseif iszero(α)
        minval = essinf(values, pmf; check_inputs=false)
        minpmf = zeros(T, length(pmf))
        minpmf[minval.index] = one(T)
        return (value=minval.value, pmf=minpmf)
    end

    # Here on: α ∈ (0,1)
    pc = zeros(T, length(pmf))  # this is the new distribution
    value = zero(T)                  # CVaR value
    p_left = one(T)           # probabilities left for allocation
    α̂ = α                      # probabilities to allocate

    # Efficiency note: sorting by values is O(n*log n);
    # quickselect is O(n) and would suffice but would need be based on quantile
    sortedi = sortperm(values)
    @inbounds for i ∈ sortedi
        # update index's probability and probability left to sum to 1.0
        increment = min(pmf[i] / α̂, p_left)
        # update  return values
        pc[i] = increment
        value += increment * values[i]
        p_left -= increment
        p_left ≤ zero(p_left) && break
    end
    return (value=value, pmf=pc)
end

function CVaR(x̃, α::Real; kwargs...)
    supp, pmf = rv2pmf(x̃)
    v1 = CVaR(supp, pmf, α; kwargs...)
    ỹ = DiscreteNonParametric(supp, v1.pmf)
    (value=v1.value, pmf=ỹ)
end
